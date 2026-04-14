"""Agent factory — centralized runtime assembly.

``AgentFactory`` resolves prompts, tools, subagents, and LLMs into a
compiled LangGraph graph.  This decouples agent construction from the
CLI session and from the compatibility shim in ``relay.graph``.

The factory supports two modes:

1. **Config-driven** (Phase 4+): pass a ``ConfigRegistry`` to resolve
   agent definitions from YAML files.
2. **Hardcoded fallback**: when no registry is provided, builds the
   same graph as earlier phases using Python-defined tool surfaces and
   prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from relay.agents.context import AgentContext
from relay.agents.deep_agent import create_deep_agent
from relay.agents.state import AgentState
from relay.prompt import COORDINATOR_PROMPT, EXPLORER_PROMPT, WORKER_PROMPT
from relay.settings import get_settings
from relay.tools.impl.filesystem import FILE_SYSTEM_TOOLS, glob_files, grep_files, ls, read_file
from relay.tools.internal.memory import MEMORY_TOOLS
from relay.tools.subagents import SubAgentRuntime
from relay.tools.impl.terminal import TERMINAL_TOOLS
from relay.tools.internal.todo import TODO_TOOLS
from relay.tools.impl.web import WEB_TOOLS

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from relay.configs.agent import AgentConfig as DeclAgentConfig
    from relay.configs.agent import SubAgentConfig as DeclSubAgentConfig
    from relay.configs.registry import ConfigRegistry


# ==============================================================================
# Tool Surface Definitions
# ==============================================================================
#
# Read-only filesystem tools shared by the coordinator and explorer.
# Mutating tools (write, edit, delete, terminal) are reserved for the
# general-purpose subagent.

_READ_ONLY_FILE_TOOLS: list = [read_file, glob_files, grep_files, ls]

# All implementation tools available for full-access subagents.
_ALL_IMPL_TOOLS: list = [*FILE_SYSTEM_TOOLS, *TERMINAL_TOOLS, *WEB_TOOLS]


def _prepare_tools(tools: list) -> list:
    """Enable ToolException-to-ToolMessage behaviour on each tool."""
    for tool_obj in tools:
        tool_obj.handle_tool_error = True
    return tools


# ==============================================================================
# Tool Pattern Resolution (simplified)
# ==============================================================================
#
# Full pattern matching with wildcards and negation will arrive in Phase 5
# with the tool factory.  For now we map well-known pattern prefixes to
# the hardcoded tool lists.

_TOOL_GROUPS: dict[str, list] = {
    "impl:web": WEB_TOOLS,
    "impl:file_system:read_file": [read_file],
    "impl:file_system:glob_files": [glob_files],
    "impl:file_system:grep_files": [grep_files],
    "impl:file_system:ls": [ls],
    "impl:file_system": FILE_SYSTEM_TOOLS,
    "impl:terminal": TERMINAL_TOOLS,
    "impl": _ALL_IMPL_TOOLS,
    "internal:memory": MEMORY_TOOLS,
    "internal:todo": TODO_TOOLS,
    "internal": [*MEMORY_TOOLS, *TODO_TOOLS],
}


def _resolve_tool_patterns(patterns: list[str]) -> list:
    """Resolve declarative tool patterns into tool objects.

    Pattern matching is deliberately coarse for now — we match the
    longest prefix of each pattern against ``_TOOL_GROUPS``.  A full
    glob/negation engine comes with the tool factory in Phase 5.
    """
    seen_names: set[str] = set()
    tools: list = []

    for pattern in patterns:
        # Strip trailing wildcards for prefix matching.
        key = pattern.rstrip(":*")
        matched = _TOOL_GROUPS.get(key)

        if matched is None:
            # Try progressively shorter prefixes.
            parts = key.split(":")
            while parts and matched is None:
                parts.pop()
                matched = _TOOL_GROUPS.get(":".join(parts)) if parts else None

        if matched:
            for t in matched:
                if t.name not in seen_names:
                    seen_names.add(t.name)
                    tools.append(t)

    return tools


# ==============================================================================
# Hardcoded Subagent Definitions (fallback when no registry is provided)
# ==============================================================================


def _explorer_config() -> SubAgentRuntime:
    return SubAgentRuntime(
        name="explorer",
        description=(
            "Deep codebase investigation — reads files, searches for "
            "patterns, and synthesises findings.  Use for questions "
            "about project structure, finding code, or understanding "
            "how things work."
        ),
        tools=[*_READ_ONLY_FILE_TOOLS, *WEB_TOOLS, *MEMORY_TOOLS],
        prompt=EXPLORER_PROMPT,
    )


def _worker_config() -> SubAgentRuntime:
    return SubAgentRuntime(
        name="general-purpose",
        description=(
            "Complex multi-step execution — file editing, shell "
            "commands, and sustained work.  Use for implementing "
            "features, fixing bugs, refactoring, or any task that "
            "requires changes."
        ),
        tools=[
            *FILE_SYSTEM_TOOLS,
            *TERMINAL_TOOLS,
            *WEB_TOOLS,
            *MEMORY_TOOLS,
            *TODO_TOOLS,
        ],
        prompt=WORKER_PROMPT,
    )


# ==============================================================================
# Agent Factory
# ==============================================================================


class AgentFactory:
    """Resolves tools, subagents, and LLMs into a compiled agent graph.

    The factory owns the policy decisions about which tools go to the
    coordinator versus subagents, how the middleware stack is ordered
    (via ``create_deep_agent`` → ``create_react_agent``), and which
    model to use.

    When *registry* is provided, agent definitions come from YAML
    configs.  Otherwise the factory falls back to the hardcoded Python
    definitions from earlier phases.
    """

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        registry: ConfigRegistry | None = None,
    ) -> None:
        self._model = model
        self._registry = registry

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _get_model(self) -> BaseChatModel:
        """Return the provided model or create one from settings."""
        if self._model is not None:
            return self._model
        settings = get_settings()
        return ChatOpenAI(
            model=settings.llm.model,
            api_key=settings.llm.openai_api_key,
        )

    # ------------------------------------------------------------------
    # Config-driven resolution
    # ------------------------------------------------------------------

    def _resolve_subagent(self, decl: DeclSubAgentConfig) -> SubAgentRuntime:
        """Convert a declarative subagent config into a runtime config."""
        tools: list = []
        if decl.tools and decl.tools.patterns:
            tools = _resolve_tool_patterns(decl.tools.patterns)
        # Fallback: if no patterns resolved, give full impl + internal.
        if not tools:
            tools = [*_ALL_IMPL_TOOLS, *MEMORY_TOOLS, *TODO_TOOLS]

        return SubAgentRuntime(
            name=decl.name,
            description=decl.description,
            tools=_prepare_tools(tools),
            prompt=decl.prompt if isinstance(decl.prompt, str) else "",
            recursion_limit=decl.recursion_limit,
        )

    def _resolve_coordinator_tools(self, agent_cfg: DeclAgentConfig) -> list:
        """Resolve the coordinator's tool surface from its config.

        The ``think`` tool is intentionally excluded — only subagents
        get it (as a return-direct exit ramp via ``create_task_tool``).
        """
        tools: list = []
        if agent_cfg.tools and agent_cfg.tools.patterns:
            tools = _resolve_tool_patterns(agent_cfg.tools.patterns)
        return _prepare_tools(tools)

    # ------------------------------------------------------------------
    # Hardcoded tool surfaces (fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _coordinator_tools() -> list[BaseTool]:
        """Assemble the coordinator's tool surface.

        The coordinator gets read-only filesystem tools for quick
        lookups, web, memory, and todo.  The task delegation tool is
        added automatically by ``create_deep_agent`` from the subagent
        configs.  The ``think`` tool is intentionally omitted — only
        subagents get it (as a return-direct exit ramp).
        """
        return _prepare_tools([
            *_READ_ONLY_FILE_TOOLS,
            *WEB_TOOLS,
            *MEMORY_TOOLS,
            *TODO_TOOLS,
        ])

    @staticmethod
    def _subagent_configs() -> list[SubAgentRuntime]:
        return [_explorer_config(), _worker_config()]

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
        agent_name: str | None = None,
    ) -> CompiledStateGraph:
        """Build the coordinator + subagent graph.

        Parameters
        ----------
        checkpointer:
            Optional checkpoint saver for conversation persistence.
            Pass ``None`` for LangGraph Studio or stateless use.
        agent_name:
            Name of the agent config to use (config-driven mode only).
            ``None`` uses the default agent.

        Returns
        -------
        CompiledStateGraph
        """
        return create_deep_agent(
            model=self._get_model(),
            tools=self._coordinator_tools(),
            prompt=COORDINATOR_PROMPT,
            subagent_configs=self._subagent_configs(),
            state_schema=AgentState,
            context_schema=AgentContext,
            checkpointer=checkpointer,
            name="coordinator",
        )

    async def create_from_config(
        self,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
        agent_name: str | None = None,
    ) -> CompiledStateGraph:
        """Build a graph from declarative YAML configs.

        Requires that the factory was initialised with a
        ``ConfigRegistry``.  Falls back to ``create()`` if no
        registry is available.

        This is ``async`` because config loading reads files.
        """
        if self._registry is None:
            return self.create(checkpointer=checkpointer)

        agent_cfg = await self._registry.get_agent(agent_name)

        # Resolve prompt.
        prompt = agent_cfg.prompt if isinstance(agent_cfg.prompt, str) else ""

        # Resolve coordinator tools.
        coordinator_tools = self._resolve_coordinator_tools(agent_cfg)

        # Resolve subagents.
        subagent_runtimes: list[SubAgentRuntime] = []
        if agent_cfg.subagents:
            subagents_batch = await self._registry.load_subagents()
            for sa_name in agent_cfg.subagents:
                sa_decl = subagents_batch.get_subagent(sa_name)
                assert sa_decl is not None, (
                    f"subagent '{sa_name}' must exist in registry "
                    f"(validated during load_agents)"
                )
                subagent_runtimes.append(self._resolve_subagent(sa_decl))

        return create_deep_agent(
            model=self._get_model(),
            tools=coordinator_tools,
            prompt=prompt,
            subagent_configs=subagent_runtimes or None,
            state_schema=AgentState,
            context_schema=AgentContext,
            checkpointer=checkpointer,
            name=agent_cfg.name,
        )
