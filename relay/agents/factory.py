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

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from relay.agents.context import AgentContext
from relay.agents.deep_agent import create_deep_agent
from relay.agents.state import AgentState
from relay.configs.llm import LLMConfig, LLMProvider
from relay.llms.factory import LLMFactory
from relay.prompt import COORDINATOR_PROMPT, EXPLORER_PROMPT, WORKER_PROMPT
from relay.settings import get_settings
from relay.tools.factory import ToolFactory
from relay.tools.impl.filesystem import FILE_SYSTEM_TOOLS, glob_files, grep_files, ls, read_file
from relay.tools.internal.memory import MEMORY_TOOLS
from relay.tools.subagents import SubAgentRuntime
from relay.tools.impl.terminal import TERMINAL_TOOLS
from relay.tools.internal.todo import TODO_TOOLS
from relay.tools.impl.web import WEB_TOOLS
from relay.utils.patterns import matches_patterns, two_part_matcher

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from relay.configs.agent import AgentConfig as DeclAgentConfig
    from relay.configs.agent import SubAgentConfig as DeclSubAgentConfig
    from relay.configs.registry import ConfigRegistry
    from relay.mcp.client import MCPClient
    from relay.skills.factory import Skill, SkillFactory

logger = logging.getLogger(__name__)


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
# Tool reference categories
# ==============================================================================

TOOL_CATEGORY_IMPL = "impl"
TOOL_CATEGORY_MCP = "mcp"
TOOL_CATEGORY_INTERNAL = "internal"


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
        model_name: str | None = None,
        registry: ConfigRegistry | None = None,
        tool_factory: ToolFactory | None = None,
        skill_factory: SkillFactory | None = None,
        llm_factory: LLMFactory | None = None,
    ) -> None:
        self._model = model
        self._model_name = model_name
        self._registry = registry
        self._tool_factory = tool_factory or ToolFactory()
        self._skill_factory = skill_factory
        self._llm_factory = llm_factory or LLMFactory(get_settings())

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _build_default_llm_config(self, *, model_name: str | None = None) -> LLMConfig:
        """Construct the env-backed default LLM config."""
        settings = get_settings()
        llm_settings = settings.llm
        rate_settings = settings.rate_limit
        provider = LLMProvider(llm_settings.provider)
        resolved_model_name = model_name or llm_settings.model

        return LLMConfig(
            provider=provider,
            model=resolved_model_name,
            alias=resolved_model_name,
            max_tokens=llm_settings.max_tokens,
            temperature=llm_settings.temperature,
            streaming=llm_settings.streaming,
            rate_config={
                "requests_per_second": rate_settings.requests_per_second,
                "check_every_n_seconds": rate_settings.check_every_n_seconds,
                "max_bucket_size": rate_settings.max_bucket_size,
            },
            input_cost_per_mtok=llm_settings.input_cost_per_mtok,
            output_cost_per_mtok=llm_settings.output_cost_per_mtok,
        )

    def _model_from_config(self, llm_config: LLMConfig | None) -> BaseChatModel:
        """Instantiate a model from a resolved LLM config."""
        if self._model is not None:
            return self._model

        return self._llm_factory.create(llm_config or self._build_default_llm_config())

    async def _resolve_llm_config(
        self,
        *,
        configured_llm_name: str | None = None,
    ) -> LLMConfig:
        """Resolve CLI override, config alias, and env default into one config."""
        selected_name = self._model_name or configured_llm_name or "default"

        if self._registry is not None and selected_name != "default":
            llm_config = await self._registry.get_llm(selected_name)
            if llm_config is not None:
                return llm_config

            return self._build_default_llm_config(model_name=selected_name)

        if selected_name != "default":
            return self._build_default_llm_config(model_name=selected_name)

        return self._build_default_llm_config()

    async def resolve_llm_metadata(
        self,
        *,
        configured_llm_name: str | None = None,
    ) -> tuple[str, float, float]:
        """Resolve the active LLM label and pricing for CLI session state."""
        llm_config = await self._resolve_llm_config(
            configured_llm_name=configured_llm_name,
        )
        return (
            llm_config.alias,
            llm_config.input_cost_per_mtok or 0.0,
            llm_config.output_cost_per_mtok or 0.0,
        )

    # ------------------------------------------------------------------
    # Tool reference parsing (three-part → two-part)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_references(
        tool_refs: list[str] | None,
    ) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
        """Split three-part tool references into per-category two-part patterns.

        Each reference has the form ``category:module:name`` (e.g.
        ``impl:file_system:read_file``, ``mcp:server:tool``).
        Negative patterns start with ``!``.

        Returns ``(impl_patterns, mcp_patterns, internal_patterns)``
        where each list contains two-part ``module:name`` patterns (or
        ``None``).
        """
        if not tool_refs:
            return None, None, None

        impl_patterns: list[str] = []
        mcp_patterns: list[str] = []
        internal_patterns: list[str] = []

        for ref in tool_refs:
            is_negative = ref.startswith("!")
            clean_ref = ref[1:] if is_negative else ref

            parts = clean_ref.split(":")
            if len(parts) != 3:
                logger.warning("Invalid tool reference format: %s", ref)
                continue

            tool_type, module_pattern, tool_pattern = parts
            pattern = f"{module_pattern}:{tool_pattern}"
            if is_negative:
                pattern = f"!{pattern}"

            if tool_type == TOOL_CATEGORY_IMPL:
                impl_patterns.append(pattern)
            elif tool_type == TOOL_CATEGORY_MCP:
                mcp_patterns.append(pattern)
            elif tool_type == TOOL_CATEGORY_INTERNAL:
                internal_patterns.append(pattern)
            else:
                logger.warning("Unknown tool type: %s", tool_type)

        return (
            impl_patterns or None,
            mcp_patterns or None,
            internal_patterns or None,
        )

    # ------------------------------------------------------------------
    # Tool dict / filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tool_dict(tools: list[BaseTool]) -> dict[str, BaseTool]:
        """Build a name → tool mapping for fast lookup."""
        return {tool.name: tool for tool in tools}

    @staticmethod
    def _filter_tools(
        tool_dict: dict[str, BaseTool],
        patterns: list[str] | None,
        module_map: dict[str, str],
    ) -> list[BaseTool]:
        """Filter tools by pattern with wildcard and negative pattern support.

        Each pattern is a two-part ``module:name`` glob
        (e.g. ``file_system:*``, ``!terminal:run_command``).  Matching
        uses ``fnmatch`` under the hood via :func:`matches_patterns`.
        """
        if not patterns:
            return []

        return [
            tool
            for name, tool in tool_dict.items()
            if matches_patterns(
                patterns, two_part_matcher(name, module_map.get(name, ""))
            )
        ]

    @staticmethod
    def _filter_mcp_tools(
        tool_dict: dict[str, BaseTool] | None,
        patterns: list[str] | None,
        module_map: dict[str, str] | None,
    ) -> list[BaseTool]:
        """Filter MCP tools by pattern.  Handles ``server__name`` prefixed format."""
        if not patterns or not tool_dict:
            return []

        effective_map = module_map or {}

        def get_original_name(prefixed_name: str) -> str:
            parts = prefixed_name.split("__", 1)
            return parts[1] if len(parts) == 2 else prefixed_name

        return [
            tool
            for name, tool in tool_dict.items()
            if matches_patterns(
                patterns,
                two_part_matcher(get_original_name(name), effective_map.get(name, "")),
            )
        ]

    # ------------------------------------------------------------------
    # Config-driven resolution
    # ------------------------------------------------------------------

    def _resolve_subagent(
        self,
        decl: DeclSubAgentConfig,
        llm_config: LLMConfig | None = None,
        mcp_tools: dict[str, BaseTool] | None = None,
        mcp_module_map: dict[str, str] | None = None,
    ) -> SubAgentRuntime:
        """Convert a declarative subagent config into a runtime config."""
        tools = self._resolve_tools_from_patterns(
            decl.tools.patterns if decl.tools else [],
            mcp_tools=mcp_tools,
            mcp_module_map=mcp_module_map,
        )
        # Fallback: if no patterns resolved, give full impl + internal.
        if not tools:
            tools = [*_ALL_IMPL_TOOLS, *MEMORY_TOOLS, *TODO_TOOLS]

        return SubAgentRuntime(
            name=decl.name,
            description=decl.description,
            tools=_prepare_tools(tools),
            prompt=decl.prompt if isinstance(decl.prompt, str) else "",
            llm_config=llm_config,
            recursion_limit=decl.recursion_limit,
        )

    def _resolve_coordinator_tools(
        self,
        agent_cfg: DeclAgentConfig,
        mcp_tools: dict[str, BaseTool] | None = None,
        mcp_module_map: dict[str, str] | None = None,
    ) -> list:
        """Resolve the coordinator's tool surface from its config.

        The ``think`` tool is intentionally excluded — only subagents
        get it (as a return-direct exit ramp via ``create_task_tool``).
        """
        tools = self._resolve_tools_from_patterns(
            agent_cfg.tools.patterns if agent_cfg.tools else [],
            mcp_tools=mcp_tools,
            mcp_module_map=mcp_module_map,
        )
        return _prepare_tools(tools)

    def _resolve_tools_from_patterns(
        self,
        patterns: list[str],
        mcp_tools: dict[str, BaseTool] | None = None,
        mcp_module_map: dict[str, str] | None = None,
    ) -> list[BaseTool]:
        """Resolve a list of three-part patterns into tool objects.

        Uses the ``ToolFactory`` module maps for proper wildcard and
        negation matching.  When *mcp_tools* is provided, ``mcp:``
        category patterns are resolved against those tools.
        """
        if not patterns:
            return []

        impl_patterns, mcp_patterns, internal_patterns = self._parse_tool_references(
            patterns
        )

        impl_dict = self._build_tool_dict(self._tool_factory.get_impl_tools())
        internal_dict = self._build_tool_dict(self._tool_factory.get_internal_tools())

        impl_tools = self._filter_tools(
            impl_dict,
            impl_patterns,
            self._tool_factory.get_impl_module_map(),
        )

        resolved_mcp: list[BaseTool] = []
        if mcp_patterns and mcp_tools:
            resolved_mcp = self._filter_mcp_tools(
                mcp_tools,
                mcp_patterns,
                mcp_module_map or {},
            )

        internal_tools = self._filter_tools(
            internal_dict,
            internal_patterns,
            self._tool_factory.get_internal_module_map(),
        )

        return [*resolved_mcp, *impl_tools, *internal_tools]

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
        llm_config = self._build_default_llm_config(model_name=self._model_name)
        model = self._model_from_config(llm_config)
        return create_deep_agent(
            model=model,
            tools=self._coordinator_tools(),
            prompt=COORDINATOR_PROMPT,
            subagent_configs=self._subagent_configs(),
            subagent_model_provider=self._model_from_config,
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
        mcp_client: MCPClient | None = None,
        skills_dir: Path | None = None,
    ) -> CompiledStateGraph:
        """Build a graph from declarative YAML configs.

        Requires that the factory was initialised with a
        ``ConfigRegistry``.  Falls back to ``create()`` if no
        registry is available.

        Parameters
        ----------
        mcp_client:
            Optional MCP client whose tools are merged into the
            ``mcp:`` category for pattern matching.
        skills_dir:
            Optional directory containing skill packages.  When
            provided and a ``SkillFactory`` is available, skills are
            loaded and appended to the system prompt.

        This is ``async`` because config loading and MCP/skill loading
        read files.
        """
        if self._registry is None:
            return self.create(checkpointer=checkpointer)

        agent_cfg = await self._registry.get_agent(agent_name)

        # Resolve prompt.
        prompt = agent_cfg.prompt if isinstance(agent_cfg.prompt, str) else ""

        # ----------------------------------------------------------
        # MCP tools (optional)
        # ----------------------------------------------------------
        mcp_tools_dict: dict[str, BaseTool] | None = None
        mcp_module_map: dict[str, str] | None = None
        if mcp_client is not None:
            mcp_tool_list = await mcp_client.tools()
            mcp_tools_dict = self._build_tool_dict(mcp_tool_list)
            mcp_module_map = mcp_client.module_map

        # ----------------------------------------------------------
        # Skills (optional)
        # ----------------------------------------------------------
        skill_list: list[Skill] = []
        if self._skill_factory is not None and skills_dir is not None:
            skill_dict = await self._skill_factory.load_skills(skills_dir)
            for category_skills in skill_dict.values():
                skill_list.extend(category_skills.values())

        # Append skill summary to prompt if skills were loaded.
        if skill_list:
            prompt = f"{prompt}{self._build_skills_text(skill_list)}"

        # Resolve coordinator tools.
        coordinator_tools = self._resolve_coordinator_tools(
            agent_cfg,
            mcp_tools=mcp_tools_dict,
            mcp_module_map=mcp_module_map,
        )

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
                subagent_runtimes.append(
                    self._resolve_subagent(
                        sa_decl,
                        llm_config=await self._resolve_llm_config(
                            configured_llm_name=sa_decl.llm,
                        ),
                        mcp_tools=mcp_tools_dict,
                        mcp_module_map=mcp_module_map,
                    )
                )

        coordinator_llm_config = await self._resolve_llm_config(
            configured_llm_name=agent_cfg.llm,
        )
        model = self._model_from_config(coordinator_llm_config)

        return create_deep_agent(
            model=model,
            tools=coordinator_tools,
            prompt=prompt,
            subagent_configs=subagent_runtimes or None,
            subagent_model_provider=self._model_from_config,
            state_schema=AgentState,
            context_schema=AgentContext,
            checkpointer=checkpointer,
            name=agent_cfg.name,
        )

    # ------------------------------------------------------------------
    # Skill helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_skills_text(skills: list[Skill]) -> str:
        """Build skills documentation text for prompt injection."""
        text = "\n\n# Available Skills\n\n"
        text += (
            "When users ask you to perform tasks, check if any of the "
            "available skills below can help complete the task more "
            "effectively.\n\n"
        )
        for skill in skills:
            text += f"- **{skill.category}/{skill.name}**: {skill.description}\n"
        return text
