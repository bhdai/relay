"""Agent factory — centralized runtime assembly.

``AgentFactory`` resolves prompts, tools, subagents, and LLMs into a
compiled LangGraph graph.  This decouples agent construction from the
CLI session and from the compatibility shim in ``relay.graph``.

The factory is intentionally simple for now: no MCP, skills, catalogs,
or sandboxes.  Those will arrive in later phases once the config
registry and tool factory exist (Phases 4–7).
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
from relay.tools.planning import PLANNING_TOOLS
from relay.tools.subagents import SubAgentConfig
from relay.tools.impl.terminal import TERMINAL_TOOLS
from relay.tools.internal.todo import TODO_TOOLS
from relay.tools.impl.web import WEB_TOOLS

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph


# ==============================================================================
# Tool Surface Definitions
# ==============================================================================
#
# Read-only filesystem tools shared by the coordinator and explorer.
# Mutating tools (write, edit, delete, terminal) are reserved for the
# general-purpose subagent.

_READ_ONLY_FILE_TOOLS: list = [read_file, glob_files, grep_files, ls]


def _prepare_tools(tools: list) -> list:
    """Enable ToolException-to-ToolMessage behaviour on each tool."""
    for tool_obj in tools:
        tool_obj.handle_tool_error = True
    return tools


# ==============================================================================
# Subagent Definitions
# ==============================================================================


def _explorer_config() -> SubAgentConfig:
    return SubAgentConfig(
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


def _worker_config() -> SubAgentConfig:
    return SubAgentConfig(
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
    """

    def __init__(self, *, model: BaseChatModel | None = None) -> None:
        self._model = model

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
    # Tool surfaces
    # ------------------------------------------------------------------

    @staticmethod
    def _coordinator_tools() -> list[BaseTool]:
        """Assemble the coordinator's tool surface.

        The coordinator gets read-only filesystem tools for quick
        lookups, web, memory, todo, and planning (think).  The task
        delegation tool is added automatically by ``create_deep_agent``
        from the subagent configs.
        """
        return _prepare_tools([
            *_READ_ONLY_FILE_TOOLS,
            *WEB_TOOLS,
            *MEMORY_TOOLS,
            *TODO_TOOLS,
            *PLANNING_TOOLS,
        ])

    @staticmethod
    def _subagent_configs() -> list[SubAgentConfig]:
        return [_explorer_config(), _worker_config()]

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> CompiledStateGraph:
        """Build the coordinator + subagent graph.

        Parameters
        ----------
        checkpointer:
            Optional checkpoint saver for conversation persistence.
            Pass ``None`` for LangGraph Studio or stateless use.

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
