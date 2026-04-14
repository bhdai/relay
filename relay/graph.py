"""Build the relay agent graph.

**Compatibility wrapper.**  The real construction logic lives in
``relay.agents.react_agent`` and ``relay.agents.deep_agent``.  This
module remains as the public entry point so that ``relay.cli.session``
and other callers keep working without changes.

Relay uses a **coordinator + worker** architecture:

- A single **coordinator agent** handles user turns.  It has a narrow
  tool surface (read-only filesystem, web, memory, todo, planning)
  plus the ``task`` delegation tool.
- Two **subagents** are available via ``task``:
  - **explorer** — read-only investigation with the ``think`` tool for
    reflection between searches.
  - **general-purpose** — full tool surface (filesystem mutating ops,
    terminal, web, memory, todo) for complex multi-step tasks.
"""

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from relay.agents.context import AgentContext
from relay.agents.deep_agent import create_deep_agent
from relay.agents.state import AgentState
from relay.prompt import COORDINATOR_PROMPT, EXPLORER_PROMPT, WORKER_PROMPT
from relay.settings import get_settings
from relay.tools.filesystem import FILE_SYSTEM_TOOLS, glob_files, grep_files, ls, read_file
from relay.tools.memory import MEMORY_TOOLS
from relay.tools.planning import PLANNING_TOOLS
from relay.tools.subagents import SubAgentConfig
from relay.tools.terminal import TERMINAL_TOOLS
from relay.tools.todo import TODO_TOOLS
from relay.tools.web import WEB_TOOLS

# ==============================================================================
# Read-only filesystem tools available to the coordinator and explorer.
# ==============================================================================

_READ_ONLY_FILE_TOOLS = [read_file, glob_files, grep_files, ls]


def _prepare_tools(tools: list):
    """Enable ToolException-to-ToolMessage behaviour on each tool."""
    for tool_obj in tools:
        tool_obj.handle_tool_error = True
    return tools


# ==============================================================================
# Subagent definitions
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
# Graph construction
# ==============================================================================


def _coordinator_tools() -> list:
    """Assemble the coordinator's tool surface.

    The coordinator gets read-only filesystem tools for quick lookups,
    web, memory, todo, planning (think).  The task delegation tool is
    added automatically by create_deep_agent from the subagent configs.
    Mutating filesystem and terminal tools are reserved for the
    general-purpose subagent.
    """
    return _prepare_tools([
        *_READ_ONLY_FILE_TOOLS,
        *WEB_TOOLS,
        *MEMORY_TOOLS,
        *TODO_TOOLS,
        *PLANNING_TOOLS,
    ])


def _create_llm():
    """Create the LLM from settings."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm.model,
        api_key=settings.llm.openai_api_key,
    )


def build_graph():
    """Construct Relay's coordinator + subagent graph.

    This zero-argument factory is the entry point for LangGraph Studio
    (``langgraph dev``) and the ``langgraph.json`` configuration.

    For callers that need a checkpointer (e.g. the CLI), use
    ``build_graph_with_checkpointer`` instead.
    """
    return create_deep_agent(
        model=_create_llm(),
        tools=_coordinator_tools(),
        prompt=COORDINATOR_PROMPT,
        subagent_configs=[_explorer_config(), _worker_config()],
        state_schema=AgentState,
        context_schema=AgentContext,
        name="coordinator",
    )


def build_graph_with_checkpointer(
    checkpointer: BaseCheckpointSaver,
):
    """Construct Relay's graph with an explicit checkpointer.

    Used by the CLI session where persistence is required.
    """
    return create_deep_agent(
        model=_create_llm(),
        tools=_coordinator_tools(),
        prompt=COORDINATOR_PROMPT,
        subagent_configs=[_explorer_config(), _worker_config()],
        state_schema=AgentState,
        context_schema=AgentContext,
        checkpointer=checkpointer,
        name="coordinator",
    )

