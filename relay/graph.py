from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.middlewares import (
    PendingToolResultMiddleware,
    ReturnDirectMiddleware,
    TokenCostMiddleware,
)
from relay.settings import get_settings
from relay.tools.filesystem import FILE_SYSTEM_TOOLS
from relay.tools.memory import MEMORY_TOOLS
from relay.tools.terminal import TERMINAL_TOOLS
from relay.tools.todo import TODO_TOOLS
from relay.tools.web import WEB_TOOLS


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Construct the ReAct agent graph with middleware."""
    settings = get_settings()
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=settings.llm.openai_api_key,
    )

    # -----------------------------------------------------------------------
    # Middleware ordering:
    #
    #   aafter_model  hooks fire first-to-last after each LLM call.
    #   abefore_agent hooks fire first-to-last before the agent loop.
    #   abefore_model hooks fire first-to-last before each LLM call.
    #   awrap_tool_call hooks nest (first wraps all others).
    # -----------------------------------------------------------------------

    # Group 1: aafter_model — after each model response
    after_model: list[AgentMiddleware] = [
        TokenCostMiddleware(),  # Extract token usage and calculate costs
    ]

    # Group 2: abefore_agent — before each agent invocation
    before_agent: list[AgentMiddleware] = [
        PendingToolResultMiddleware(),  # Repair missing tool results after interrupts
    ]

    # Group 3: abefore_model — before each model call
    before_model: list[AgentMiddleware] = [
        ReturnDirectMiddleware(),  # Short-circuit when return_direct=True
    ]

    middlewares: list[AgentMiddleware] = (
        prompt_mw + after_model + before_agent + before_model
    )

    # -----------------------------------------------------------------------
    # Enable error-as-message for all tools that raise ToolException.
    # Without this, ToolException propagates up and crashes the graph;
    # with it, LangGraph converts the exception into an error ToolMessage
    # so the agent can self-correct.
    # -----------------------------------------------------------------------

    tools = [
        *FILE_SYSTEM_TOOLS,
        *TERMINAL_TOOLS,
        *WEB_TOOLS,
        *MEMORY_TOOLS,
        *TODO_TOOLS,
    ]
    for t in tools:
        t.handle_tool_error = True

    return create_agent(
        llm,
        tools=tools,
        state_schema=AgentState,
        context_schema=AgentContext,
        checkpointer=checkpointer,
        middleware=middlewares,
    )
