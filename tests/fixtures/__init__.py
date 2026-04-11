"""Reusable test helpers for tool integration tests."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def make_tool_call(tool_name: str, call_id: str = "call_1", **kwargs: Any) -> dict:
    """Create a tool-call message state suitable for ``app.ainvoke()``.

    Args:
        tool_name: Name of the tool to invoke.
        call_id: Unique call identifier (default ``"call_1"``).
        **kwargs: Keyword arguments forwarded as tool arguments.

    Returns:
        A dict with a ``messages`` key ready for graph invocation.
    """
    return {
        "messages": [
            HumanMessage(content="Execute tool"),
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": tool_name, "args": kwargs}],
            ),
        ]
    }


async def run_tool(
    app,
    tool_call_state: dict,
    thread_id: str = "test",
) -> Any:
    """Execute a tool call through a compiled graph.

    Args:
        app: A compiled LangGraph application.
        tool_call_state: State produced by :func:`make_tool_call`.
        thread_id: Conversation thread identifier.

    Returns:
        The graph invocation result (typically contains ``messages``).
    """
    config = {"configurable": {"thread_id": thread_id}}
    return await app.ainvoke(tool_call_state, config=config)
