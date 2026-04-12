"""Todo-list tools for structured task tracking.

The agent uses these to plan multi-step work and show progress.
Todos live in ``state["todos"]`` — a ``list[Todo]`` accumulated by
the lambda reducer in :mod:`relay.state`.

``write_todos`` replaces the full list each time (the reducer appends,
but we always send the complete snapshot so upstream state is
effectively replaced).
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from relay.agents.state import Todo


@tool
def write_todos(
    todos: list[Todo],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Replace the full todo list.

    Args:
        todos: Complete list of todo items (content + status).
    """
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(
                    content=f"Updated todos ({len(todos)} items)",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def read_todos(state: Annotated[dict, InjectedState]) -> str:
    """Read the current todo list."""
    todos = state.get("todos") or []
    if not todos:
        return "(no todos)"
    return "\n".join(f"[{t['status']}] {t['content']}" for t in todos)


TODO_TOOLS = [write_todos, read_todos]
