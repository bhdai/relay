"""Todo-list tools for structured task tracking.

The agent uses these to plan multi-step work and show progress.
Todos live in ``state["todos"]`` — a ``list[Todo]`` that is replaced
entirely on each ``write_todos`` call (the reducer keeps the latest
snapshot, not an accumulation).
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

    Use this tool VERY frequently to track multi-step work.  Create a
    todo list at the start of any task that requires more than 2 steps,
    and update it after every completed action.

    Rules:
    - Always send the *complete* list — this call replaces the previous
      snapshot entirely.  Sending only changed items will lose the rest.
    - Keep exactly ONE item ``in_progress`` at a time.
    - Mark tasks ``done`` immediately after finishing them — do not
      batch completions.
    - Cancel items that become irrelevant rather than leaving them stale.

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
    """Read the current todo list.

    Call this at the start of a turn to stay oriented on remaining tasks
    and track progress through a multi-step plan.
    """
    todos = state.get("todos") or []
    if not todos:
        return "(no todos)"
    return "\n".join(f"[{t['status']}] {t['content']}" for t in todos)


TODO_TOOLS = [write_todos, read_todos]
