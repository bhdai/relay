"""Agent state definition with custom LangGraph reducers.

Each field uses an ``Annotated[T, reducer]`` pattern so that LangGraph
knows how to merge partial updates from parallel nodes or tool
``Command`` returns.
"""

from typing import Annotated, Literal, TypedDict

from langchain.agents import AgentState as BaseAgentState


# ==============================================================================
# Reducer Functions
# ==============================================================================
#
# Reducers tell LangGraph how to combine the *existing* value in the
# state with a *new* value supplied by a node or ``Command(update=…)``.


def file_reducer(
    left: dict[str, str] | None, right: dict[str, str] | None
) -> dict[str, str]:
    """Deep-merge dicts; right wins on key conflict."""
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}


def sum_reducer(left: float | None, right: float | None) -> float:
    """Accumulate costs across turns."""
    return (left or 0.0) + (right or 0.0)


def replace_reducer(left: int | None, right: int | None) -> int:
    """Use the latest value (e.g. cumulative token counts from the LLM)."""
    return right if right is not None else (left or 0)


# ==============================================================================
# Data Types
# ==============================================================================


class Todo(TypedDict):
    content: str
    status: Literal["todo", "in_progress", "done"]


# ==============================================================================
# Agent State
# ==============================================================================


class AgentState(BaseAgentState):
    """Extended agent state with memory files, todos, and cost tracking."""

    # Replace-on-write: the agent always sends the complete todo snapshot, so
    # the latest write wins.  An append reducer would accumulate duplicates on
    # every call to write_todos.
    todos: Annotated[list[Todo] | None, lambda l, r: r if r is not None else l]
    files: Annotated[dict[str, str] | None, file_reducer]
    current_input_tokens: Annotated[int | None, replace_reducer]
    current_output_tokens: Annotated[int | None, replace_reducer]
    total_cost: Annotated[float | None, sum_reducer]
