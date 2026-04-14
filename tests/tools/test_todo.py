"""Unit tests for todo tools — no graph or I/O required."""

from langgraph.types import Command

from relay.tools.internal.todo import read_todos, write_todos


def _state(**overrides) -> dict:
    """Build a minimal AgentState dict with sensible defaults."""
    base = {
        "messages": [],
        "files": None,
        "todos": None,
        "current_input_tokens": None,
        "current_output_tokens": None,
        "total_cost": None,
    }
    base.update(overrides)
    return base


class TestWriteTodos:
    def test_returns_command(self):
        todos = [{"content": "first task", "status": "todo"}]
        result = write_todos.func(todos=todos, tool_call_id="call_1")
        assert isinstance(result, Command)
        assert result.update["todos"] == todos


class TestReadTodos:
    def test_empty_state(self):
        result = read_todos.func(state=_state())
        assert result == "(no todos)"

    def test_formats_todos(self):
        result = read_todos.func(
            state=_state(
                todos=[
                    {"content": "first", "status": "done"},
                    {"content": "second", "status": "in_progress"},
                ]
            )
        )
        assert "[done] first" in result
        assert "[in_progress] second" in result
