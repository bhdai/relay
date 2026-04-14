"""Unit tests for memory tools — no graph or I/O required."""

import pytest
from langchain_core.tools import ToolException
from langgraph.types import Command

from relay.tools.internal.memory import (
    edit_memory_file,
    list_memory_files,
    read_memory_file,
    write_memory_file,
)


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


# Tools with InjectedState / InjectedToolCallId cannot be invoked
# through .ainvoke() without a graph.  We call the underlying function
# directly to test the business logic.


class TestListMemoryFiles:
    def test_empty_state(self):
        result = list_memory_files.func(state=_state())
        assert result == "(empty)"

    def test_lists_keys(self):
        result = list_memory_files.func(
            state=_state(files={"b.md": "bbb", "a.md": "aaa"})
        )
        assert "a.md" in result
        assert "b.md" in result


class TestReadMemoryFile:
    def test_reads_content(self):
        result = read_memory_file.func(
            filename="notes.md",
            state=_state(files={"notes.md": "hello world"}),
        )
        assert result == "hello world"

    def test_missing_file_raises(self):
        with pytest.raises(ToolException, match="File not found"):
            read_memory_file.func(
                filename="gone.md",
                state=_state(files={}),
            )


class TestWriteMemoryFile:
    def test_returns_command(self):
        result = write_memory_file.func(
            filename="notes.md", content="hello", tool_call_id="call_1"
        )
        assert isinstance(result, Command)
        assert result.update["files"]["notes.md"] == "hello"


class TestEditMemoryFile:
    def test_patches_content(self):
        result = edit_memory_file.func(
            filename="notes.md",
            old_content="hello",
            new_content="goodbye",
            state=_state(files={"notes.md": "hello world"}),
            tool_call_id="call_1",
        )
        assert isinstance(result, Command)
        assert result.update["files"]["notes.md"] == "goodbye world"

    def test_missing_file_raises(self):
        with pytest.raises(ToolException, match="File not found"):
            edit_memory_file.func(
                filename="gone.md",
                old_content="x",
                new_content="y",
                state=_state(files={}),
                tool_call_id="call_1",
            )

    def test_missing_content_raises(self):
        with pytest.raises(ToolException, match="old_content not found"):
            edit_memory_file.func(
                filename="notes.md",
                old_content="nonexistent",
                new_content="y",
                state=_state(files={"notes.md": "hello world"}),
                tool_call_id="call_1",
            )
