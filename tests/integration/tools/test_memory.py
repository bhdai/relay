"""Integration tests for memory tools — exercises real graph invocations."""

import pytest

from relay.agents.state import AgentState
from relay.tools.memory import (
    edit_memory_file,
    list_memory_files,
    read_memory_file,
    write_memory_file,
)
from tests.fixtures.tool_helpers import make_tool_call, run_tool

MEMORY_TOOLS = [list_memory_files, read_memory_file, write_memory_file, edit_memory_file]


@pytest.mark.asyncio
async def test_write_then_read(create_test_graph):
    """Write a memory file, then read it back through the graph."""
    app = create_test_graph(MEMORY_TOOLS, state_schema=AgentState)

    state = make_tool_call("write_memory_file", filename="notes.md", content="hello")
    result = await run_tool(app, state)

    state = make_tool_call("read_memory_file", call_id="call_2", filename="notes.md")
    result = await run_tool(app, state)
    last_msg = result["messages"][-1]
    assert "hello" in last_msg.content


@pytest.mark.asyncio
async def test_list_after_write(create_test_graph):
    """list_memory_files should show written file keys."""
    app = create_test_graph(MEMORY_TOOLS, state_schema=AgentState)

    state = make_tool_call("write_memory_file", filename="plan.md", content="step 1")
    await run_tool(app, state)

    state = make_tool_call("list_memory_files", call_id="call_2")
    result = await run_tool(app, state)
    last_msg = result["messages"][-1]
    assert "plan.md" in last_msg.content


@pytest.mark.asyncio
async def test_edit_patches_content(create_test_graph):
    """edit_memory_file should replace a substring in an existing file."""
    app = create_test_graph(MEMORY_TOOLS, state_schema=AgentState)

    state = make_tool_call("write_memory_file", filename="notes.md", content="hello world")
    await run_tool(app, state)

    state = make_tool_call(
        "edit_memory_file",
        call_id="call_2",
        filename="notes.md",
        old_content="hello",
        new_content="goodbye",
    )
    await run_tool(app, state)

    state = make_tool_call("read_memory_file", call_id="call_3", filename="notes.md")
    result = await run_tool(app, state)
    last_msg = result["messages"][-1]
    assert "goodbye world" in last_msg.content
