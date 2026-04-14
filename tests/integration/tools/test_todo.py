"""Integration tests for todo tools — exercises real graph invocations."""

import pytest

from relay.agents.state import AgentState
from relay.tools.internal.todo import read_todos, write_todos
from tests.fixtures.tool_helpers import make_tool_call, run_tool

TODO_TOOLS = [write_todos, read_todos]


@pytest.mark.asyncio
async def test_write_then_read_todos(create_test_graph):
    """Write todos, then read them back through the graph."""
    app = create_test_graph(TODO_TOOLS, state_schema=AgentState)

    todos = [
        {"content": "first task", "status": "todo"},
        {"content": "second task", "status": "in_progress"},
    ]
    state = make_tool_call("write_todos", todos=todos)
    await run_tool(app, state)

    state = make_tool_call("read_todos", call_id="call_2")
    result = await run_tool(app, state)
    last_msg = result["messages"][-1]
    assert "[todo] first task" in last_msg.content
    assert "[in_progress] second task" in last_msg.content
