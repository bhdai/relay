"""Integration tests for terminal tools — exercises real subprocesses."""

import pytest

from relay.tools.terminal import get_directory_structure, run_command
from tests.fixtures.tool_helpers import make_tool_call, run_tool


@pytest.mark.asyncio
async def test_run_command_echo(create_test_graph):
    app = create_test_graph([run_command])

    state = make_tool_call("run_command", command="echo hello")
    result = await run_tool(app, state)
    assert "hello" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_run_command_failure(create_test_graph):
    """A failing command should surface an error message, not crash."""
    app = create_test_graph([run_command])

    state = make_tool_call("run_command", command="false")
    result = await run_tool(app, state)
    last_msg = result["messages"][-1]
    # ToolNode wraps ToolException content into the message.
    assert "exit code" in last_msg.content or "Error" in last_msg.content


@pytest.mark.asyncio
async def test_get_directory_structure(create_test_graph, tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y")

    app = create_test_graph([get_directory_structure])

    state = make_tool_call("get_directory_structure", dir_path=str(tmp_path))
    result = await run_tool(app, state)
    content = result["messages"][-1].content
    assert "a.txt" in content
    assert "b.txt" in content
