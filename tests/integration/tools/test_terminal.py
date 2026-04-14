"""Integration tests for terminal tools — exercises real subprocesses."""

import pytest

from relay.tools.impl.filesystem import ls
from relay.tools.impl.terminal import run_command
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
async def test_ls(create_test_graph, tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y")

    app = create_test_graph([ls])

    state = make_tool_call("ls", dir_path=str(tmp_path))
    result = await run_tool(app, state)
    content = result["messages"][-1].content
    assert "a.txt" in content
    assert "b.txt" in content


@pytest.mark.asyncio
async def test_ls_with_ignore(create_test_graph, tmp_path):
    (tmp_path / "keep.py").write_text("x")
    (tmp_path / "skip.bak").write_text("y")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("z")

    app = create_test_graph([ls])

    state = make_tool_call("ls", dir_path=str(tmp_path), ignore=["*.bak", "tests/"])
    result = await run_tool(app, state)
    content = result["messages"][-1].content
    assert "keep.py" in content
    assert "skip.bak" not in content
    assert "tests" not in content
