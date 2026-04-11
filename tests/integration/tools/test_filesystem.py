"""Integration tests for filesystem tools — exercises real I/O via ``tmp_path``."""

from pathlib import Path

import pytest

from relay.tools.filesystem import (
    EditOperation,
    create_dir,
    delete_file,
    edit_file,
    glob_files,
    grep_files,
    move_file,
    read_file,
    write_file,
)
from tests.fixtures.tool_helpers import make_tool_call, run_tool


@pytest.mark.asyncio
async def test_write_and_read_file(create_test_graph, tmp_path: Path):
    """Round-trip: write a file, then read it back through the graph."""
    app = create_test_graph([write_file, read_file])
    target = str(tmp_path / "test.txt")

    state = make_tool_call("write_file", file_path=target, content="Hello World")
    await run_tool(app, state)

    assert (tmp_path / "test.txt").exists()
    assert (tmp_path / "test.txt").read_text() == "Hello World"

    state = make_tool_call("read_file", call_id="call_2", file_path=target)
    result = await run_tool(app, state)
    last_msg = result["messages"][-1]
    assert "Hello World" in last_msg.content


@pytest.mark.asyncio
async def test_write_file_rejects_existing(create_test_graph, tmp_path: Path):
    """write_file must refuse to overwrite an existing file."""
    (tmp_path / "exists.txt").write_text("original")
    app = create_test_graph([write_file])

    state = make_tool_call("write_file", file_path=str(tmp_path / "exists.txt"), content="new")
    result = await run_tool(app, state)
    last_msg = result["messages"][-1]
    assert "already exists" in last_msg.content


@pytest.mark.asyncio
async def test_edit_file_single(create_test_graph, tmp_path: Path):
    """A single edit replaces the matched text."""
    (tmp_path / "edit.txt").write_text("line 1\nline 2\nline 3")
    app = create_test_graph([edit_file])

    state = make_tool_call(
        "edit_file",
        file_path=str(tmp_path / "edit.txt"),
        edits=[EditOperation(old_content="line 2", new_content="modified line 2")],
    )
    await run_tool(app, state)
    assert "modified line 2" in (tmp_path / "edit.txt").read_text()


@pytest.mark.asyncio
async def test_edit_file_multiple_sequential(create_test_graph, tmp_path: Path):
    """Multiple non-overlapping edits all apply correctly."""
    (tmp_path / "multi.txt").write_text("line 1\nline 2\nline 3\nline 4\n")
    app = create_test_graph([edit_file])

    state = make_tool_call(
        "edit_file",
        file_path=str(tmp_path / "multi.txt"),
        edits=[
            EditOperation(old_content="line 1", new_content="FIRST"),
            EditOperation(old_content="line 2", new_content="SECOND"),
            EditOperation(old_content="line 3", new_content="THIRD"),
        ],
    )
    await run_tool(app, state)
    content = (tmp_path / "multi.txt").read_text()
    assert "FIRST" in content
    assert "SECOND" in content
    assert "THIRD" in content
    assert "line 4" in content


@pytest.mark.asyncio
async def test_edit_file_overlapping_rejected(create_test_graph, tmp_path: Path):
    """Overlapping edits are detected and rejected with an error."""
    (tmp_path / "overlap.txt").write_text("0123456789")
    app = create_test_graph([edit_file])

    state = make_tool_call(
        "edit_file",
        file_path=str(tmp_path / "overlap.txt"),
        edits=[
            EditOperation(old_content="234", new_content="ABC"),
            EditOperation(old_content="456", new_content="XYZ"),
        ],
    )
    result = await run_tool(app, state)
    assert "Overlapping" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_create_dir_and_nested(create_test_graph, tmp_path: Path):
    app = create_test_graph([create_dir])
    target = str(tmp_path / "a" / "b" / "c")

    state = make_tool_call("create_dir", dir_path=target)
    await run_tool(app, state)
    assert Path(target).is_dir()


@pytest.mark.asyncio
async def test_move_file(create_test_graph, tmp_path: Path):
    (tmp_path / "src.txt").write_text("data")
    app = create_test_graph([move_file])

    state = make_tool_call(
        "move_file",
        source_path=str(tmp_path / "src.txt"),
        destination_path=str(tmp_path / "dst.txt"),
    )
    await run_tool(app, state)
    assert not (tmp_path / "src.txt").exists()
    assert (tmp_path / "dst.txt").read_text() == "data"


@pytest.mark.asyncio
async def test_delete_file(create_test_graph, tmp_path: Path):
    (tmp_path / "doomed.txt").write_text("bye")
    app = create_test_graph([delete_file])

    state = make_tool_call("delete_file", file_path=str(tmp_path / "doomed.txt"))
    await run_tool(app, state)
    assert not (tmp_path / "doomed.txt").exists()


# ==============================================================================
# glob_files
# ==============================================================================


@pytest.mark.asyncio
async def test_glob_files_basic(create_test_graph, tmp_path: Path):
    (tmp_path / "foo.py").write_text("")
    (tmp_path / "bar.ts").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "baz.py").write_text("")

    app = create_test_graph([glob_files])
    state = make_tool_call("glob_files", pattern="*.py", dir_path=str(tmp_path))
    result = await run_tool(app, state)
    content = result["messages"][-1].content
    assert "foo.py" in content
    # fnmatch with *.py matches filename component only, so sub/baz.py may not match.
    assert "bar.ts" not in content


@pytest.mark.asyncio
async def test_glob_files_ignores_venv(create_test_graph, tmp_path: Path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pkg.py").write_text("")
    (tmp_path / "real.py").write_text("")

    app = create_test_graph([glob_files])
    state = make_tool_call("glob_files", pattern="*.py", dir_path=str(tmp_path))
    result = await run_tool(app, state)
    content = result["messages"][-1].content
    assert "real.py" in content
    assert ".venv" not in content


@pytest.mark.asyncio
async def test_glob_files_no_matches(create_test_graph, tmp_path: Path):
    (tmp_path / "a.txt").write_text("")

    app = create_test_graph([glob_files])
    state = make_tool_call("glob_files", pattern="*.rs", dir_path=str(tmp_path))
    result = await run_tool(app, state)
    assert "No files matching" in result["messages"][-1].content


# ==============================================================================
# grep_files
# ==============================================================================


@pytest.mark.asyncio
async def test_grep_files_literal(create_test_graph, tmp_path: Path):
    (tmp_path / "a.py").write_text("# TODO: fix this\nprint('hello')\n")
    (tmp_path / "b.py").write_text("pass\n")

    app = create_test_graph([grep_files])
    state = make_tool_call("grep_files", pattern="TODO", dir_path=str(tmp_path))
    result = await run_tool(app, state)
    content = result["messages"][-1].content
    assert "a.py:1:" in content
    assert "TODO" in content


@pytest.mark.asyncio
async def test_grep_files_regex(create_test_graph, tmp_path: Path):
    (tmp_path / "data.txt").write_text("abc123\ndef456\nxyz\n")

    app = create_test_graph([grep_files])
    state = make_tool_call(
        "grep_files", pattern=r"\d+", dir_path=str(tmp_path), is_regex=True,
    )
    result = await run_tool(app, state)
    content = result["messages"][-1].content
    assert "data.txt:1:" in content
    assert "data.txt:2:" in content


@pytest.mark.asyncio
async def test_grep_files_no_matches(create_test_graph, tmp_path: Path):
    (tmp_path / "a.txt").write_text("nothing\n")

    app = create_test_graph([grep_files])
    state = make_tool_call("grep_files", pattern="missing", dir_path=str(tmp_path))
    result = await run_tool(app, state)
    assert "No matches" in result["messages"][-1].content
