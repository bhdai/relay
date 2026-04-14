"""Unit tests for terminal tool helpers — no subprocess calls."""

from pathlib import Path

from relay.tools import IGNORE_DIRS
from relay.tools.impl.filesystem.ls import _collect_files, _render_tree
from relay.tools.impl.terminal import _extract_command_parts, _format_output


# ==============================================================================
# _extract_command_parts
# ==============================================================================


class TestExtractCommandParts:
    def test_simple_command(self):
        parts = _extract_command_parts("ls -la")
        assert parts == ["ls -la"]

    def test_chained_and(self):
        parts = _extract_command_parts("cd src && make build")
        assert "cd src" in parts
        assert "make build" in parts

    def test_pipe(self):
        parts = _extract_command_parts("cat file.txt | grep error")
        assert "cat file.txt" in parts
        assert "grep error" in parts

    def test_nested_substitution(self):
        parts = _extract_command_parts("echo $(whoami)")
        assert "echo $(whoami)" in parts
        assert "whoami" in parts

    def test_empty_command(self):
        assert _extract_command_parts("") == []

    def test_semicolon_chain(self):
        parts = _extract_command_parts("cmd1 ; cmd2 ; cmd3")
        assert len(parts) == 3


# ==============================================================================
# _format_output
# ==============================================================================


class TestFormatOutput:
    def test_stdout_only(self):
        assert _format_output("hello\n", "") == "hello"

    def test_stderr_only(self):
        assert _format_output("", "warning\n") == "warning"

    def test_both(self):
        result = _format_output("out\n", "err\n")
        assert "out" in result
        assert "err" in result

    def test_empty(self):
        assert _format_output("", "") == "Command completed successfully"


# ==============================================================================
# _collect_files
# ==============================================================================


class TestCollectFiles:
    def test_basic_collection(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("y")

        files, truncated = _collect_files(tmp_path, max_files=100)
        assert not truncated
        assert "a.txt" in files
        assert "sub/b.txt" in files

    def test_ignores_venv(self, tmp_path: Path):
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "big_pkg.py").write_text("x")
        (tmp_path / "real.py").write_text("y")

        files, _ = _collect_files(tmp_path, max_files=100)
        assert "real.py" in files
        assert not any(".venv" in f for f in files)

    def test_ignores_node_modules(self, tmp_path: Path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("x")
        (tmp_path / "index.js").write_text("y")

        files, _ = _collect_files(tmp_path, max_files=100)
        assert "index.js" in files
        assert not any("node_modules" in f for f in files)

    def test_ignores_nested_pycache(self, tmp_path: Path):
        pkg = tmp_path / "src" / "__pycache__"
        pkg.mkdir(parents=True)
        (pkg / "mod.cpython-312.pyc").write_text("x")
        (tmp_path / "src" / "mod.py").write_text("y")

        files, _ = _collect_files(tmp_path, max_files=100)
        assert "src/mod.py" in files
        assert not any("__pycache__" in f for f in files)

    def test_truncation(self, tmp_path: Path):
        for i in range(10):
            (tmp_path / f"file_{i:02d}.txt").write_text("")

        files, truncated = _collect_files(tmp_path, max_files=5)
        assert truncated
        assert len(files) == 5

    def test_empty_dir(self, tmp_path: Path):
        files, truncated = _collect_files(tmp_path, max_files=100)
        assert files == []
        assert not truncated

    def test_sorted_output(self, tmp_path: Path):
        for name in ["c.txt", "a.txt", "b.txt"]:
            (tmp_path / name).write_text("")

        files, _ = _collect_files(tmp_path, max_files=100)
        assert files == sorted(files)

    def test_extra_ignore_files(self, tmp_path: Path):
        (tmp_path / "keep.py").write_text("")
        (tmp_path / "skip.bak").write_text("")

        files, _ = _collect_files(tmp_path, max_files=100, extra_ignore=["*.bak"])
        assert "keep.py" in files
        assert "skip.bak" not in files

    def test_extra_ignore_dirs(self, tmp_path: Path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_a.py").write_text("")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("")

        files, _ = _collect_files(tmp_path, max_files=100, extra_ignore=["tests/"])
        assert "src/main.py" in files
        assert not any("tests" in f for f in files)


# ==============================================================================
# _render_tree
# ==============================================================================


class TestRenderTree:
    def test_flat_files(self):
        result = _render_tree(["a.txt", "b.txt"])
        assert result == "a.txt\nb.txt"

    def test_nested_structure(self):
        result = _render_tree(["src/foo.py", "src/bar/baz.ts", "README.md"])
        lines = result.splitlines()
        assert "README.md" in lines
        assert "src/" in lines
        # Children of src/ are indented.
        assert any("  foo.py" in line for line in lines)
        assert any("  bar/" in line for line in lines)
        assert any("    baz.ts" in line for line in lines)

    def test_directories_before_files(self):
        """Directories sort before files at each level."""
        result = _render_tree(["z.txt", "a_dir/x.txt"])
        lines = result.splitlines()
        dir_idx = next(i for i, l in enumerate(lines) if "a_dir/" in l)
        file_idx = next(i for i, l in enumerate(lines) if l.strip() == "z.txt")
        assert dir_idx < file_idx

    def test_empty_list(self):
        assert _render_tree([]) == ""


# ==============================================================================
# IGNORE_DIRS
# ==============================================================================


class TestIgnoreDirs:
    """Sanity checks that important directories are in the ignore set."""

    def test_contains_common_entries(self):
        for name in [".git", "node_modules", ".venv", "__pycache__", "dist"]:
            assert name in IGNORE_DIRS
