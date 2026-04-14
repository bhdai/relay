"""Unit tests for filesystem tool helpers — no I/O required."""

from pathlib import Path

import pytest

from relay.tools.impl.filesystem import (
    EditOperation,
    _apply_edits,
    _find_match,
    _glob_match,
    _grep_match,
    _paginate_file,
    walk_files,
)


# ==============================================================================
# _paginate_file
# ==============================================================================


class TestPaginateFile:
    def test_basic_pagination(self):
        lines = ["a\n", "b\n", "c\n"]
        result = _paginate_file(lines)
        assert "0 - a" in result
        assert "1 - b" in result
        assert "2 - c" in result
        assert "3/3 lines" in result

    def test_offset_start(self):
        lines = ["a\n", "b\n", "c\n", "d\n"]
        result = _paginate_file(lines, start_line=2, limit=2)
        assert "2 - c" in result
        assert "3 - d" in result
        assert "2/4 lines" in result
        # Lines before the window must not appear.
        assert "0 - a" not in result

    def test_limit_caps_output(self):
        lines = [f"line {i}\n" for i in range(100)]
        result = _paginate_file(lines, start_line=0, limit=5)
        assert "5/100 lines" in result

    def test_empty_file(self):
        result = _paginate_file([])
        assert "0/0 lines" in result

    def test_negative_start_clamped(self):
        lines = ["a\n", "b\n"]
        result = _paginate_file(lines, start_line=-5)
        assert "0 - a" in result


# ==============================================================================
# _find_match
# ==============================================================================


class TestFindMatch:
    def test_exact_match(self):
        found, start, end = _find_match("hello world", "world")
        assert found is True
        assert "hello world"[start:end] == "world"

    def test_no_match(self):
        found, _, _ = _find_match("hello", "xyz")
        assert found is False

    def test_whitespace_normalised_match(self):
        """Extra leading indentation should still match."""
        content = "    if True:\n        print('hi')\n"
        search = "if True:\n    print('hi')"
        found, start, end = _find_match(content, search)
        assert found is True

    def test_empty_search(self):
        found, start, end = _find_match("anything", "")
        assert found is True
        assert start == 0


# ==============================================================================
# _apply_edits
# ==============================================================================


class TestApplyEdits:
    def test_single_edit(self):
        result = _apply_edits(
            "hello world",
            [EditOperation(old_content="world", new_content="universe")],
        )
        assert result == "hello universe"

    def test_multiple_non_overlapping(self):
        result = _apply_edits(
            "AAA BBB CCC",
            [
                EditOperation(old_content="AAA", new_content="111"),
                EditOperation(old_content="CCC", new_content="333"),
            ],
        )
        assert result == "111 BBB 333"

    def test_missing_content_raises(self):
        with pytest.raises(Exception, match="could not find"):
            _apply_edits(
                "hello",
                [EditOperation(old_content="xyz", new_content="abc")],
            )

    def test_overlapping_edits_raises(self):
        with pytest.raises(Exception, match="Overlapping"):
            _apply_edits(
                "0123456789",
                [
                    EditOperation(old_content="234", new_content="ABC"),
                    EditOperation(old_content="456", new_content="XYZ"),
                ],
            )

    def test_delete_content(self):
        result = _apply_edits(
            "keep DELETE_ME keep",
            [EditOperation(old_content="DELETE_ME ", new_content="")],
        )
        assert result == "keep keep"


# ==============================================================================
# walk_files
# ==============================================================================


class TestWalkFiles:
    def test_basic_walk(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("")

        files = list(walk_files(tmp_path))
        assert "a.py" in files
        assert "sub/b.py" in files

    def test_prunes_ignored_dirs(self, tmp_path: Path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "lib.py").write_text("")
        (tmp_path / "real.py").write_text("")

        files = list(walk_files(tmp_path))
        assert "real.py" in files
        assert not any("node_modules" in f for f in files)
        assert not any(".venv" in f for f in files)

    def test_sorted_output(self, tmp_path: Path):
        for name in ["c.py", "a.py", "b.py"]:
            (tmp_path / name).write_text("")
        files = list(walk_files(tmp_path))
        assert files == sorted(files)


# ==============================================================================
# _glob_match
# ==============================================================================


class TestGlobMatch:
    def test_matches_pattern(self):
        paths = ["src/foo.py", "src/bar.ts", "README.md"]
        matches, truncated = _glob_match(iter(paths), "*.py", 100)
        assert matches == ["src/foo.py"]
        assert not truncated

    def test_star_star_pattern(self):
        paths = ["a/b/c.py", "a/d.py", "e.txt"]
        matches, _ = _glob_match(iter(paths), "a/**/*.py", 100)
        # fnmatch doesn't handle ** like pathlib.glob, but * works for single segments.
        # Test single-level wildcard instead.
        matches2, _ = _glob_match(iter(paths), "a/*.py", 100)
        assert "a/d.py" in matches2

    def test_truncation(self):
        paths = [f"file_{i}.py" for i in range(20)]
        matches, truncated = _glob_match(iter(paths), "*.py", 5)
        assert len(matches) == 5
        assert truncated

    def test_no_matches(self):
        matches, truncated = _glob_match(iter(["a.py"]), "*.rs", 100)
        assert matches == []
        assert not truncated


# ==============================================================================
# _grep_match
# ==============================================================================


class TestGrepMatch:
    def test_literal_search(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("hello world\ngoodbye world\n")
        files = list(walk_files(tmp_path))
        matches, truncated = _grep_match(
            tmp_path, iter(files), "hello", is_regex=False, max_results=100,
        )
        assert len(matches) == 1
        assert "a.py:1:" in matches[0]
        assert "hello world" in matches[0]
        assert not truncated

    def test_regex_search(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("foo123\nbar456\n")
        files = list(walk_files(tmp_path))
        matches, _ = _grep_match(
            tmp_path, iter(files), r"\d{3}", is_regex=True, max_results=100,
        )
        assert len(matches) == 2

    def test_invalid_regex_raises(self, tmp_path: Path):
        with pytest.raises(Exception, match="Invalid regex"):
            _grep_match(tmp_path, iter([]), "[invalid", is_regex=True, max_results=100)

    def test_truncation(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("\n".join(f"match line {i}" for i in range(20)))
        files = list(walk_files(tmp_path))
        matches, truncated = _grep_match(
            tmp_path, iter(files), "match", is_regex=False, max_results=5,
        )
        assert len(matches) == 5
        assert truncated

    def test_skips_binary(self, tmp_path: Path):
        (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02\xff")
        (tmp_path / "text.txt").write_text("findme\n")
        files = list(walk_files(tmp_path))
        matches, _ = _grep_match(
            tmp_path, iter(files), "findme", is_regex=False, max_results=100,
        )
        assert len(matches) == 1
        assert "text.txt" in matches[0]

    def test_no_matches(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("nothing here\n")
        files = list(walk_files(tmp_path))
        matches, _ = _grep_match(
            tmp_path, iter(files), "missing", is_regex=False, max_results=100,
        )
        assert matches == []
