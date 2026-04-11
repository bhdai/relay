"""Unit tests for filesystem tool helpers — no I/O required."""

import pytest

from relay.tools.filesystem import (
    EditOperation,
    _apply_edits,
    _find_match,
    _paginate_file,
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
