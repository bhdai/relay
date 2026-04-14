"""Tests for relay.utils.patterns — wildcard and negative pattern matching."""

from relay.utils.patterns import matches_patterns, two_part_matcher


# ==============================================================================
# matches_patterns
# ==============================================================================


class TestMatchesPatterns:
    def test_positive_match(self):
        matcher = lambda p: p == "hello"
        assert matches_patterns(["hello"], matcher) is True

    def test_positive_no_match(self):
        matcher = lambda p: p == "hello"
        assert matches_patterns(["world"], matcher) is False

    def test_negative_excludes(self):
        """A matching negative pattern vetoes the result."""
        matcher = lambda p: p == "hello"
        assert matches_patterns(["hello", "!hello"], matcher) is False

    def test_negative_does_not_affect_unmatched(self):
        """A negative pattern that doesn't match leaves the positive in effect."""
        matcher = lambda p: p == "hello"
        assert matches_patterns(["hello", "!world"], matcher) is True

    def test_no_positive_patterns_returns_false(self):
        """If every pattern is negative, there's nothing to include."""
        matcher = lambda p: True
        assert matches_patterns(["!hello"], matcher) is False

    def test_empty_patterns_returns_false(self):
        matcher = lambda p: True
        assert matches_patterns([], matcher) is False

    def test_multiple_positives_any_matches(self):
        """Only one positive needs to match."""
        matcher = lambda p: p == "b"
        assert matches_patterns(["a", "b", "c"], matcher) is True

    def test_multiple_negatives_any_excludes(self):
        """Any matching negative vetoes the result."""
        matcher = lambda p: p in {"a", "b"}
        assert matches_patterns(["a", "!b"], matcher) is False


# ==============================================================================
# two_part_matcher
# ==============================================================================


class TestTwoPartMatcher:
    def test_exact_match(self):
        matcher = two_part_matcher("read_file", "file_system")
        assert matcher("file_system:read_file") is True

    def test_no_match(self):
        matcher = two_part_matcher("read_file", "file_system")
        assert matcher("terminal:run_command") is False

    def test_wildcard_name(self):
        matcher = two_part_matcher("read_file", "file_system")
        assert matcher("file_system:*") is True

    def test_wildcard_module(self):
        matcher = two_part_matcher("read_file", "file_system")
        assert matcher("*:read_file") is True

    def test_wildcard_both(self):
        matcher = two_part_matcher("anything", "whatever")
        assert matcher("*:*") is True

    def test_glob_prefix(self):
        matcher = two_part_matcher("read_file", "file_system")
        assert matcher("file_system:read_*") is True

    def test_glob_prefix_no_match(self):
        matcher = two_part_matcher("write_file", "file_system")
        assert matcher("file_system:read_*") is False

    def test_invalid_format_returns_false(self):
        matcher = two_part_matcher("read_file", "file_system")
        assert matcher("only_one_part") is False

    def test_invalid_format_calls_callback(self):
        invalids = []
        matcher = two_part_matcher(
            "read_file", "file_system", on_invalid=invalids.append
        )
        matcher("bad")
        assert invalids == ["bad"]

    def test_three_parts_returns_false(self):
        matcher = two_part_matcher("read_file", "file_system")
        assert matcher("impl:file_system:read_file") is False
