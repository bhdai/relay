"""Unit tests for custom state reducers."""

from relay.agents.state import file_reducer, replace_reducer, sum_reducer


class TestFileReducer:
    def test_both_none(self):
        assert file_reducer(None, None) == {}

    def test_left_none(self):
        assert file_reducer(None, {"a": "1"}) == {"a": "1"}

    def test_right_none(self):
        assert file_reducer({"a": "1"}, None) == {"a": "1"}

    def test_merge_non_overlapping(self):
        result = file_reducer({"a": "1"}, {"b": "2"})
        assert result == {"a": "1", "b": "2"}

    def test_merge_overlapping_right_wins(self):
        left = {"a": "old", "b": "keep"}
        right = {"a": "new", "c": "added"}
        result = file_reducer(left, right)
        assert result == {"a": "new", "b": "keep", "c": "added"}

    def test_empty_dicts(self):
        assert file_reducer({}, {}) == {}

    def test_does_not_mutate_inputs(self):
        left = {"a": "1"}
        right = {"b": "2"}
        left_copy, right_copy = left.copy(), right.copy()
        file_reducer(left, right)
        assert left == left_copy
        assert right == right_copy


class TestSumReducer:
    def test_both_values(self):
        assert sum_reducer(1.5, 2.5) == 4.0

    def test_left_none(self):
        assert sum_reducer(None, 3.0) == 3.0

    def test_right_none(self):
        assert sum_reducer(10.0, None) == 10.0

    def test_both_none(self):
        assert sum_reducer(None, None) == 0.0

    def test_precision(self):
        assert abs(sum_reducer(0.1, 0.2) - 0.3) < 1e-10


class TestReplaceReducer:
    def test_both_values(self):
        assert replace_reducer(10, 20) == 20

    def test_left_none(self):
        assert replace_reducer(None, 20) == 20

    def test_right_none(self):
        assert replace_reducer(10, None) == 10

    def test_both_none(self):
        assert replace_reducer(None, None) == 0

    def test_zero_replaces(self):
        assert replace_reducer(100, 0) == 0

    def test_cumulative_token_scenario(self):
        """replace_reducer should keep the latest cumulative value, not sum."""
        state_tokens = 6307
        new_tokens = 6322
        assert replace_reducer(state_tokens, new_tokens) == 6322
