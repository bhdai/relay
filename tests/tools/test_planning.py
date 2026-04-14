"""Unit tests for planning tools."""

from relay.tools.planning import PLANNING_TOOLS, think


class TestThink:
    def test_returns_reflection_summary(self):
        result = think.func(reflection="Need one more code search before editing")
        assert result == "Reflection recorded: Need one more code search before editing"

    def test_exported_in_planning_tools(self):
        assert PLANNING_TOOLS == [think]