"""Tests for relay.mcp.registry — tool filtering and module mapping."""

import pytest

from relay.mcp.registry import MCPRegistry


class TestMCPRegistry:
    def test_allowed_no_filter(self):
        registry = MCPRegistry()
        assert registry.allowed("any_tool", "server_a") is True

    def test_allowed_include_filter(self):
        registry = MCPRegistry(
            filters={"server_a": {"include": ["tool_a", "tool_b"]}}
        )
        assert registry.allowed("tool_a", "server_a") is True
        assert registry.allowed("tool_c", "server_a") is False

    def test_allowed_exclude_filter(self):
        registry = MCPRegistry(
            filters={"server_a": {"exclude": ["tool_x"]}}
        )
        assert registry.allowed("tool_a", "server_a") is True
        assert registry.allowed("tool_x", "server_a") is False

    def test_allowed_both_include_exclude_raises(self):
        registry = MCPRegistry(
            filters={"server_a": {"include": ["tool_a"], "exclude": ["tool_b"]}}
        )
        with pytest.raises(ValueError, match="Both include/exclude"):
            registry.allowed("tool_a", "server_a")

    def test_register_tracks_tools(self):
        registry = MCPRegistry()
        assert registry.register("tool_a", "server_a") is True
        assert registry.register("tool_a", "server_a") is False  # duplicate

    def test_module_map_uses_server_prefix(self):
        registry = MCPRegistry()
        registry.register("tool_a", "server_a")
        registry.register("tool_b", "server_b")

        module_map = registry.module_map
        assert module_map["server_a__tool_a"] == "server_a"
        assert module_map["server_b__tool_b"] == "server_b"

    def test_allowed_different_server_no_filter(self):
        registry = MCPRegistry(
            filters={"server_a": {"include": ["tool_a"]}}
        )
        # server_b has no filters, so all tools are allowed.
        assert registry.allowed("any_tool", "server_b") is True
