"""Tests for relay.configs.approval."""

import json
import tempfile
from pathlib import Path

import pytest

from relay.configs.approval import (
    ApprovalMode,
    ToolApprovalConfig,
    ToolApprovalRule,
    _default_always_ask_rules,
)


class TestToolApprovalRule:
    """Tests for ToolApprovalRule.matches_call."""

    def test_name_must_match(self):
        rule = ToolApprovalRule(name="foo")
        assert rule.matches_call("foo", {}) is True
        assert rule.matches_call("bar", {}) is False

    def test_matches_without_args(self):
        rule = ToolApprovalRule(name="foo", args=None)
        assert rule.matches_call("foo", {"anything": "value"}) is True

    def test_exact_arg_match(self):
        rule = ToolApprovalRule(name="foo", args={"key": "val"})
        assert rule.matches_call("foo", {"key": "val"}) is True
        assert rule.matches_call("foo", {"key": "other"}) is False

    def test_regex_arg_match(self):
        rule = ToolApprovalRule(name="run_command", args={"command": r"rm\s+-rf.*"})
        assert rule.matches_call("run_command", {"command": "rm -rf /"}) is True
        assert rule.matches_call("run_command", {"command": "ls -la"}) is False

    def test_missing_arg_key(self):
        rule = ToolApprovalRule(name="foo", args={"missing": "val"})
        assert rule.matches_call("foo", {"other": "val"}) is False

    def test_invalid_regex_falls_through(self):
        rule = ToolApprovalRule(name="foo", args={"key": "[invalid"})
        assert rule.matches_call("foo", {"key": "nope"}) is False


class TestToolApprovalConfig:
    """Tests for ToolApprovalConfig persistence."""

    def test_creates_file_with_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.approval.json"
            config = ToolApprovalConfig.from_json_file(config_file)

            assert config_file.exists()
            assert len(config.always_ask) == len(_default_always_ask_rules())

    def test_loads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.approval.json"
            data = {
                "always_allow": [{"name": "safe_tool"}],
                "always_deny": [],
                "always_ask": [],
            }
            config_file.write_text(json.dumps(data))

            config = ToolApprovalConfig.from_json_file(config_file)
            assert len(config.always_allow) == 1
            assert config.always_allow[0].name == "safe_tool"

    def test_adds_default_always_ask_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.approval.json"
            data = {"always_allow": [], "always_deny": []}
            config_file.write_text(json.dumps(data))

            config = ToolApprovalConfig.from_json_file(config_file)
            assert len(config.always_ask) > 0

    def test_save_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.approval.json"
            config = ToolApprovalConfig(
                always_allow=[ToolApprovalRule(name="a")],
                always_deny=[ToolApprovalRule(name="b", args={"x": "1"})],
            )
            config.save_to_json_file(config_file)

            loaded = ToolApprovalConfig.from_json_file(config_file)
            assert len(loaded.always_allow) == 1
            assert loaded.always_allow[0].name == "a"
            assert len(loaded.always_deny) == 1
            assert loaded.always_deny[0].args == {"x": "1"}


class TestApprovalMode:
    """Basic enum checks."""

    def test_values(self):
        assert ApprovalMode.SEMI_ACTIVE.value == "semi-active"
        assert ApprovalMode.ACTIVE.value == "active"
        assert ApprovalMode.AGGRESSIVE.value == "aggressive"
