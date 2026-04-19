"""Tests for relay.permission.config — normalization and migration."""

from __future__ import annotations

import os

import pytest

from relay.permission.config import (
    DEFAULT_PERMISSION,
    READONLY_PERMISSION,
    from_config,
    merge,
    migrate_from_approval_config,
)
from relay.permission.evaluate import evaluate
from relay.permission.schema import PermissionRule


# ==============================================================================
# from_config — shorthand form
# ==============================================================================


class TestFromConfigShorthand:
    """Shorthand YAML: ``"bash": "allow"`` → single Rule with pattern ``"*"``."""

    def test_allow_shorthand(self):
        rules = from_config({"bash": "allow"})
        assert len(rules) == 1
        assert rules[0] == PermissionRule(permission="bash", pattern="*", action="allow")

    def test_deny_shorthand(self):
        rules = from_config({"edit": "deny"})
        assert rules[0].action == "deny"
        assert rules[0].pattern == "*"

    def test_ask_shorthand(self):
        rules = from_config({"task": "ask"})
        assert rules[0].action == "ask"

    def test_wildcard_permission_key(self):
        rules = from_config({"*": "allow"})
        assert rules[0] == PermissionRule(permission="*", pattern="*", action="allow")

    def test_multiple_shorthand_keys_ordering(self):
        """Insertion order is preserved — critical for last-match-wins."""
        rules = from_config({"*": "allow", "bash": "ask"})
        assert rules[0].permission == "*"
        assert rules[1].permission == "bash"

    def test_unknown_permission_key_accepted(self):
        """Unknown keys (future extensions) must not raise."""
        rules = from_config({"doom_loop": "ask"})
        assert rules[0].permission == "doom_loop"

    def test_empty_dict_returns_empty_list(self):
        assert from_config({}) == []

    def test_invalid_action_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid permission action"):
            from_config({"bash": "permit"})

    def test_non_string_non_dict_value_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a string or dict"):
            from_config({"bash": 42})  # type: ignore[dict-item]


# ==============================================================================
# from_config — pattern map form
# ==============================================================================


class TestFromConfigPatternMap:
    """Pattern map YAML: ``"read": {"*.env": "ask", "*": "allow"}``."""

    def test_single_pattern_entry(self):
        rules = from_config({"bash": {"git push *": "ask"}})
        assert len(rules) == 1
        assert rules[0] == PermissionRule(
            permission="bash", pattern="git push *", action="ask"
        )

    def test_multiple_patterns_preserve_order(self):
        """Pattern insertion order is preserved for last-match-wins correctness."""
        rules = from_config(
            {
                "read": {
                    "*.env": "ask",
                    "*.env.*": "ask",
                    "*.env.example": "allow",
                    "*": "allow",
                }
            }
        )
        assert len(rules) == 4
        assert rules[0].pattern == "*.env"
        assert rules[0].action == "ask"
        assert rules[1].pattern == "*.env.*"
        assert rules[2].pattern == "*.env.example"
        assert rules[2].action == "allow"
        assert rules[3].pattern == "*"
        assert rules[3].action == "allow"

    def test_pattern_map_empty_dict_produces_no_rules(self):
        rules = from_config({"read": {}})
        assert rules == []

    def test_invalid_action_in_pattern_map_raises(self):
        with pytest.raises(ValueError, match="Invalid permission action"):
            from_config({"bash": {"git *": "execute"}})

    def test_mixed_shorthand_and_pattern_map(self):
        """Shorthand and pattern-map keys can coexist in the same config."""
        rules = from_config(
            {
                "*": "allow",
                "read": {"*.env": "ask", "*": "allow"},
            }
        )
        assert len(rules) == 3
        assert rules[0] == PermissionRule(permission="*", pattern="*", action="allow")
        assert rules[1].permission == "read"
        assert rules[1].pattern == "*.env"
        assert rules[2].permission == "read"
        assert rules[2].pattern == "*"


# ==============================================================================
# from_config — path expansion
# ==============================================================================


class TestFromConfigPathExpansion:
    """``~`` and ``$HOME`` in patterns are expanded to absolute paths."""

    def test_tilde_expansion_in_pattern_map(self):
        rules = from_config({"read": {"~/secret.txt": "deny"}})
        home = os.path.expanduser("~")
        assert rules[0].pattern == f"{home}/secret.txt"

    def test_env_var_expansion_in_pattern_map(self, monkeypatch):
        monkeypatch.setenv("MYDIR", "/some/dir")
        rules = from_config({"edit": {"$MYDIR/file.py": "ask"}})
        assert rules[0].pattern == "/some/dir/file.py"

    def test_expansion_does_not_apply_to_shorthand_action(self):
        """Shorthand produces pattern ``"*"`` — expansion is a no-op there."""
        rules = from_config({"bash": "allow"})
        assert rules[0].pattern == "*"


# ==============================================================================
# merge
# ==============================================================================


class TestMerge:
    """merge() concatenates rulesets; later entries win over earlier ones."""

    def test_merge_two_rulesets(self):
        rs1 = from_config({"*": "allow"})
        rs2 = from_config({"bash": "deny"})
        merged = merge(rs1, rs2)
        assert len(merged) == 2
        assert merged[0].permission == "*"
        assert merged[1].permission == "bash"

    def test_merge_order_produces_correct_last_match_wins(self):
        """A deny rule in rs2 overrides the allow rule in rs1 for same key."""
        rs1 = from_config({"bash": "allow"})
        rs2 = from_config({"bash": "deny"})
        merged = merge(rs1, rs2)
        # evaluate uses last-match-wins so "bash" → deny
        result = evaluate("bash", "*", merged)
        assert result.action == "deny"

    def test_merge_no_args_returns_empty(self):
        assert merge() == []

    def test_merge_single_ruleset_returns_copy(self):
        rs = from_config({"bash": "allow"})
        merged = merge(rs)
        assert merged == rs
        assert merged is not rs  # must be a new list

    def test_merge_preserves_all_entries(self):
        rs1 = from_config({"*": "allow", "doom_loop": "ask"})
        rs2 = from_config({"read": {"*.env": "ask"}})
        merged = merge(rs1, rs2)
        assert len(merged) == 3


# ==============================================================================
# DEFAULT_PERMISSION evaluation smoke tests
# ==============================================================================


class TestDefaultPermission:
    """DEFAULT_PERMISSION produces expected evaluation results."""

    def setup_method(self):
        self.ruleset = from_config(DEFAULT_PERMISSION)

    def test_bash_defaults_to_allow(self):
        """Generic bash call is allowed (base "*": "allow" rule)."""
        result = evaluate("bash", "ls -la", self.ruleset)
        assert result.action == "allow"

    def test_edit_defaults_to_allow(self):
        result = evaluate("edit", "src/main.py", self.ruleset)
        assert result.action == "allow"

    def test_read_normal_file_allowed(self):
        result = evaluate("read", "src/utils.py", self.ruleset)
        assert result.action == "allow"

    def test_read_env_file_asks(self):
        """Reading ``.env`` files should require confirmation."""
        result = evaluate("read", ".env", self.ruleset)
        assert result.action == "ask"

    def test_read_env_prefixed_file_asks(self):
        result = evaluate("read", ".env.production", self.ruleset)
        assert result.action == "ask"

    def test_read_env_example_allowed(self):
        """Example env files are explicitly re-allowed after the ask rules."""
        result = evaluate("read", ".env.example", self.ruleset)
        assert result.action == "allow"

    def test_doom_loop_asks(self):
        result = evaluate("doom_loop", "*", self.ruleset)
        assert result.action == "ask"

    def test_external_directory_asks(self):
        result = evaluate("external_directory", "/tmp/outside", self.ruleset)
        assert result.action == "ask"


# ==============================================================================
# READONLY_PERMISSION evaluation smoke tests
# ==============================================================================


class TestReadonlyPermission:
    """READONLY_PERMISSION denies writes; allows read/glob/grep/list/web/bash."""

    def setup_method(self):
        self.ruleset = from_config(READONLY_PERMISSION)

    def test_read_is_allowed(self):
        result = evaluate("read", "any/file.py", self.ruleset)
        assert result.action == "allow"

    def test_glob_is_allowed(self):
        result = evaluate("glob", "**/*.py", self.ruleset)
        assert result.action == "allow"

    def test_grep_is_allowed(self):
        result = evaluate("grep", "TODO", self.ruleset)
        assert result.action == "allow"

    def test_list_is_allowed(self):
        result = evaluate("list", "src/", self.ruleset)
        assert result.action == "allow"

    def test_bash_is_allowed(self):
        result = evaluate("bash", "ls -la", self.ruleset)
        assert result.action == "allow"

    def test_web_is_allowed(self):
        result = evaluate("web", "https://example.com", self.ruleset)
        assert result.action == "allow"

    def test_edit_is_denied(self):
        result = evaluate("edit", "src/main.py", self.ruleset)
        assert result.action == "deny"

    def test_task_is_denied(self):
        result = evaluate("task", "general-purpose", self.ruleset)
        assert result.action == "deny"


# ==============================================================================
# AgentContext serialization round-trip
# ==============================================================================


class TestAgentContextRoundTrip:
    """permission_ruleset field survives Pydantic serialize → reconstruct."""

    def test_default_permission_ruleset_is_empty_list(self):
        from relay.agents.context import AgentContext

        ctx = AgentContext()
        assert ctx.permission_ruleset == []

    def test_permission_ruleset_set_and_retrieved(self):
        from relay.agents.context import AgentContext

        rules = [{"permission": "bash", "pattern": "git push *", "action": "allow"}]
        ctx = AgentContext(permission_ruleset=rules)
        assert ctx.permission_ruleset == rules

    def test_permission_ruleset_round_trips_via_model_dump(self):
        from relay.agents.context import AgentContext
        from relay.permission.schema import PermissionRule

        rules_input = [
            PermissionRule(permission="bash", pattern="git push *", action="allow").model_dump(),
            PermissionRule(permission="read", pattern="*.env", action="ask").model_dump(),
        ]
        ctx = AgentContext(permission_ruleset=rules_input)
        dumped = ctx.model_dump()
        restored = AgentContext(**dumped)
        assert restored.permission_ruleset == rules_input

    def test_approval_mode_field_removed(self):
        """approval_mode must no longer exist on AgentContext."""
        from relay.agents.context import AgentContext

        ctx = AgentContext()
        assert not hasattr(ctx, "approval_mode")


# ==============================================================================
# migrate_from_approval_config
# ==============================================================================


class TestMigrateFromApprovalConfig:
    """Legacy ToolApprovalConfig converts to equivalent Ruleset."""

    def _make_tool_rule(self, name: str, args: dict | None = None):
        from relay.configs.approval import ToolApprovalRule

        return ToolApprovalRule(name=name, args=args)

    def _make_approval_config(
        self,
        always_allow=None,
        always_deny=None,
        always_ask=None,
    ):
        from relay.configs.approval import ToolApprovalConfig

        return ToolApprovalConfig(
            always_allow=always_allow or [],
            always_deny=always_deny or [],
            always_ask=always_ask or [],
        )

    def test_always_allow_run_command_becomes_bash_allow(self):
        cfg = self._make_approval_config(
            always_allow=[self._make_tool_rule("run_command", {"command": "git status"})]
        )
        rules = migrate_from_approval_config(cfg)
        assert len(rules) == 1
        assert rules[0].permission == "bash"
        assert rules[0].pattern == "git status"
        assert rules[0].action == "allow"

    def test_always_deny_run_command_becomes_bash_deny(self):
        cfg = self._make_approval_config(
            always_deny=[self._make_tool_rule("run_command", {"command": r"rm -rf.*"})]
        )
        rules = migrate_from_approval_config(cfg)
        assert rules[0].permission == "bash"
        assert rules[0].action == "deny"

    def test_always_ask_run_command_becomes_bash_ask(self):
        cfg = self._make_approval_config(
            always_ask=[self._make_tool_rule("run_command", {"command": r"git push.*"})]
        )
        rules = migrate_from_approval_config(cfg)
        assert rules[0].permission == "bash"
        assert rules[0].action == "ask"

    def test_unknown_tool_name_maps_to_itself(self):
        cfg = self._make_approval_config(
            always_allow=[self._make_tool_rule("my_custom_tool")]
        )
        rules = migrate_from_approval_config(cfg)
        assert rules[0].permission == "my_custom_tool"
        assert rules[0].pattern == "*"

    def test_rule_without_args_uses_wildcard_pattern(self):
        cfg = self._make_approval_config(
            always_allow=[self._make_tool_rule("run_command")]
        )
        rules = migrate_from_approval_config(cfg)
        assert rules[0].pattern == "*"

    def test_ordering_allow_before_deny_before_ask(self):
        """Migration preserves the always_allow / always_deny / always_ask order."""
        cfg = self._make_approval_config(
            always_allow=[self._make_tool_rule("read_file", {"path": "src/*"})],
            always_deny=[self._make_tool_rule("run_command", {"command": "rm -rf *"})],
            always_ask=[self._make_tool_rule("run_command", {"command": "git push *"})],
        )
        rules = migrate_from_approval_config(cfg)
        assert len(rules) == 3
        assert rules[0].action == "allow"
        assert rules[1].action == "deny"
        assert rules[2].action == "ask"

    def test_empty_approval_config_produces_empty_ruleset(self):
        cfg = self._make_approval_config()
        assert migrate_from_approval_config(cfg) == []

    def test_read_file_tool_maps_to_read_permission(self):
        cfg = self._make_approval_config(
            always_allow=[self._make_tool_rule("read_file", {"path": "src/**"})]
        )
        rules = migrate_from_approval_config(cfg)
        assert rules[0].permission == "read"
        assert rules[0].pattern == "src/**"
