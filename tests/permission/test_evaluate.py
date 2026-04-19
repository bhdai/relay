"""Tests for relay.permission.evaluate — wildcard_match and evaluate."""

from __future__ import annotations

import pytest

from relay.permission.evaluate import evaluate, wildcard_match
from relay.permission.schema import PermissionRule


# ==============================================================================
# wildcard_match
# ==============================================================================


class TestWildcardMatch:
    """Unit tests for the wildcard_match helper."""

    # ------------------------------------------------------------------
    # Exact matches
    # ------------------------------------------------------------------

    def test_exact_match(self):
        assert wildcard_match("bash", "bash") is True

    def test_exact_mismatch(self):
        assert wildcard_match("bash", "edit") is False

    # ------------------------------------------------------------------
    # Star wildcard
    # ------------------------------------------------------------------

    def test_star_matches_everything(self):
        assert wildcard_match("git push --force origin main", "*") is True

    def test_star_matches_empty_string(self):
        assert wildcard_match("", "*") is True

    def test_star_prefix_matches(self):
        assert wildcard_match("src/foo/bar.py", "src/*") is True

    def test_star_mid_path(self):
        assert wildcard_match("src/foo/bar.py", "src/*/bar.py") is True

    def test_star_does_not_match_wrong_prefix(self):
        assert wildcard_match("tests/foo.py", "src/*") is False

    # ------------------------------------------------------------------
    # Double-star (glob-style) — * matches / because we use .* in regex
    # ------------------------------------------------------------------

    def test_double_star_matches_nested_paths(self):
        assert wildcard_match("src/foo/bar/baz.py", "src/**") is True

    def test_double_star_pattern(self):
        assert wildcard_match("a/b/c/d.txt", "**/*.txt") is True

    # ------------------------------------------------------------------
    # Question-mark wildcard
    # ------------------------------------------------------------------

    def test_question_mark_matches_single_char(self):
        assert wildcard_match("abc", "a?c") is True

    def test_question_mark_does_not_match_empty(self):
        assert wildcard_match("ac", "a?c") is False

    def test_question_mark_does_not_match_multiple(self):
        assert wildcard_match("aXXc", "a?c") is False

    # ------------------------------------------------------------------
    # Trailing `` *`` — optional suffix
    # ------------------------------------------------------------------

    def test_trailing_space_star_matches_bare_command(self):
        """``"git push *"`` must match ``"git push"`` (no args)."""
        assert wildcard_match("git push", "git push *") is True

    def test_trailing_space_star_matches_with_args(self):
        assert wildcard_match("git push --force origin main", "git push *") is True

    def test_trailing_space_star_no_false_positive(self):
        """``"git push *"`` should NOT match ``"git commit"``."""
        assert wildcard_match("git commit -m 'msg'", "git push *") is False

    def test_trailing_space_star_with_single_arg(self):
        assert wildcard_match("npm install foo", "npm install *") is True

    # ------------------------------------------------------------------
    # Path separator normalisation
    # ------------------------------------------------------------------

    def test_backslash_normalised_to_forward_slash(self):
        """Windows-style paths must be normalised before matching."""
        assert wildcard_match("src\\foo\\bar.py", "src/*") is True

    # ------------------------------------------------------------------
    # Regex special characters in text
    # ------------------------------------------------------------------

    def test_dot_in_text_is_literal(self):
        assert wildcard_match("file.py", "file.py") is True

    def test_dot_in_pattern_is_literal(self):
        """A literal ``.`` in the pattern must not match any character."""
        assert wildcard_match("filepy", "file.py") is False

    def test_regex_specials_in_text(self):
        assert wildcard_match("(a+b)", "(*") is True

    def test_regex_specials_in_pattern_escaped(self):
        assert wildcard_match("file[0].py", "file[0].py") is True

    # ------------------------------------------------------------------
    # Case sensitivity (Linux)
    # ------------------------------------------------------------------

    def test_case_sensitive_mismatch(self):
        assert wildcard_match("Bash", "bash") is False

    def test_case_sensitive_match(self):
        assert wildcard_match("bash", "bash") is True


# ==============================================================================
# evaluate
# ==============================================================================


class TestEvaluate:
    """Unit tests for evaluate() — ruleset merging and last-match-wins."""

    def _rule(self, permission, pattern, action):
        return PermissionRule(permission=permission, pattern=pattern, action=action)

    # ------------------------------------------------------------------
    # Default behaviour
    # ------------------------------------------------------------------

    def test_no_rulesets_returns_default_ask(self):
        result = evaluate("bash", "ls -la")
        assert result.action == "ask"

    def test_empty_rulesets_returns_default_ask(self):
        result = evaluate("bash", "ls -la", [], [])
        assert result.action == "ask"

    # ------------------------------------------------------------------
    # Single-ruleset basics
    # ------------------------------------------------------------------

    def test_matching_allow_rule(self):
        ruleset = [self._rule("bash", "*", "allow")]
        result = evaluate("bash", "ls -la", ruleset)
        assert result.action == "allow"

    def test_matching_deny_rule(self):
        ruleset = [self._rule("bash", "rm -rf *", "deny")]
        result = evaluate("bash", "rm -rf /", ruleset)
        assert result.action == "deny"

    def test_no_match_in_single_ruleset_returns_ask(self):
        ruleset = [self._rule("edit", "*", "allow")]
        result = evaluate("bash", "ls", ruleset)
        assert result.action == "ask"

    # ------------------------------------------------------------------
    # Last-match-wins ordering
    # ------------------------------------------------------------------

    def test_later_deny_overrides_earlier_allow(self):
        """[allow *, deny git*] on "git push" → deny."""
        ruleset = [
            self._rule("bash", "*", "allow"),
            self._rule("bash", "git*", "deny"),
        ]
        result = evaluate("bash", "git push", ruleset)
        assert result.action == "deny"

    def test_later_allow_overrides_earlier_deny(self):
        """[deny *, allow git status*] on "git status" → allow."""
        ruleset = [
            self._rule("bash", "*", "deny"),
            self._rule("bash", "git status *", "allow"),
        ]
        result = evaluate("bash", "git status", ruleset)
        assert result.action == "allow"

    def test_last_matching_rule_is_returned_not_first(self):
        ruleset = [
            self._rule("bash", "*", "ask"),
            self._rule("bash", "*", "allow"),
        ]
        result = evaluate("bash", "echo hi", ruleset)
        assert result.action == "allow"

    # ------------------------------------------------------------------
    # Multiple rulesets (merged in order)
    # ------------------------------------------------------------------

    def test_second_ruleset_overrides_first(self):
        first = [self._rule("bash", "*", "allow")]
        second = [self._rule("bash", "*", "deny")]
        result = evaluate("bash", "ls", first, second)
        assert result.action == "deny"

    def test_first_ruleset_applies_when_no_match_in_second(self):
        first = [self._rule("bash", "*", "allow")]
        second = [self._rule("edit", "*", "deny")]
        result = evaluate("bash", "ls", first, second)
        assert result.action == "allow"

    # ------------------------------------------------------------------
    # Permission key wildcards
    # ------------------------------------------------------------------

    def test_star_permission_key_matches_bash(self):
        ruleset = [self._rule("*", "*", "allow")]
        result = evaluate("bash", "anything", ruleset)
        assert result.action == "allow"

    def test_star_permission_key_matches_edit(self):
        ruleset = [self._rule("*", "*", "ask")]
        result = evaluate("edit", "src/main.py", ruleset)
        assert result.action == "ask"

    # ------------------------------------------------------------------
    # Path patterns
    # ------------------------------------------------------------------

    def test_path_pattern_matches_nested(self):
        ruleset = [self._rule("read", "src/**", "allow")]
        result = evaluate("read", "src/foo/bar.py", ruleset)
        assert result.action == "allow"

    def test_path_pattern_no_match_outside_src(self):
        ruleset = [self._rule("read", "src/**", "allow")]
        result = evaluate("read", "tests/foo.py", ruleset)
        assert result.action == "ask"  # default

    # ------------------------------------------------------------------
    # Sensitive-file override example
    # ------------------------------------------------------------------

    def test_env_file_denied_despite_broad_allow(self):
        """read=allow for * but read=ask for *.env files."""
        ruleset = [
            self._rule("read", "*", "allow"),
            self._rule("read", "*.env", "ask"),
        ]
        result = evaluate("read", ".env", ruleset)
        assert result.action == "ask"

    def test_non_env_file_allowed(self):
        ruleset = [
            self._rule("read", "*", "allow"),
            self._rule("read", "*.env", "ask"),
        ]
        result = evaluate("read", "src/main.py", ruleset)
        assert result.action == "allow"

    def test_env_example_explicitly_allowed(self):
        ruleset = [
            self._rule("read", "*", "allow"),
            self._rule("read", "*.env", "ask"),
            self._rule("read", "*.env.example", "allow"),
        ]
        result = evaluate("read", ".env.example", ruleset)
        assert result.action == "allow"

    # ------------------------------------------------------------------
    # Return value is the matched rule, not a copy
    # ------------------------------------------------------------------

    def test_returned_rule_contains_original_values(self):
        rule = self._rule("bash", "git push *", "deny")
        ruleset = [rule]
        result = evaluate("bash", "git push --force", ruleset)
        assert result.action == "deny"
        assert result.permission == "bash"
        assert result.pattern == "git push *"
