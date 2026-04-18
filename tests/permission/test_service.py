"""Tests for relay.permission.service — PermissionService ask/reply lifecycle."""

from __future__ import annotations

import pytest

from relay.permission.schema import (
    Allowed,
    Denied,
    NeedsAsk,
    PermissionRequest,
    PermissionRule,
)
from relay.permission.service import PermissionService


# ==============================================================================
# Helpers
# ==============================================================================


def _rule(permission: str, pattern: str, action: str) -> PermissionRule:
    return PermissionRule(permission=permission, pattern=pattern, action=action)


def _req(
    session_id: str = "session-1",
    permission: str = "bash",
    patterns: list[str] | None = None,
    always: list[str] | None = None,
    req_id: str | None = None,
) -> PermissionRequest:
    return PermissionRequest(
        **({"id": req_id} if req_id else {}),
        session_id=session_id,
        permission=permission,
        patterns=patterns or ["ls"],
        always=always or ["ls *"],
    )


# ==============================================================================
# ask() — basic decisions
# ==============================================================================


class TestAskBasicDecisions:
    """Tests for PermissionService.ask() decision paths."""

    def test_all_allow_returns_allowed(self):
        """When every pattern evaluates to allow, we get Allowed."""
        svc = PermissionService()
        ruleset = [_rule("bash", "*", "allow")]
        req = _req(patterns=["ls", "pwd"])
        result = svc.ask(req, ruleset)
        assert isinstance(result, Allowed)

    def test_any_deny_returns_denied_immediately(self):
        """A single deny rule terminates evaluation and returns Denied."""
        svc = PermissionService()
        ruleset = [_rule("bash", "rm -rf *", "deny")]
        req = _req(patterns=["rm -rf /"])
        result = svc.ask(req, ruleset)
        assert isinstance(result, Denied)
        assert "bash" in result.reason

    def test_deny_short_circuits_allow(self):
        """Deny takes priority over allow in the same ruleset."""
        svc = PermissionService()
        # deny rm -rf, allow everything else
        ruleset = [
            _rule("bash", "*", "allow"),
            _rule("bash", "rm -rf *", "deny"),
        ]
        req = _req(patterns=["rm -rf /"])
        result = svc.ask(req, ruleset)
        assert isinstance(result, Denied)

    def test_ask_rule_returns_needs_ask(self):
        """Ask rule causes NeedsAsk — request is registered as pending."""
        svc = PermissionService()
        ruleset = [_rule("bash", "*", "ask")]
        req = _req()
        result = svc.ask(req, ruleset)
        assert isinstance(result, NeedsAsk)
        assert result.request.id == req.id

    def test_needs_ask_registers_pending(self):
        svc = PermissionService()
        ruleset = [_rule("bash", "*", "ask")]
        req = _req()
        svc.ask(req, ruleset)
        assert any(r.id == req.id for r in svc.pending_requests())

    def test_no_rules_returns_needs_ask(self):
        """Default evaluation (no rules) → ask."""
        svc = PermissionService()
        req = _req()
        result = svc.ask(req, [])
        assert isinstance(result, NeedsAsk)

    def test_mixed_patterns_ask_triggers_on_any(self):
        """If one pattern evaluates to ask, the whole request is pending."""
        svc = PermissionService()
        # Allow 'ls' but ask for 'git push'
        ruleset = [
            _rule("bash", "ls", "allow"),
        ]
        req = _req(patterns=["ls", "git push"])
        result = svc.ask(req, ruleset)
        assert isinstance(result, NeedsAsk)

    def test_allowed_does_not_register_pending(self):
        svc = PermissionService()
        ruleset = [_rule("bash", "*", "allow")]
        req = _req()
        svc.ask(req, ruleset)
        assert len(svc.pending_requests()) == 0


# ==============================================================================
# ask() — with accumulated approvals
# ==============================================================================


class TestAskWithAccumulatedApprovals:
    """Tests showing that _approved rules layer on top of the agent ruleset."""

    def test_initial_approved_rules_bypass_ask(self):
        approved = [_rule("bash", "git status *", "allow")]
        svc = PermissionService(initial_approved=approved)
        req = _req(patterns=["git status --short"])
        # Agent ruleset says ask for all bash, but approved says allow git status
        ruleset = [_rule("bash", "*", "ask")]
        result = svc.ask(req, ruleset)
        assert isinstance(result, Allowed)

    def test_approved_does_not_override_deny(self):
        """Deny from ruleset wins even if accumulated rule says allow."""
        svc = PermissionService(initial_approved=[_rule("bash", "*", "allow")])
        ruleset = [_rule("bash", "rm -rf *", "deny")]
        req = _req(patterns=["rm -rf /"])
        result = svc.ask(req, ruleset)
        assert isinstance(result, Denied)


# ==============================================================================
# reply("once")
# ==============================================================================


class TestReplyOnce:
    """reply("once") — allow this call only, no rules persisted."""

    def test_reply_once_clears_pending(self):
        svc = PermissionService()
        req = _req()
        svc.ask(req, [])  # → NeedsAsk, now pending

        svc.reply(req.id, "once")

        assert svc.is_resolved(req.id) is True
        # No approved rules added
        assert svc.approved == []

    def test_reply_once_returns_empty_auto_resolved(self):
        svc = PermissionService()
        req = _req()
        svc.ask(req, [])
        auto = svc.reply(req.id, "once")
        assert auto == []

    def test_reply_once_does_not_affect_other_pending(self):
        svc = PermissionService()
        req1 = _req(req_id="r1")
        req2 = _req(req_id="r2")
        svc.ask(req1, [])
        svc.ask(req2, [])
        svc.reply("r1", "once")
        # req2 still pending
        assert svc.is_resolved("r2") is None


# ==============================================================================
# reply("always")
# ==============================================================================


class TestReplyAlways:
    """reply("always") — persist allow rules, auto-resolve siblings."""

    def test_reply_always_adds_approved_rules(self):
        svc = PermissionService()
        req = _req(patterns=["git push"], always=["git push *"])
        svc.ask(req, [])
        svc.reply(req.id, "always")
        approved = svc.approved
        assert len(approved) == 1
        assert approved[0].permission == "bash"
        assert approved[0].pattern == "git push *"
        assert approved[0].action == "allow"

    def test_reply_always_multiple_always_patterns(self):
        req = _req(
            patterns=["git push --force"], always=["git push *", "git *"]
        )
        svc = PermissionService()
        svc.ask(req, [])
        svc.reply(req.id, "always")
        assert len(svc.approved) == 2

    def test_reply_always_auto_resolves_sibling_matching_new_rules(self):
        """Another pending request that now evaluates to allow should be auto-resolved."""
        svc = PermissionService()
        # req1: git push (will be replied "always" with pattern "git push *")
        req1 = _req(req_id="r1", patterns=["git push"], always=["git push *"])
        # req2: git push --dry-run — would now be covered by "git push *"
        req2 = _req(req_id="r2", patterns=["git push --dry-run"])
        svc.ask(req1, [])
        svc.ask(req2, [])
        auto = svc.reply("r1", "always")
        assert "r2" in auto
        assert svc.is_resolved("r2") is True

    def test_reply_always_does_not_auto_resolve_unrelated_sibling(self):
        """A sibling whose patterns don't match the new approved rule stays pending."""
        svc = PermissionService()
        req1 = _req(req_id="r1", patterns=["git push"], always=["git push *"])
        req2 = _req(req_id="r2", patterns=["rm -rf /"])  # not covered by git push *
        svc.ask(req1, [])
        svc.ask(req2, [])
        auto = svc.reply("r1", "always")
        assert "r2" not in auto
        assert svc.is_resolved("r2") is None

    def test_reply_always_returns_resolved_sibling_ids(self):
        svc = PermissionService()
        req1 = _req(req_id="r1", patterns=["ls"], always=["ls *"])
        req2 = _req(req_id="r2", patterns=["ls -la"])
        svc.ask(req1, [])
        svc.ask(req2, [])
        auto = svc.reply("r1", "always")
        assert "r2" in auto

    def test_approved_rules_persist_across_asks(self):
        """Once "always" approved, subsequent asks with matching patterns resolve automatically."""
        svc = PermissionService()
        req1 = _req(req_id="r1", patterns=["ls"], always=["ls *"])
        svc.ask(req1, [])
        svc.reply("r1", "always")

        # New request for same pattern
        req2 = _req(req_id="r2", patterns=["ls -la"])
        result = svc.ask(req2, [])
        assert isinstance(result, Allowed)


# ==============================================================================
# reply("reject")
# ==============================================================================


class TestReplyReject:
    """reply("reject") — deny + fan-out reject all same-session pending."""

    def test_reply_reject_resolves_to_false(self):
        svc = PermissionService()
        req = _req()
        svc.ask(req, [])
        svc.reply(req.id, "reject")
        assert svc.is_resolved(req.id) is False

    def test_reply_reject_fan_out_same_session(self):
        svc = PermissionService()
        req1 = _req(req_id="r1", session_id="sess-A")
        req2 = _req(req_id="r2", session_id="sess-A")
        svc.ask(req1, [])
        svc.ask(req2, [])
        auto = svc.reply("r1", "reject")
        assert "r2" in auto
        assert svc.is_resolved("r2") is False

    def test_reply_reject_returns_auto_rejected_ids(self):
        svc = PermissionService()
        req1 = _req(req_id="r1", session_id="sess-A")
        req2 = _req(req_id="r2", session_id="sess-A")
        req3 = _req(req_id="r3", session_id="sess-A")
        svc.ask(req1, [])
        svc.ask(req2, [])
        svc.ask(req3, [])
        auto = svc.reply("r1", "reject")
        assert set(auto) == {"r2", "r3"}

    def test_reply_reject_does_not_affect_other_session(self):
        """Reject in session A must not touch session B pending requests."""
        svc = PermissionService()
        req_a = _req(req_id="rA", session_id="sess-A")
        req_b = _req(req_id="rB", session_id="sess-B")
        svc.ask(req_a, [])
        svc.ask(req_b, [])
        auto = svc.reply("rA", "reject")
        assert "rB" not in auto
        assert svc.is_resolved("rB") is None

    def test_reply_reject_no_approved_rules_added(self):
        svc = PermissionService()
        req = _req(always=["git push *"])
        svc.ask(req, [])
        svc.reply(req.id, "reject")
        assert svc.approved == []


# ==============================================================================
# Cross-session isolation
# ==============================================================================


class TestCrossSessionIsolation:
    """pending_requests and replies must be isolated across sessions."""

    def test_always_auto_resolve_limited_to_same_session(self):
        svc = PermissionService()
        req_a = _req(req_id="rA", session_id="sess-A", patterns=["ls"], always=["ls *"])
        req_b = _req(req_id="rB", session_id="sess-B", patterns=["ls -la"])
        svc.ask(req_a, [])
        svc.ask(req_b, [])
        auto = svc.reply("rA", "always")
        # rB is in a different session → not auto-resolved by session isolation rule
        assert "rB" not in auto

    def test_pending_requests_includes_all_sessions(self):
        svc = PermissionService()
        req_a = _req(req_id="rA", session_id="sess-A")
        req_b = _req(req_id="rB", session_id="sess-B")
        svc.ask(req_a, [])
        svc.ask(req_b, [])
        ids = {r.id for r in svc.pending_requests()}
        assert ids == {"rA", "rB"}


# ==============================================================================
# Error handling
# ==============================================================================


class TestErrorHandling:
    """Verify KeyError on unknown request_id in reply()."""

    def test_reply_unknown_id_raises(self):
        svc = PermissionService()
        with pytest.raises(KeyError, match="no-such-id"):
            svc.reply("no-such-id", "once")


# ==============================================================================
# Persistence helpers
# ==============================================================================


class TestPersistence:
    """Tests for to_persistence_dict / from_persistence_dict round-trip."""

    def test_empty_service_serialises(self):
        svc = PermissionService()
        d = svc.to_persistence_dict()
        assert d == {"approved": []}

    def test_round_trip_with_approved_rules(self):
        svc = PermissionService(
            initial_approved=[
                _rule("bash", "git push *", "allow"),
                _rule("edit", "*", "allow"),
            ]
        )
        d = svc.to_persistence_dict()
        restored = PermissionService.from_persistence_dict(d)
        assert len(restored.approved) == 2
        assert restored.approved[0].permission == "bash"
        assert restored.approved[1].permission == "edit"

    def test_from_persistence_dict_empty(self):
        svc = PermissionService.from_persistence_dict({})
        assert svc.approved == []

    def test_from_persistence_dict_ignores_unknown_keys(self):
        """Forward-compat: extra keys in the dict must not raise."""
        svc = PermissionService.from_persistence_dict({"approved": [], "future_key": 42})
        assert svc.approved == []

    def test_always_reply_round_trips(self):
        """Rules acquired through "always" reply survive serialisation."""
        svc = PermissionService()
        req = _req(patterns=["git push"], always=["git push *"])
        svc.ask(req, [])
        svc.reply(req.id, "always")

        d = svc.to_persistence_dict()
        restored = PermissionService.from_persistence_dict(d)
        assert len(restored.approved) == 1
        assert restored.approved[0].pattern == "git push *"
