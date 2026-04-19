"""Tests for relay.permission.schema — data type construction and validation."""

from __future__ import annotations

import pytest

from relay.permission.schema import (
    Allowed,
    Denied,
    NeedsAsk,
    PermissionRequest,
    PermissionRule,
)


# ==============================================================================
# PermissionRule
# ==============================================================================


class TestPermissionRule:
    """Tests for PermissionRule construction and field constraints."""

    def test_basic_construction(self):
        rule = PermissionRule(permission="bash", pattern="git push *", action="allow")
        assert rule.permission == "bash"
        assert rule.pattern == "git push *"
        assert rule.action == "allow"

    def test_wildcard_permission(self):
        rule = PermissionRule(permission="*", pattern="*", action="ask")
        assert rule.permission == "*"
        assert rule.pattern == "*"

    def test_deny_action(self):
        rule = PermissionRule(permission="bash", pattern="rm -rf *", action="deny")
        assert rule.action == "deny"

    def test_invalid_action_raises(self):
        with pytest.raises(Exception):
            PermissionRule(permission="bash", pattern="*", action="unknown")  # type: ignore[arg-type]


# ==============================================================================
# PermissionRequest
# ==============================================================================


class TestPermissionRequest:
    """Tests for PermissionRequest construction and defaults."""

    def test_id_auto_generated(self):
        req = PermissionRequest(session_id="s1", permission="bash", patterns=["ls"])
        assert req.id  # non-empty
        assert "-" in req.id  # UUID format

    def test_two_requests_have_different_ids(self):
        a = PermissionRequest(session_id="s1", permission="bash", patterns=["ls"])
        b = PermissionRequest(session_id="s1", permission="bash", patterns=["ls"])
        assert a.id != b.id

    def test_custom_id(self):
        req = PermissionRequest(
            id="fixed-id", session_id="s1", permission="bash", patterns=["ls"]
        )
        assert req.id == "fixed-id"

    def test_metadata_defaults_to_empty_dict(self):
        req = PermissionRequest(session_id="s1", permission="bash", patterns=["ls"])
        assert req.metadata == {}

    def test_always_defaults_to_empty_list(self):
        req = PermissionRequest(session_id="s1", permission="bash", patterns=["ls"])
        assert req.always == []

    def test_tool_call_id_optional(self):
        req = PermissionRequest(
            session_id="s1", permission="bash", patterns=["ls"], tool_call_id="tc1"
        )
        assert req.tool_call_id == "tc1"

        req_no_tc = PermissionRequest(
            session_id="s1", permission="bash", patterns=["ls"]
        )
        assert req_no_tc.tool_call_id is None

    def test_multiple_patterns(self):
        req = PermissionRequest(
            session_id="s1",
            permission="edit",
            patterns=["src/main.py", "tests/test_main.py"],
        )
        assert len(req.patterns) == 2

    def test_metadata_preserved(self):
        req = PermissionRequest(
            session_id="s1",
            permission="bash",
            patterns=["git push --force"],
            metadata={"command": "git push --force"},
        )
        assert req.metadata["command"] == "git push --force"


# ==============================================================================
# Decision types
# ==============================================================================


class TestDecisionTypes:
    """Tests for Allowed, Denied, NeedsAsk discriminated union types."""

    def test_allowed_kind(self):
        d = Allowed()
        assert d.kind == "allowed"

    def test_denied_default_reason(self):
        d = Denied()
        assert d.kind == "denied"
        assert "denied" in d.reason.lower()

    def test_denied_custom_reason(self):
        d = Denied(reason="No bash allowed.")
        assert d.reason == "No bash allowed."

    def test_needs_ask_carries_request(self):
        req = PermissionRequest(
            session_id="s1", permission="bash", patterns=["rm -rf /"]
        )
        d = NeedsAsk(request=req)
        assert d.kind == "needs_ask"
        assert d.request is req
        assert d.request.permission == "bash"
