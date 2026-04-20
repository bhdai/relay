"""Permission service — ask/reply lifecycle and accumulated approvals.

``PermissionService`` is instantiated once per LangGraph session and
carries two pieces of mutable state:

1. ``_approved`` — rules accumulated through "always" replies during
   the current session.  These are persisted to
   ``.relay/permission.json`` so they survive restarts.
2. ``_pending`` — unresolved ``PermissionRequest`` objects waiting for
   a user reply via ``reply()``.

The lifecycle maps to LangGraph's interrupt/resume pattern:
- ``ask()``  → evaluation yields ``NeedsAsk``  → middleware calls
  ``interrupt(payload)`` to suspend the graph.
- ``reply()`` → middleware receives ``Command(resume=...)`` from the
  user → calls ``service.reply(request_id, reply)`` → graph resumes.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from relay.permission.evaluate import evaluate
from relay.permission.schema import (
    Allowed,
    Denied,
    NeedsAsk,
    PermissionDecision,
    PermissionRequest,
    PermissionRule,
    Reply,
    Ruleset,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Internal pending-entry bookkeeping
# ==============================================================================


@dataclasses.dataclass
class _PendingEntry:
    """Internal record for a request that is awaiting a user reply."""

    request: PermissionRequest
    # Resolved to True (once/always) or False (reject) once reply() is called.
    resolved: bool | None = None


# ==============================================================================
# PermissionService
# ==============================================================================


class PermissionService:
    """Stateful permission evaluator and reply manager for a single session.

    Thread-safety note: LangGraph's async tool node runs tool calls
    concurrently within a single turn.  ``ask`` and ``reply`` should
    only be called from a single coroutine at a time (the middleware
    serialises access via LangGraph's interrupt mechanism).
    """

    def __init__(self, initial_approved: Ruleset | None = None) -> None:
        # Session-accumulated "always" rules.  Start from any pre-loaded
        # persisted rules and grow as the user approves patterns.
        self._approved: Ruleset = list(initial_approved or [])

        # Pending requests indexed by request_id.
        self._pending: dict[str, _PendingEntry] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def approved(self) -> Ruleset:
        """Currently accumulated always-approved rules (read-only view)."""
        return list(self._approved)

    def pending_requests(self) -> list[PermissionRequest]:
        """Return all unresolved pending permission requests."""
        return [
            entry.request
            for entry in self._pending.values()
            if entry.resolved is None
        ]

    def ask(
        self,
        request: PermissionRequest,
        ruleset: Ruleset,
    ) -> PermissionDecision:
        """Evaluate *request* against *ruleset* plus accumulated approvals.

        Evaluation proceeds pattern-by-pattern.  A single ``deny`` rule
        short-circuits the entire request immediately.  If all patterns
        resolve to ``allow``, the decision is ``Allowed``.  If any
        pattern has no ``allow`` rule (evaluates to ``ask``), the
        request is registered as pending and ``NeedsAsk`` is returned.

        Args:
            request: The permission request describing the tool call.
            ruleset: The agent-level ruleset (from config).  The service's
                accumulated ``_approved`` rules are layered on top.

        Returns:
            ``Allowed``, ``Denied``, or ``NeedsAsk``.
        """
        needs_ask = False

        for pattern in request.patterns:
            # Phase 1 — check the config ruleset for a hard deny.
            # ``deny`` rules in the agent config are explicit blocks that
            # user-accumulated approvals must NOT be able to override.
            # We evaluate the config ruleset FIRST and short-circuit on deny
            # before consulting accumulated ``_approved`` rules.
            config_rule = evaluate(request.permission, pattern, ruleset)
            if config_rule.action == "deny":
                logger.debug(
                    "Permission denied: permission=%r pattern=%r matched rule=%r",
                    request.permission,
                    pattern,
                    config_rule,
                )
                return Denied(reason=f"Permission denied for {request.permission!r}: {pattern!r}.")

            # Phase 2 — evaluate the merged ruleset (config + accumulated
            # session approvals) to determine allow vs. ask.  The approved
            # rules layer on top of config via last-match-wins, so an "always"
            # reply can turn an "ask" config rule into "allow".  It cannot,
            # however, turn a "deny" rule into "allow" (handled above).
            rule = evaluate(
                request.permission,
                pattern,
                ruleset,
                self._approved,
            )

            if rule.action == "ask":
                needs_ask = True
                # Do not break early — a later pattern might trigger deny.

        if not needs_ask:
            logger.debug(
                "Permission allowed: permission=%r patterns=%r",
                request.permission,
                request.patterns,
            )
            return Allowed()

        # At least one pattern needs user input.
        logger.debug(
            "Permission needs user input: permission=%r patterns=%r",
            request.permission,
            request.patterns,
        )
        self._pending[request.id] = _PendingEntry(request=request)
        return NeedsAsk(request=request)

    def reply(
        self,
        request_id: str,
        reply: Reply,
        message: str | None = None,
    ) -> list[str]:
        """Process a user reply for a pending request.

        Args:
            request_id: The ``PermissionRequest.id`` of the pending request.
            reply: The user's choice: ``"once"``, ``"always"``, or
                ``"reject"``.
            message: Optional free-text message from the user (reserved for
                future use — not currently persisted).

        Returns:
            Request IDs that were auto-resolved as a side-effect of this
            reply (e.g. other pending requests from the same session that
            become allowed after an ``"always"`` reply, or fan-out
            rejections from a ``"reject"`` reply).

        Raises:
            KeyError: If *request_id* is not in the pending list.
        """
        if request_id not in self._pending:
            raise KeyError(f"No pending permission request with id={request_id!r}.")

        entry = self._pending[request_id]
        request = entry.request
        auto_resolved: list[str] = []

        if reply == "reject":
            # Deny this request and fan-out reject all other pending
            # requests from the same session.  This mirrors Opencode's
            # fan-out behaviour: a single "reject" stops all queued work.
            entry.resolved = False
            for rid, other in list(self._pending.items()):
                if rid != request_id and other.request.session_id == request.session_id:
                    if other.resolved is None:
                        other.resolved = False
                        auto_resolved.append(rid)
                        logger.debug(
                            "Auto-rejected pending request %r (fan-out from %r)",
                            rid,
                            request_id,
                        )

        elif reply == "once":
            # Allow this specific call only; no rules are persisted.
            entry.resolved = True

        elif reply == "always":
            # Persist allow rules for each pattern in ``request.always``
            # so future calls with matching patterns are auto-approved.
            for always_pattern in request.always:
                new_rule = PermissionRule(
                    permission=request.permission,
                    pattern=always_pattern,
                    action="allow",
                )
                self._approved.append(new_rule)
                logger.debug(
                    "Persisted always-allow rule: %r",
                    new_rule,
                )
            entry.resolved = True

            # Auto-resolve any other pending requests from the same session
            # that would now evaluate to ``allow`` given the updated rules.
            for rid, other in list(self._pending.items()):
                if rid == request_id or other.request.session_id != request.session_id:
                    continue
                if other.resolved is not None:
                    continue
                if self._is_now_allowed(other.request):
                    other.resolved = True
                    auto_resolved.append(rid)
                    logger.debug(
                        "Auto-resolved pending request %r after always-approval of %r",
                        rid,
                        request_id,
                    )

        return auto_resolved

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_now_allowed(self, request: PermissionRequest) -> bool:
        """Check whether *request* would now evaluate to ``allow`` using only ``self._approved``.

        Excludes the configuration ruleset.  Used after an ``"always"`` reply
        to auto-resolve sibling pending requests without a full re-evaluation
        against agent config.

        Args:
            request: The pending permission request to re-evaluate.

        Returns:
            ``True`` if every pattern in *request* now evaluates to ``allow``
            under the accumulated approved rules alone.
        """
        for pattern in request.patterns:
            # We check only against the accumulated approved rules here.
            # If any pattern still evaluates to ``ask`` (no approved rule
            # covers it), the request stays pending.
            rule = evaluate(request.permission, pattern, self._approved)
            if rule.action != "allow":
                return False
        return True

    def get_pending(self, request_id: str) -> PermissionRequest | None:
        """Return the pending request with the given ID, or ``None``.

        Args:
            request_id: The ``PermissionRequest.id`` to look up.

        Returns:
            The matching ``PermissionRequest``, or ``None`` if not found.
        """
        entry = self._pending.get(request_id)
        if entry is not None:
            return entry.request
        return None

    def is_resolved(self, request_id: str) -> bool | None:
        """Return the resolved state of a pending request.

        Args:
            request_id: The ``PermissionRequest.id`` to look up.

        Returns:
            ``True`` if the request was approved (once or always),
            ``False`` if it was rejected, or ``None`` if still pending
            or unknown.
        """
        entry = self._pending.get(request_id)
        if entry is None:
            return None
        return entry.resolved

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def to_persistence_dict(self) -> dict[str, Any]:
        """Serialise accumulated approvals to a JSON-compatible dict.

                Example:
                        {
                            "approved": [
                                {"permission": "bash", "pattern": "git push *", "action": "allow"},
                                ...
                            ]
                        }

        Returns:
            A ``dict`` suitable for ``json.dump``.
        """
        return {
            "approved": [rule.model_dump() for rule in self._approved],
        }

    @classmethod
    def from_persistence_dict(cls, data: dict[str, Any]) -> "PermissionService":
        """Restore a ``PermissionService`` from a persisted dict.

        Unrecognised keys are silently ignored to allow forward compatibility.

        Args:
            data: A dict previously produced by ``to_persistence_dict``.

        Returns:
            A new ``PermissionService`` pre-loaded with the persisted
            approved rules.
        """
        approved = [
            PermissionRule.model_validate(r)
            for r in data.get("approved", [])
        ]
        return cls(initial_approved=approved)
