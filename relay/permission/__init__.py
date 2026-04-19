"""Permission system for Relay.

Provides a unified, rule-based execution control model where every
side-effecting tool action is gated by a permission ruleset.  Rules are
evaluated with last-match-wins semantics (matching Opencode's design),
and users can approve, deny, or defer decisions interactively.

Public surface
--------------
schema    — Core data types: ``PermissionRule``, ``PermissionRequest``, …
evaluate  — Pure ``evaluate()`` function + ``wildcard_match`` helper.
service   — ``PermissionService`` that manages ask/reply lifecycle.
config    — YAML config normalization (``from_config``, ``merge``) and defaults.
"""

from relay.permission.config import (
    DEFAULT_PERMISSION,
    READONLY_PERMISSION,
    from_config,
    merge,
    migrate_from_approval_config,
)
from relay.permission.evaluate import evaluate, wildcard_match
from relay.permission.schema import (
    PermissionAction,
    PermissionDecision,
    PermissionRequest,
    PermissionRule,
    Reply,
    Ruleset,
)
from relay.permission.service import (
    Allowed,
    Denied,
    NeedsAsk,
    PermissionService,
)

__all__ = [
    # schema
    "PermissionAction",
    "PermissionDecision",
    "PermissionRequest",
    "PermissionRule",
    "Reply",
    "Ruleset",
    # evaluate
    "evaluate",
    "wildcard_match",
    # service
    "Allowed",
    "Denied",
    "NeedsAsk",
    "PermissionService",
    # config
    "DEFAULT_PERMISSION",
    "READONLY_PERMISSION",
    "from_config",
    "merge",
    "migrate_from_approval_config",
]
