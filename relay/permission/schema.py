"""Core permission data types.

These types form the vocabulary shared by the evaluator, service, and
middleware.  The design mirrors Opencode's permission model:

- ``PermissionRule`` — a single evaluated constraint.
- ``Ruleset`` — an ordered list of rules; later rules win on conflict.
- ``PermissionRequest`` — a single tool-call permission check.
- ``Reply`` — the user's interactive answer to a permission prompt.
- ``PermissionDecision`` — the result returned by ``PermissionService.ask``.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

# ==============================================================================
# Primitive types
# ==============================================================================

PermissionAction = Literal["allow", "deny", "ask"]
"""The result of evaluating a permission rule: allow, deny, or ask the user."""

Reply = Literal["once", "always", "reject"]
"""The user's interactive response to a permission prompt.

- ``"once"``   — allow this specific call only; do not persist.
- ``"always"`` — allow this pattern permanently for the session and persist.
- ``"reject"`` — deny this call and fan-out reject all same-session pending
                  requests.
"""

# A compiled ruleset is just an ordered list of rules.
# Evaluation is last-match-wins, so appending to this list is how
# higher-priority rules override earlier ones.
Ruleset = list["PermissionRule"]


# ==============================================================================
# PermissionRule
# ==============================================================================


class PermissionRule(BaseModel):
    """A single permission rule matching a permission key and pattern.

    Both ``permission`` and ``pattern`` support wildcard syntax:
    ``*`` matches any sequence of characters, ``?`` matches a single
    character.  The special trailing ``' *'`` (space + star) is treated
    as optional — ``"git push *"`` matches both ``"git push"`` and
    ``"git push --force origin main"``.
    """

    permission: str
    """Permission key, e.g. ``"bash"``, ``"edit"``, ``"read"``, ``"*"``."""

    pattern: str
    """Concrete value pattern, e.g. ``"git push *"``, ``"src/**"``, ``"*"``."""

    action: PermissionAction
    """What to do when this rule matches: allow, deny, or ask."""


# ==============================================================================
# PermissionRequest
# ==============================================================================


class PermissionRequest(BaseModel):
    """A request for permission to execute a specific tool action.

    The middleware constructs one of these per tool call and passes it to
    ``PermissionService.ask``.  If evaluation yields ``NeedsAsk``, the
    request is persisted inside the service until the user replies.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique request identifier.  Used to correlate prompts with replies."""

    session_id: str
    """LangGraph ``thread_id`` / session scope for fan-out operations."""

    permission: str
    """Permission key, e.g. ``"bash"``, ``"edit"``, ``"read"``."""

    patterns: list[str]
    """Concrete values being evaluated, e.g. ``["git push --force"]``."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Tool-specific context shown to the user in the prompt.

    Examples: ``{"command": "git push --force"}`` for bash,
    ``{"filepath": "src/main.py", "diff": "..."}`` for edit.
    """

    always: list[str] = Field(default_factory=list)
    """Patterns to convert into ``allow`` rules when the user replies
    ``"always"``.

    Typically a higher-level prefix pattern, e.g. ``["git push *"]``.
    """

    tool_call_id: str | None = None
    """LangGraph tool-call correlation ID, if available."""


# ==============================================================================
# PermissionDecision
# ==============================================================================


class Allowed(BaseModel):
    """Decision: the request is permitted; proceed with tool execution."""

    kind: Literal["allowed"] = "allowed"


class Denied(BaseModel):
    """Decision: the request is denied; return an error tool message."""

    kind: Literal["denied"] = "denied"
    reason: str = "Permission denied."
    """Human-readable explanation surfaced as the tool's error message."""


class NeedsAsk(BaseModel):
    """Decision: evaluation is inconclusive; prompt the user."""

    kind: Literal["needs_ask"] = "needs_ask"
    request: PermissionRequest
    """The pending request that requires a user reply."""


PermissionDecision = Allowed | Denied | NeedsAsk
"""Union of the three possible outcomes from ``PermissionService.ask``."""
