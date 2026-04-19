"""CLI-specific session context.

``Context`` holds the mutable per-session state that used to live
directly on ``Session``.  Pulling it into its own dataclass makes the
boundary between *lifecycle* (Session) and *state* (Context) explicit,
mirroring the langrepl ``cli.core.context.Context`` pattern.

The class is intentionally minimal for now — relay's config subsystem
is not yet rich enough for a full ``Context.create()`` factory method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from relay.configs.approval import ApprovalMode


@dataclass
class Context:
    """Runtime state scoped to a single CLI session."""

    # Project root used for config loading, tool execution, and memory.
    working_dir: str = field(default_factory=lambda: str(Path.cwd()))

    # Selected top-level agent config and optional model override.
    agent: str | None = None
    model: str | None = None

    # Checkpointer backend name (e.g. "sqlite", "memory").
    backend: str = "sqlite"

    # Active conversation thread.
    thread_id: str = field(default_factory=lambda: str(uuid4()))

    # Human-in-the-loop approval policy for tool calls.
    approval_mode: ApprovalMode = ApprovalMode.SEMI_ACTIVE

    # Cumulative token / cost counters across all turns.
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    # Model pricing used by TokenCostMiddleware to compute per-call cost.
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0

    # Whether the REPL loop should keep running.
    running: bool = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def new_thread(self) -> str:
        """Switch to a fresh thread and return the new thread_id."""
        self.thread_id = str(uuid4())
        return self.thread_id

    def cycle_approval_mode(self) -> ApprovalMode:
        """Cycle approval mode in declaration order."""
        modes = list(ApprovalMode)
        idx = modes.index(self.approval_mode)
        self.approval_mode = modes[(idx + 1) % len(modes)]
        return self.approval_mode

    def permission_mode_overlay(self) -> list[dict[str, Any]]:
        """Translate ``approval_mode`` into a serialisable permission ruleset overlay.

        The returned list is seeded into ``AgentContext.permission_ruleset``
        at session start so that mode-based auto-approval takes effect via the
        ``PermissionMiddleware``.  It is placed before any session-accumulated
        "always" rules so that user approvals are always additive on top of the
        mode baseline.

        - ``SEMI_ACTIVE`` → ``[]``: normal evaluation; every tool call that is
          not explicitly allowed by the agent config is prompted.
        - ``ACTIVE``      → ``[{"permission": "*", "pattern": "*", "action": "allow"}]``:
          auto-approve everything unless a deny rule fires.
        - ``AGGRESSIVE``  → same overlay as ``ACTIVE``; deny rules from the
          agent config are still enforced by the ``PermissionService`` regardless
          of this overlay (deny is always checked first in the service).
        """
        if self.approval_mode in (ApprovalMode.ACTIVE, ApprovalMode.AGGRESSIVE):
            return [{"permission": "*", "pattern": "*", "action": "allow"}]
        return []

    def accumulate(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Add per-turn token/cost data to the running totals."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
