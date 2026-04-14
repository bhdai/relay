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
from uuid import uuid4


@dataclass
class Context:
    """Runtime state scoped to a single CLI session."""

    # Checkpointer backend name (e.g. "sqlite", "memory").
    backend: str = "sqlite"

    # Active conversation thread.
    thread_id: str = field(default_factory=lambda: str(uuid4()))

    # Cumulative token / cost counters across all turns.
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    # Whether the REPL loop should keep running.
    running: bool = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def new_thread(self) -> str:
        """Switch to a fresh thread and return the new thread_id."""
        self.thread_id = str(uuid4())
        return self.thread_id

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
