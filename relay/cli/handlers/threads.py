"""Thread tracking and resume handling.

Keeps an in-memory registry of thread IDs seen in this session as a
supplement to checkpointer-backed discovery.  The ``/resume`` workflow
is primarily driven by :meth:`BaseCheckpointer.get_thread_summaries`
so that conversations survive across CLI restarts.

When a :class:`relay.checkpointer.base.BaseCheckpointer` is available,
thread listing queries persisted thread summaries (with last-message
previews and timestamps) and merges in any session-local threads that
have not yet been checkpointed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Interrupt
from prompt_toolkit import PromptSession

from relay.cli.theme import console
from relay.cli.ui.renderer import render_error, render_info

if TYPE_CHECKING:
    from relay.checkpointer.base import BaseCheckpointer as RelayCheckpointer
    from relay.checkpointer.base import ThreadSummary


class ThreadManager:
    """Tracks threads and drives the ``/resume`` thread picker.

    Thread data comes from two sources, in priority order:

    1. **Checkpointer** — ``get_thread_summaries()`` returns persisted
       threads with real message previews and timestamps.
    2. **Session-local dict** — ``record()`` captures threads created
       in this session that may not have been checkpointed yet.
    """

    def __init__(self) -> None:
        # Maps thread_id → first human message preview.  Supplementary
        # to checkpointer data — used for threads not yet persisted.
        self._threads: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Recording (session-local)
    # ------------------------------------------------------------------

    def record(self, thread_id: str, preview: str | None = None) -> None:
        """Record a thread with an optional message preview."""
        if preview:
            self._threads[thread_id] = preview
        elif thread_id not in self._threads:
            self._threads[thread_id] = "(no messages)"

    # ------------------------------------------------------------------
    # Resume UI
    # ------------------------------------------------------------------

    async def select_thread(
        self,
        prompt_session: PromptSession,
        checkpointer: BaseCheckpointSaver | None = None,
        current_thread_id: str | None = None,
    ) -> str | None:
        """Show thread list and return the selected thread_id, or ``None``.

        When a checkpointer implementing ``get_thread_summaries()`` is
        available, the list is populated from persisted data (with real
        previews and timestamps).  Session-local threads are merged in
        so that threads created but not yet checkpointed still appear.
        """
        entries = await self._build_thread_list(checkpointer, current_thread_id)

        if not entries:
            render_info("No previous threads to resume.")
            return None

        console.print()
        for i, entry in enumerate(entries, 1):
            short_id = entry.thread_id[:8]
            display = (
                entry.last_message[:60] + "..."
                if len(entry.last_message) > 60
                else entry.last_message
            )
            timestamp_suffix = f"  ({entry.timestamp[:19]})" if entry.timestamp else ""
            console.print(
                f"  {i}. [{short_id}] {display}{timestamp_suffix}",
                style="muted",
            )
        console.print()

        try:
            choice = await prompt_session.prompt_async("Thread #: ")
        except (EOFError, KeyboardInterrupt):
            return None

        try:
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(entries):
                return entries[idx].thread_id
            render_error("Invalid selection.")
            return None
        except ValueError:
            render_error("Enter a number.")
            return None

    # ------------------------------------------------------------------
    # Internal — assemble thread list
    # ------------------------------------------------------------------

    async def _build_thread_list(
        self,
        checkpointer: BaseCheckpointSaver | None,
        current_thread_id: str | None,
    ) -> list[ThreadSummary]:
        """Merge checkpointer summaries with session-local threads.

        Returns a list of ``ThreadSummary`` sorted newest-first, with
        the current thread filtered out.
        """
        from relay.checkpointer.base import (
            BaseCheckpointer as _Base,
        )
        from relay.checkpointer.base import ThreadSummary as _TS

        persisted: list[_TS] = []

        if checkpointer is not None and isinstance(checkpointer, _Base):
            try:
                persisted = await checkpointer.get_thread_summaries()
            except NotImplementedError:
                pass

        # Set of thread IDs already covered by the checkpointer.
        persisted_ids = {s.thread_id for s in persisted}

        # Merge session-local threads that the checkpointer doesn't know about.
        for tid, preview in self._threads.items():
            if tid not in persisted_ids:
                persisted.append(
                    _TS(thread_id=tid, last_message=preview)
                )

        # Filter out the current thread.
        if current_thread_id:
            persisted = [s for s in persisted if s.thread_id != current_thread_id]

        return persisted

    # ------------------------------------------------------------------
    # Pending interrupt detection
    # ------------------------------------------------------------------

    @staticmethod
    async def get_pending_interrupts(
        checkpointer: BaseCheckpointSaver,
        thread_id: str,
    ) -> list[Interrupt]:
        """Return pending interrupts for a thread, or an empty list."""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            checkpoint_tuple = await checkpointer.aget_tuple(config)
        except Exception:
            return []

        if not checkpoint_tuple or not checkpoint_tuple.pending_writes:
            return []

        interrupts: list[Interrupt] = []
        for _task_id, channel, value in checkpoint_tuple.pending_writes:
            if channel == "__interrupt__":
                if isinstance(value, list):
                    interrupts.extend(value)
                else:
                    interrupts.append(value)
        return interrupts
