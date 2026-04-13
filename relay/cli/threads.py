"""Thread tracking and resume handling.

Keeps an in-memory registry of thread IDs seen in this session and
provides the ``/resume`` workflow: list threads, select one, detect
pending interrupts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Interrupt
from prompt_toolkit import PromptSession

from relay.cli.renderer import console, render_error, render_info

if TYPE_CHECKING:
    pass


class ThreadManager:
    """Tracks threads seen in this session and handles /resume."""

    def __init__(self) -> None:
        # Maps thread_id → first human message preview.
        self._threads: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Recording
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
        self, prompt_session: PromptSession
    ) -> str | None:
        """Show thread list and return the selected thread_id, or None."""
        if not self._threads:
            render_info("No previous threads to resume.")
            return None

        console.print()
        for i, (tid, preview) in enumerate(self._threads.items(), 1):
            short_id = tid[:8]
            display = preview[:60] + "..." if len(preview) > 60 else preview
            console.print(f"  {i}. [{short_id}] {display}", style="dim")
        console.print()

        try:
            choice = await prompt_session.prompt_async("Thread #: ")
        except (EOFError, KeyboardInterrupt):
            return None

        try:
            idx = int(choice.strip()) - 1
            thread_ids = list(self._threads.keys())
            if 0 <= idx < len(thread_ids):
                return thread_ids[idx]
            render_error("Invalid selection.")
            return None
        except ValueError:
            render_error("Enter a number.")
            return None

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
