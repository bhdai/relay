"""Resume handler — thread selection and pending-interrupt workflow.

``ResumeHandler`` owns the ``/resume`` command workflow:

1. Load persisted threads from the checkpointer.
2. Let the user pick a thread (via ``ThreadManager``).
3. Switch the session context to the chosen thread.
4. If the thread has pending interrupts, prompt and resume.

Langrepl equivalent:
    ``langrepl.cli.handlers.resume.ResumeHandler``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from relay.cli.handlers.interrupts import InterruptHandler
from relay.cli.handlers.threads import ThreadManager
from relay.cli.ui.renderer import render_info

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession

    from relay.cli.core.session import Session


class ResumeHandler:
    """Handles the /resume command workflow."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.interrupt_handler = InterruptHandler()

    async def handle(self, prompt_session: PromptSession) -> None:
        """Show thread list, switch context, and handle pending interrupts."""
        checkpointer = getattr(self.session.graph, "checkpointer", None)
        if checkpointer:
            await self.session.threads.load_persisted_threads(checkpointer)

        selected = await self.session.threads.select_thread(prompt_session)
        if not selected:
            return

        self.session.context.thread_id = selected
        render_info(f"Resumed thread {selected[:8]}.")

        # Check for pending interrupts in the checkpoint.
        if not checkpointer:
            return

        interrupts = await ThreadManager.get_pending_interrupts(
            checkpointer, self.session.context.thread_id
        )
        if interrupts:
            render_info("This thread has a pending interrupt.")
            await self.session.message_dispatcher.resume_from_interrupt(interrupts)
