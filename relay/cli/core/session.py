"""REPL session — main loop, lifecycle, and graph ownership.

This is the top-level orchestrator.  It wires together the
checkpointer, graph, command dispatcher, thread manager, and
message dispatcher.
"""

from __future__ import annotations

import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from relay.cli.bootstrap import Initializer
from relay.cli.core.context import Context
from relay.cli.dispatchers.commands import dispatch_command
from relay.cli.dispatchers.messages import MessageDispatcher
from relay.cli.handlers.threads import ThreadManager
from relay.cli.ui.renderer import render_info


class Session:
    """Manages a single REPL session with streaming and thread management.

    The session owns the checkpointer lifecycle and the compiled graph.
    Streaming and interrupt/resume are delegated to ``MessageDispatcher``.
    """

    def __init__(self, context: Context | None = None) -> None:
        self.context = context or Context()

        # Set by start() inside the initializer context manager.
        self.graph = None
        self._initializer = Initializer()

        self.threads = ThreadManager()
        self.message_dispatcher = MessageDispatcher(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the REPL until the user exits."""
        async with self._initializer.get_graph(backend=self.context.backend) as graph:
            self.graph = graph
            await self._main_loop()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_input(prompt_session: PromptSession) -> str | None:
        """Read a line of input, returning None on EOF/interrupt."""
        try:
            return await prompt_session.prompt_async("❯ ")
        except (EOFError, KeyboardInterrupt):
            return None

    # ------------------------------------------------------------------
    # Resume handling
    # ------------------------------------------------------------------

    async def _handle_resume(self, prompt_session: PromptSession) -> None:
        """Show thread list, switch to the selected thread, handle pending interrupts."""
        # Load persisted threads from the checkpointer so that threads
        # from previous sessions are discoverable.
        checkpointer = getattr(self.graph, "checkpointer", None)
        if checkpointer:
            await self.threads.load_persisted_threads(checkpointer)

        selected = await self.threads.select_thread(prompt_session)
        if not selected:
            return

        self.context.thread_id = selected
        render_info(f"Resumed thread {selected[:8]}.")

        # Check for pending interrupts in the checkpoint.
        if not checkpointer:
            return

        interrupts = await ThreadManager.get_pending_interrupts(
            checkpointer, self.context.thread_id
        )
        if interrupts:
            render_info("This thread has a pending interrupt.")
            await self.message_dispatcher.resume_from_interrupt(interrupts)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        prompt_session: PromptSession = PromptSession(history=InMemoryHistory())

        render_info("relay agent ready. Type /help for commands.")

        while self.context.running:
            text = await self._get_input(prompt_session)

            if text is None:
                break

            text = text.strip()
            if not text:
                continue

            if text.lower() in ("exit", "quit", "stop"):
                break

            # Slash commands.
            if text.startswith("/"):
                cmd = text.strip().lower()
                if cmd == "/resume":
                    await self._handle_resume(prompt_session)
                    continue
                should_exit = dispatch_command(text, self)
                if should_exit:
                    break
                continue

            # Record thread and stream response.
            self.threads.record(self.context.thread_id, preview=text)

            try:
                await self.message_dispatcher.dispatch(text)
            except asyncio.CancelledError:
                pass
