"""REPL session — main loop, lifecycle, and graph ownership.

This is the top-level orchestrator.  It wires together the
checkpointer, graph, command dispatcher, thread manager, and
message dispatcher.
"""

from __future__ import annotations

import asyncio

from relay.cli.bootstrap import Initializer
from relay.cli.core.context import Context
from relay.cli.dispatchers.commands import CommandDispatcher
from relay.cli.dispatchers.messages import MessageDispatcher
from relay.cli.handlers.threads import ThreadManager
from relay.cli.ui.prompt import InteractivePrompt
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

        self.prompt = InteractivePrompt(self.context)
        self.threads = ThreadManager()
        self.message_dispatcher = MessageDispatcher(self)
        self.command_dispatcher = CommandDispatcher(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the REPL until the user exits."""
        async with self._initializer.get_graph(backend=self.context.backend) as graph:
            self.graph = graph
            await self._main_loop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        render_info("relay agent ready. Type /help for commands.")

        while self.context.running:
            text = await self.prompt.get_input()

            if text is None:
                break

            text = text.strip()
            if not text:
                continue

            if text.lower() in ("exit", "quit", "stop"):
                break

            # Slash commands — all routed through the async dispatcher.
            if text.startswith("/"):
                should_exit = await self.command_dispatcher.dispatch(
                    text, prompt_session=self.prompt.session
                )
                if should_exit:
                    break
                continue

            # Record thread and stream response.
            self.threads.record(self.context.thread_id, preview=text)

            try:
                await self.message_dispatcher.dispatch(text)
            except asyncio.CancelledError:
                pass
