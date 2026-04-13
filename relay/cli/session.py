"""REPL session — main loop, lifecycle, and graph ownership.

This is the top-level orchestrator.  It wires together the
checkpointer, graph, command dispatcher, thread manager, and
streaming loop.
"""

from __future__ import annotations

import asyncio
import signal
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from relay.checkpointer import create_checkpointer
from relay.cli.commands import dispatch_command
from relay.cli.renderer import render_cost_summary, render_info
from relay.cli.streaming import prompt_for_interrupt, stream_response
from relay.cli.threads import ThreadManager
from relay.graph import build_graph


class Session:
    """Manages a single REPL session with streaming and thread management.

    The session owns the checkpointer lifecycle and the compiled graph.
    Interrupt/resume is handled automatically: when a graph node calls
    ``interrupt()``, the streaming loop prompts the user and loops back
    with ``Command(resume=value)``.
    """

    def __init__(self, *, backend: str = "sqlite") -> None:
        self.backend = backend
        self.thread_id = str(uuid4())
        self.total_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.running = True

        # Set by start() inside the checkpointer context manager.
        self.graph = None
        self.checkpointer = None

        self.threads = ThreadManager()

        # Track the active streaming task so Ctrl+C can cancel it.
        self._stream_task: asyncio.Task | None = None
        self._original_sigint = signal.getsignal(signal.SIGINT)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the REPL until the user exits."""
        async with create_checkpointer(backend=self.backend) as checkpointer:
            self.checkpointer = checkpointer
            self.graph = build_graph(checkpointer=checkpointer)
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
    # SIGINT handling
    # ------------------------------------------------------------------

    def _sigint_handler(self, signum: int, frame: object) -> None:
        """First Ctrl+C cancels the stream; second exits."""
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
        else:
            signal.signal(signal.SIGINT, self._original_sigint)

    # ------------------------------------------------------------------
    # Streaming wrapper
    # ------------------------------------------------------------------

    async def _run_stream(self, input_value) -> None:
        """Stream a response and accumulate cost."""
        signal.signal(signal.SIGINT, self._sigint_handler)
        try:
            stats = await stream_response(
                self.graph,
                input_value,
                thread_id=self.thread_id,
            )
        except asyncio.CancelledError:
            render_info("\n  (cancelled)")
            return
        finally:
            signal.signal(signal.SIGINT, self._original_sigint)

        # Newline after the raw-streamed text.
        if stats.collected_text:
            print()

        self.total_input_tokens += stats.input_tokens
        self.total_output_tokens += stats.output_tokens
        self.total_cost += stats.cost

        if stats.input_tokens or stats.output_tokens:
            render_cost_summary(stats.input_tokens, stats.output_tokens, stats.cost)

    # ------------------------------------------------------------------
    # Resume handling
    # ------------------------------------------------------------------

    async def _handle_resume(self, prompt_session: PromptSession) -> None:
        """Show thread list, switch to the selected thread, handle pending interrupts."""
        selected = await self.threads.select_thread(prompt_session)
        if not selected:
            return

        self.thread_id = selected
        render_info(f"Resumed thread {selected[:8]}.")

        # Check for pending interrupts in the checkpoint.
        if not self.checkpointer:
            return

        interrupts = await ThreadManager.get_pending_interrupts(
            self.checkpointer, self.thread_id
        )
        if interrupts:
            render_info("This thread has a pending interrupt.")
            resume_value = await prompt_for_interrupt(interrupts)
            if resume_value is not None:
                await self._run_stream(Command(resume=resume_value))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        prompt_session: PromptSession = PromptSession(history=InMemoryHistory())

        render_info("relay agent ready. Type /help for commands.")

        while self.running:
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
            self.threads.record(self.thread_id, preview=text)

            self._stream_task = asyncio.ensure_future(
                self._run_stream({"messages": [HumanMessage(content=text)]})
            )
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            finally:
                self._stream_task = None
