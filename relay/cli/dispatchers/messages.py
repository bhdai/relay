"""Message dispatching — owns a user turn from input to cost accounting.

``MessageDispatcher`` sits between the REPL session and the low-level
``stream_response`` helper.  It builds the LangChain message, manages
the SIGINT handler around streaming, and feeds token/cost data back
into the session ``Context``.

Langrepl equivalent:
    ``langrepl.cli.dispatchers.messages.MessageDispatcher``
"""

from __future__ import annotations

import asyncio
import re
import signal
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from relay.cli.core.streaming import stream_response
from relay.cli.handlers.interrupts import InterruptHandler
from relay.cli.ui.renderer import render_cost_summary, render_error, render_info

if TYPE_CHECKING:
    from relay.cli.core.session import Session


class MessageDispatcher:
    """Dispatch user messages through the graph and collect results."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.interrupt_handler = InterruptHandler()
        self._stream_task: asyncio.Task | None = None
        self._original_sigint = signal.getsignal(signal.SIGINT)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(self, content: str) -> None:
        """Send a user message through the graph.

        Builds a ``HumanMessage``, streams the graph response, and
        accumulates token/cost data into ``session.context``.
        """
        try:
            input_value = {"messages": [HumanMessage(content=content)]}
            await self._run_stream(input_value)
        except Exception as exc:
            render_error(self._format_stream_error(exc))

    async def resume_from_interrupt(self, interrupts: list) -> None:
        """Prompt the user for each pending interrupt, then resume the graph.

        Does nothing if the user cancels the prompt.
        """
        try:
            resume_value = await self.interrupt_handler.handle(interrupts)
            if resume_value is not None:
                await self._run_stream(Command(resume=resume_value))
        except Exception as exc:
            render_error(self._format_stream_error(exc))

    # ------------------------------------------------------------------
    # SIGINT handling
    # ------------------------------------------------------------------

    @staticmethod
    def _format_stream_error(exc: Exception) -> str:
        """Convert provider exceptions into concise CLI-facing messages."""
        message = str(exc).strip() or type(exc).__name__

        if "Rate limit reached" not in message:
            return message

        wait_match = re.search(r"Please try again in ([0-9]+(?:\.[0-9]+)?)s", message)
        requested_match = re.search(r"Requested ([0-9]+)", message)

        parts = ["Rate limit reached from the model provider"]
        if wait_match:
            parts.append(f"retry in {wait_match.group(1)}s")
        if requested_match:
            parts.append(f"request wanted {requested_match.group(1)} tokens")

        return "; ".join(parts) + "."

    def _sigint_handler(self, signum: int, frame: object) -> None:
        """First Ctrl+C cancels the stream; second falls through."""
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
        else:
            signal.signal(signal.SIGINT, self._original_sigint)

    # ------------------------------------------------------------------
    # Streaming wrapper
    # ------------------------------------------------------------------

    async def _run_stream(self, input_value: dict[str, Any] | Command) -> None:
        """Stream a graph response and accumulate cost into the context."""
        signal.signal(signal.SIGINT, self._sigint_handler)
        try:
            stats = await stream_response(
                self.session.graph,
                input_value,
                thread_id=self.session.context.thread_id,
                working_dir=self.session.context.working_dir,
                input_cost_per_mtok=self.session.context.input_cost_per_mtok,
                output_cost_per_mtok=self.session.context.output_cost_per_mtok,
            )
        except asyncio.CancelledError:
            render_info("\n  (cancelled)")
            return
        finally:
            signal.signal(signal.SIGINT, self._original_sigint)

        # Newline after the raw-streamed text.
        if stats.collected_text:
            print()

        self.session.context.accumulate(
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
            cost=stats.cost,
        )

        if stats.input_tokens or stats.output_tokens:
            render_cost_summary(stats.input_tokens, stats.output_tokens, stats.cost)
