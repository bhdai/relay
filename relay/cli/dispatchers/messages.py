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
import signal
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from relay.cli.core.streaming import prompt_for_interrupt, stream_response
from relay.cli.ui.renderer import render_cost_summary, render_info

if TYPE_CHECKING:
    from relay.cli.core.session import Session


class MessageDispatcher:
    """Dispatch user messages through the graph and collect results."""

    def __init__(self, session: Session) -> None:
        self.session = session
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
        input_value = {"messages": [HumanMessage(content=content)]}
        await self._run_stream(input_value)

    async def resume_from_interrupt(self, interrupts: list) -> None:
        """Prompt the user for each pending interrupt, then resume the graph.

        Does nothing if the user cancels the prompt.
        """
        resume_value = await prompt_for_interrupt(interrupts)
        if resume_value is not None:
            await self._run_stream(Command(resume=resume_value))

    # ------------------------------------------------------------------
    # SIGINT handling
    # ------------------------------------------------------------------

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
