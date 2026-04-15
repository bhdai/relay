"""Interrupt handling — prompts the user when the graph pauses.

``InterruptHandler`` owns the UI interaction for LangGraph interrupts
(e.g. tool-approval prompts).  Both ``MessageDispatcher`` and
``ResumeHandler`` delegate to it instead of duplicating the prompt
logic.

Langrepl equivalent:
    ``langrepl.cli.handlers.interrupts.InterruptHandler``
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Interrupt
from prompt_toolkit import PromptSession

from relay.cli.theme import console


class InterruptHandler:
    """Collect user input for LangGraph interrupt(s) and return resume data."""

    async def handle(
        self, interrupts: list[Interrupt]
    ) -> dict[str, Any] | None:
        """Prompt the user for each pending interrupt.

        Returns a ``{interrupt_id: user_choice}`` dict suitable for
        ``Command(resume=...)``, or ``None`` if the user cancels.
        """
        resume_map: dict[str, Any] = {}

        for intr in interrupts:
            payload = intr.value

            # Display the interrupt question.
            if hasattr(payload, "question"):
                console.print(
                    f"  ? {payload.question}",
                    style=console.get_style("warning", bold=True),
                )
            else:
                console.print(
                    f"  ? {payload}",
                    style=console.get_style("warning", bold=True),
                )

            # Show options if available.
            options: list[str] = []
            if hasattr(payload, "options") and payload.options:
                options = payload.options
                for j, opt in enumerate(options, 1):
                    console.print(f"    {j}. {opt}", style="muted")

            # Collect user response.
            try:
                session = PromptSession()
                answer = await session.prompt_async("  → ")
            except (EOFError, KeyboardInterrupt):
                return None

            answer = answer.strip()

            # If the user typed a number and we have options, map it.
            if options and answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx < len(options):
                    answer = options[idx]

            resume_map[intr.id] = answer

        return resume_map
