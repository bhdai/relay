"""Interrupt handling — prompts the user when the graph pauses.

``InterruptHandler`` owns the UI interaction for LangGraph interrupts
(e.g. tool-approval prompts).  Both ``MessageDispatcher`` and
``ResumeHandler`` delegate to it instead of duplicating the prompt
logic.

Langrepl equivalent:
    ``langrepl.cli.handlers.interrupts.InterruptHandler``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prompt_toolkit.completion import WordCompleter
from langgraph.types import Interrupt
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import CompleteStyle

from relay.cli.core.context import Context
from relay.cli.ui.shared import create_bottom_toolbar, create_prompt_style
from relay.configs.approval import ApprovalMode
from relay.cli.theme import console


class InterruptHandler:
    """Collect user input for LangGraph interrupt(s) and return resume data."""

    def __init__(
        self,
        *,
        context: Context,
        on_mode_change: Callable[[ApprovalMode], None] | None = None,
    ) -> None:
        self.context = context
        self.on_mode_change = on_mode_change

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
            kb = KeyBindings()

            @kb.add(Keys.BackTab)
            def _cycle_mode(event):
                mode = self.context.cycle_approval_mode()
                if self.on_mode_change is not None:
                    self.on_mode_change(mode)
                session.style = create_prompt_style(self.context.approval_mode)
                event.app.invalidate()

            try:
                session = PromptSession(
                    completer=WordCompleter(options, ignore_case=True),
                    complete_style=CompleteStyle.COLUMN,
                    complete_while_typing=False,
                    key_bindings=kb,
                    style=create_prompt_style(self.context.approval_mode),
                    bottom_toolbar=lambda: create_bottom_toolbar(
                        "0.1.0",
                        self.context.thread_id,
                        agent_name=self.context.agent,
                        model_name=self.context.model,
                        approval_mode=self.context.approval_mode,
                    ),
                )
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
