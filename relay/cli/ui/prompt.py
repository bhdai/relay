"""Prompt-toolkit session and input handling.

``InteractivePrompt`` owns the ``PromptSession``, keybindings, and
bottom toolbar so that ``Session`` no longer imports prompt-toolkit
directly.

Langrepl equivalent:
    ``langrepl.cli.ui.prompt.InteractivePrompt``
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from relay.cli.ui.shared import create_bottom_toolbar, create_prompt_style

if TYPE_CHECKING:
    from relay.cli.core.context import Context


class InteractivePrompt:
    """Interactive prompt using prompt-toolkit with themed styling.

    Handles:
    - PromptSession creation with themed style and bottom toolbar
    - Ctrl+C / double-press-to-quit behaviour
    - Ctrl+J for multiline input
    """

    def __init__(self, context: Context) -> None:
        self.context = context

        self._last_ctrl_c_time: float | None = None
        self._ctrl_c_timeout = 0.30
        self._show_quit_message = False

        self._session = self._build_session()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_session(self) -> PromptSession:
        kb = self._create_key_bindings()
        style = create_prompt_style()

        return PromptSession(
            history=InMemoryHistory(),
            key_bindings=kb,
            style=style,
            bottom_toolbar=self._get_bottom_toolbar,
            multiline=False,
            wrap_lines=True,
        )

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add(Keys.ControlC)
        def _ctrl_c(event):
            """Clear input on first press; quit on double-press."""
            buf = event.current_buffer
            now = time.time()

            if buf.text.strip():
                buf.text = ""
                self._reset_ctrl_c()
                return

            if (
                self._last_ctrl_c_time is not None
                and (now - self._last_ctrl_c_time < self._ctrl_c_timeout
                     or self._show_quit_message)
            ):
                self._reset_ctrl_c()
                event.app.exit(exception=EOFError())
                return

            self._last_ctrl_c_time = now
            self._show_quit_message = True
            self._schedule_hide(event.app)

        @kb.add(Keys.ControlJ)
        def _ctrl_j(event):
            """Insert a literal newline for multiline editing."""
            event.current_buffer.insert_text("\n")

        return kb

    # ------------------------------------------------------------------
    # Toolbar / state helpers
    # ------------------------------------------------------------------

    def _get_bottom_toolbar(self) -> HTML:
        if self._show_quit_message:
            return HTML("<muted> Ctrl+C again to quit</muted>")
        return create_bottom_toolbar("0.1.0", self.context.thread_id)

    def _reset_ctrl_c(self) -> None:
        self._last_ctrl_c_time = None
        self._show_quit_message = False

    def _schedule_hide(self, app) -> None:
        """Clear the quit banner after the timeout expires."""
        def _hide():
            self._reset_ctrl_c()
            app.invalidate()

        try:
            app.loop.call_later(self._ctrl_c_timeout, _hide)
        except Exception:
            self._reset_ctrl_c()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def session(self) -> PromptSession:
        """Underlying ``PromptSession`` (for handlers that need it)."""
        return self._session

    async def get_input(self) -> str | None:
        """Read a line of input, returning ``None`` on EOF/interrupt."""
        try:
            return await self._session.prompt_async("❯ ")
        except (EOFError, KeyboardInterrupt):
            return None
