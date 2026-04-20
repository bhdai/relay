"""Slash-command dispatcher — routes ``/`` commands to handlers.

``CommandDispatcher`` is a class-based async dispatcher that mirrors
langrepl's ``CommandDispatcher``.  Every command is async so that
commands like ``/resume`` can do I/O without special-casing in the
session loop.

Langrepl equivalent:
    ``langrepl.cli.dispatchers.commands.CommandDispatcher``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from relay.configs.approval import ApprovalMode
from relay.cli.handlers.resume import ResumeHandler
from relay.cli.theme import console
from relay.cli.ui.renderer import render_cost_summary, render_error, render_info

if TYPE_CHECKING:
    from relay.cli.core.session import Session


class CommandDispatcher:
    """Dispatch slash commands to the appropriate handler."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.resume_handler = ResumeHandler(session)

        # Registry of command name → (async handler, description).
        self.commands: dict[str, str] = {
            "/help": "Show available commands",
            "/new": "Start a new conversation thread",
            "/resume": "Resume a previous thread",
            "/approval": "Cycle/set permission mode (semi-active|active|aggressive)",
            "/approve": "Alias for /approval",
            "/cost": "Show cumulative token cost",
            "/exit": "Exit the REPL (also: exit, quit, stop)",
        }

    async def dispatch(self, command: str, **kwargs) -> bool:
        """Route a slash command.  Returns True if the REPL should exit.

        Args:
            command: The full command string (for example, ``"/resume"``).
            **kwargs: Extra context forwarded to individual handlers, such as
                ``prompt_session`` for ``/resume``.

        Returns:
            ``True`` if the REPL should exit, otherwise ``False``.
        """
        parts = command.strip().lower().split()
        if not parts:
            render_error("Empty command. Type /help for options.")
            return False

        cmd = parts[0]
        args = parts[1:]

        if cmd == "/help":
            for name, desc in self.commands.items():
                console.print(f"  {name:<10} {desc}", style="muted")
            return False

        if cmd == "/new":
            self.session.threads.record(self.session.context.thread_id)
            self.session.context.new_thread()
            render_info("New thread started.")
            return False

        if cmd == "/resume":
            prompt_session = kwargs.get("prompt_session")
            assert prompt_session is not None, "/resume requires a prompt_session"
            await self.resume_handler.handle(prompt_session)
            return False

        if cmd in ("/approval", "/approve"):
            if not args:
                mode = self.session.context.cycle_approval_mode()
                self.session.prompt.refresh_style()
                render_info(f"Permission mode: {mode.value}")
                return False

            raw = args[0]
            aliases = {
                "semi": ApprovalMode.SEMI_ACTIVE,
                "semi-active": ApprovalMode.SEMI_ACTIVE,
                "active": ApprovalMode.ACTIVE,
                "aggressive": ApprovalMode.AGGRESSIVE,
            }
            mode = aliases.get(raw)
            if mode is None:
                render_error(
                    "Invalid permission mode. Use: semi-active, active, aggressive"
                )
                return False

            self.session.context.approval_mode = mode
            self.session.prompt.refresh_style()
            render_info(f"Permission mode: {mode.value}")
            return False

        if cmd == "/cost":
            render_cost_summary(
                self.session.context.total_input_tokens,
                self.session.context.total_output_tokens,
                self.session.context.total_cost,
            )
            return False

        if cmd == "/exit":
            return True

        render_error(f"Unknown command: {cmd}. Type /help for options.")
        return False
