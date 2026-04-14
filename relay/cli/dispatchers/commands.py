"""Slash command definitions and dispatch."""

from __future__ import annotations

from relay.cli.ui.renderer import console, render_cost_summary, render_error, render_info


# ==============================================================================
# Command Registry
# ==============================================================================

COMMANDS = {
    "/help": "Show available commands",
    "/new": "Start a new conversation thread",
    "/resume": "Resume a previous thread",
    "/cost": "Show cumulative token cost",
    "/exit": "Exit the REPL (also: exit, quit, stop)",
}


# ==============================================================================
# Dispatch
# ==============================================================================


def dispatch_command(command: str, session: "Session") -> bool:  # noqa: F821
    """Handle a slash command.  Returns True if the REPL should exit.

    ``/resume`` is intentionally *not* handled here because it requires
    async I/O — the session's main loop intercepts it before calling
    this function.
    """
    cmd = command.strip().lower()

    if cmd == "/help":
        for name, desc in COMMANDS.items():
            console.print(f"  {name:<10} {desc}", style="dim")
        return False

    if cmd == "/new":
        session.threads.record(session.context.thread_id)
        session.context.new_thread()
        render_info("New thread started.")
        return False

    if cmd == "/resume":
        # Handled async in _main_loop; this is a sync fallback.
        return False

    if cmd == "/cost":
        render_cost_summary(
            session.context.total_input_tokens,
            session.context.total_output_tokens,
            session.context.total_cost,
        )
        return False

    if cmd == "/exit":
        return True

    render_error(f"Unknown command: {cmd}. Type /help for options.")
    return False
