"""Shared prompt-toolkit styling functions.

Provides ``create_prompt_style`` and ``create_bottom_toolbar`` so that
``InteractivePrompt`` and any future prompt helpers share the same
look-and-feel derived from the active theme.

Langrepl equivalent:
    ``langrepl.cli.ui.shared``
"""

from __future__ import annotations

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from relay.cli.theme import theme


def create_prompt_style() -> Style:
    """Build a prompt-toolkit ``Style`` from the active theme."""
    return Style.from_dict(
        {
            # Prompt caret / arrow
            "prompt": f"{theme.prompt_color} bold",
            # Default text
            "": f"{theme.primary_text}",
            # Completion menu
            "completion-menu.completion": (
                f"{theme.primary_text} bg:{theme.background_light}"
            ),
            "completion-menu.completion.current": (
                f"{theme.background} bg:{theme.prompt_color}"
            ),
            # Auto-suggestions
            "auto-suggestion": f"{theme.muted_text} italic",
            # Placeholder
            "placeholder": f"{theme.muted_text} italic",
            # Muted helper class
            "muted": f"{theme.muted_text}",
            # Bottom toolbar — override default reverse video
            "bottom-toolbar": f"noreverse {theme.muted_text}",
            "bottom-toolbar.text": f"noreverse {theme.muted_text}",
        }
    )


def create_bottom_toolbar(version: str, thread_id: str) -> HTML:
    """Build the bottom toolbar showing version and current thread.

    Parameters
    ----------
    version:
        Relay version string (e.g. ``"0.1.0"``).
    thread_id:
        Active conversation thread ID (truncated for display).
    """
    short_thread = thread_id[:8]
    return HTML(f"<muted> relay v{version} · thread {short_thread}</muted>")
