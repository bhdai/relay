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
from relay.configs.approval import ApprovalMode


def _prompt_color_for_mode(mode: ApprovalMode | None) -> str:
    """Return prompt accent color for the current permission mode."""
    if mode == ApprovalMode.ACTIVE:
        return theme.warning_color
    if mode == ApprovalMode.AGGRESSIVE:
        return theme.error_color
    return theme.prompt_color


def create_prompt_style(approval_mode: ApprovalMode | None = None) -> Style:
    """Build a prompt-toolkit ``Style`` from theme + permission mode."""
    prompt_color = _prompt_color_for_mode(approval_mode)

    return Style.from_dict(
        {
            # Prompt caret / arrow
            "prompt": f"{prompt_color} bold",
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
            # Permission mode segment in toolbar
            "toolbar.mode": f"noreverse {prompt_color}",
        }
    )


def create_bottom_toolbar(
    version: str,
    thread_id: str,
    agent_name: str | None = None,
    model_name: str | None = None,
    approval_mode: ApprovalMode | None = None,
) -> HTML:
    """Build the bottom toolbar showing version and current thread.

    Parameters
    ----------
    version:
        Relay version string (e.g. ``"0.1.0"``).
    thread_id:
        Active conversation thread ID (truncated for display).
    agent_name:
        Active agent profile, if one was selected.
    model_name:
        Active model override, if one was selected.
    """
    short_thread = thread_id[:8]
    segments = [f"relay v{version}"]

    if agent_name and model_name:
        segments.append(f"{agent_name}:{model_name}")
    elif agent_name:
        segments.append(agent_name)
    elif model_name:
        segments.append(model_name)

    segments.append(f"thread {short_thread}")

    if approval_mode is None:
        return HTML(f"<muted> {' · '.join(segments)}</muted>")

    base = " · ".join(segments)
    return HTML(
        f"<muted> {base} · </muted><toolbar.mode>{approval_mode.value}</toolbar.mode><muted> </muted>"
    )
