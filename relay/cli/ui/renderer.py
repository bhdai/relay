"""Rich-based rendering for agent output.

Handles markdown rendering of AI responses, tool call display,
and cost/token summaries.  All output is routed through the themed
console so it picks up the active colour palette.
"""

from __future__ import annotations

import re

from rich.markdown import Markdown
from rich.text import Text

from relay.cli.theme import console


# ==============================================================================
# Markdown Helpers
# ==============================================================================
#
# LLMs sometimes emit escaped closing fences (e.g. ``\`\`\```) which
# confuse markdown renderers.  We normalise them before display.

_ESCAPED_FENCE_RE = re.compile(r"\\`\\`\\`")


def _normalise_markdown(content: str) -> str:
    """Fix common markdown quirks from LLM output."""
    return _ESCAPED_FENCE_RE.sub("```", content)


# ==============================================================================
# Public Rendering Functions
# ==============================================================================


def render_assistant_message(content: str) -> None:
    """Render a complete AI response as rich markdown."""
    if not content.strip():
        return
    normalised = _normalise_markdown(content)
    md = Markdown(normalised, code_theme="monokai")
    console.print(md)


def render_tool_call(name: str, args: dict) -> None:
    """Show a tool invocation notification.

    Format::

        ⚙ tool_name
          key: value (truncated)
    """
    header = Text(f"  ⚙ {name}", style="indicator bold")
    console.print(header)
    for key, value in args.items():
        display = str(value)
        if len(display) > 200:
            display = display[:197] + "..."
        console.print(f"    {key}: {display}", style="muted")


def render_tool_error(name: str, error: str) -> None:
    """Show a tool error result."""
    console.print(f"  ✗ {name}: {error}", style="error bold")


def render_cost_summary(
    input_tokens: int,
    output_tokens: int,
    total_cost: float,
) -> None:
    """Print a token/cost summary line after a response."""
    parts = []
    if input_tokens or output_tokens:
        parts.append(f"{input_tokens:,}↑ {output_tokens:,}↓")
    if total_cost > 0:
        parts.append(f"${total_cost:.4f}")
    if parts:
        summary = " · ".join(parts)
        console.print(f"  [{summary}]", style="muted")


def render_error(message: str) -> None:
    """Display an error message."""
    console.print(f"Error: {message}", style="error bold")


def render_info(message: str) -> None:
    """Display an informational message."""
    console.print(message, style="muted")
