"""Rich-based rendering for agent output.

Handles markdown rendering of AI responses, tool call display,
and cost/token summaries.  All output is routed through the themed
console so it picks up the active colour palette.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage
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


def _message_text(content: Any) -> str:
    """Extract printable text from LangChain message content values."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)

    return ""


# ==============================================================================
# Public Rendering Functions
# ==============================================================================


def render_assistant_message(
    content: str | AIMessage,
    *,
    indent_level: int = 0,
) -> None:
    """Render a complete AI response as rich markdown."""
    if isinstance(content, AIMessage):
        content = _message_text(content.content)

    if not content.strip():
        return

    normalised = _normalise_markdown(content)

    if indent_level > 0:
        prefix = "  " * indent_level
        for line in normalised.splitlines() or [normalised]:
            console.print(f"{prefix}{line}")
        return

    md = Markdown(normalised, code_theme="monokai")
    console.print(md)


def render_tool_call(name: str, args: dict, *, indent_level: int = 0) -> None:
    """Show a tool invocation notification.

    Format::

        ⚙ tool_name
          key: value (truncated)
    """
    prefix = "  " * indent_level
    header = Text(
        f"{prefix}  ⚙ {name}",
        style=console.get_style("indicator", bold=True),
    )
    console.print(header)
    for key, value in args.items():
        display = str(value)
        if len(display) > 200:
            display = display[:197] + "..."
        console.print(f"{prefix}    {key}: {display}", style="muted")


def render_tool_error(name: str, error: str, *, indent_level: int = 0) -> None:
    """Show a tool error result."""
    prefix = "  " * indent_level
    console.print(
        f"{prefix}  ✗ {name}: {error}",
        style=console.get_style("error", bold=True),
    )


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
    console.print(
        f"Error: {message}",
        style=console.get_style("error", bold=True),
    )


def render_info(message: str) -> None:
    """Display an informational message."""
    console.print(message, style="muted")
