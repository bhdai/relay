"""Rich-based rendering for agent output.

Handles markdown rendering of AI responses, including provider-native
reasoning content, plus tool call display and cost/token summaries.
All output is routed through the themed console so it picks up the
active colour palette.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage
from rich.markdown import Markdown
from rich.style import Style
from rich.text import Text

from relay.cli.theme import console


# ==============================================================================
# Markdown Helpers
# ==============================================================================
#
# LLMs sometimes emit escaped closing fences (e.g. ``\`\`\```) which
# confuse markdown renderers.  We normalise them before display.

_ESCAPED_FENCE_RE = re.compile(r"\\`\\`\\`")
_THINK_TAG_PREFIX_RE = re.compile(r"\s*((?:<think>.*?</think>\s*)+)(.*)\Z", re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _normalise_markdown(content: str) -> str:
    """Fix common markdown quirks from LLM output."""
    return _ESCAPED_FENCE_RE.sub("```", content)


def _append_text_fragment(parts: list[str], fragment: Any) -> None:
    """Normalize text fragments so block-based content stays readable."""
    text = str(fragment).strip(" ")
    if not text:
        return

    if not text.endswith("\n"):
        text += "\n"

    parts.append(text)


def _extract_thinking_from_metadata(message: AIMessage) -> str | None:
    """Extract provider-native reasoning stored in message metadata."""
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if not isinstance(additional_kwargs, dict):
        return None

    thinking_data = additional_kwargs.get("thinking")
    if isinstance(thinking_data, dict):
        thinking_text = thinking_data.get("text")
        if isinstance(thinking_text, str) and thinking_text.strip():
            return thinking_text.strip()

    if isinstance(thinking_data, str) and thinking_data.strip():
        return thinking_data.strip()

    return None


def _extract_thinking_and_text_from_blocks(
    blocks: list[Any],
) -> tuple[list[str], list[str]]:
    """Split content blocks into visible answer text and reasoning text."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for block in blocks:
        if isinstance(block, str):
            _append_text_fragment(text_parts, block)
            continue

        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            _append_text_fragment(text_parts, block.get("text", ""))
            continue

        if block_type == "thinking":
            thinking_text = block.get("thinking")
            if isinstance(thinking_text, str) and thinking_text.strip():
                thinking_parts.append(thinking_text.strip())
            continue

        if block_type == "reasoning":
            summary = block.get("summary")
            if not isinstance(summary, list):
                continue

            summary_parts = [
                item.get("text", "").strip()
                for item in summary
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
                and item.get("text", "").strip()
            ]
            if summary_parts:
                thinking_parts.append("\n".join(summary_parts))
            continue

        if block_type == "reasoning_content":
            reasoning_text = block.get("reasoning_content")
            if isinstance(reasoning_text, str) and reasoning_text.strip():
                thinking_parts.append(reasoning_text.strip())

    return text_parts, thinking_parts


def _extract_thinking_tags(content: str) -> tuple[str, str | None]:
    """Extract leading provider-style ``<think>`` tags from content."""
    match = _THINK_TAG_PREFIX_RE.match(content)
    if not match:
        return content, None

    prefix, remainder = match.groups()
    thinking_parts = [part.strip() for part in _THINK_TAG_RE.findall(prefix) if part.strip()]
    if not thinking_parts:
        return content, None

    return remainder.strip(), "\n\n".join(thinking_parts)


def _assistant_text_and_thinking(message: AIMessage) -> tuple[str, list[str]]:
    """Extract the visible answer text and any reasoning snippets."""
    thinking_parts: list[str] = []

    metadata_thinking = _extract_thinking_from_metadata(message)
    if metadata_thinking:
        thinking_parts.append(metadata_thinking)

    content = message.content
    text = ""

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts, block_thinking = _extract_thinking_and_text_from_blocks(content)
        thinking_parts.extend(block_thinking)
        text = "".join(text_parts)

    text, xml_thinking = _extract_thinking_tags(text)
    if xml_thinking:
        thinking_parts.append(xml_thinking)

    cleaned_thinking = [part.strip() for part in thinking_parts if part.strip()]
    return text.strip(), cleaned_thinking


def assistant_message_text(message: AIMessage) -> str:
    """Return the answer text that should count as assistant output."""
    text, _thinking = _assistant_text_and_thinking(message)
    return text


def assistant_message_has_renderable_content(message: AIMessage) -> bool:
    """Report whether a message has answer text or visible reasoning."""
    text, thinking = _assistant_text_and_thinking(message)
    return bool(text or thinking)


def _render_plain_text(
    content: str,
    *,
    indent_level: int = 0,
    style: Style | None = None,
) -> None:
    """Render plain multiline content with optional indentation."""
    prefix = "  " * indent_level
    lines = content.splitlines() or [content]
    for line in lines:
        text = f"{prefix}{line}"
        if style is None:
            console.print(text)
        else:
            console.print(Text(text, style=style))


# ==============================================================================
# Public Rendering Functions
# ==============================================================================


def render_assistant_message(
    content: str | AIMessage,
    *,
    indent_level: int = 0,
) -> None:
    """Render a complete assistant response, including reasoning content."""
    is_error = False
    thinking_parts: list[str] = []

    if isinstance(content, AIMessage):
        is_error = bool(getattr(content, "is_error", False))
        content, thinking_parts = _assistant_text_and_thinking(content)
    elif isinstance(content, str):
        content, xml_thinking = _extract_thinking_tags(content)
        content = content.strip()
        if xml_thinking:
            thinking_parts.append(xml_thinking)

    if not content and not thinking_parts:
        return

    if is_error:
        error_text = content or "\n\n".join(thinking_parts)
        if error_text:
            _render_plain_text(
                error_text,
                indent_level=indent_level,
                style=console.get_style("error", bold=True),
            )
        return

    if thinking_parts:
        thinking_style = Style.combine([console.get_style("muted"), Style(italic=True)])
        _render_plain_text(
            "\n\n".join(thinking_parts),
            indent_level=indent_level,
            style=thinking_style,
        )
        if content:
            console.print("")

    if not content:
        return

    normalised = _normalise_markdown(content)

    if indent_level > 0:
        _render_plain_text(normalised, indent_level=indent_level)
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
