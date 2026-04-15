"""Tests for assistant message rendering and reasoning extraction."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage

from relay.cli.ui.renderer import (
    _extract_thinking_and_text_from_blocks,
    _extract_thinking_from_metadata,
    _extract_thinking_tags,
    assistant_message_has_renderable_content,
    assistant_message_text,
    render_assistant_message,
)


def test_extract_thinking_from_metadata_bedrock_style() -> None:
    """Provider metadata reasoning should be extracted when present."""
    message = AIMessage(
        content="Final answer",
        additional_kwargs={"thinking": {"text": "My reasoning here"}},
    )

    assert _extract_thinking_from_metadata(message) == "My reasoning here"


def test_extract_thinking_and_text_from_blocks_supports_reasoning_variants() -> None:
    """Structured reasoning blocks should be separated from visible answer text."""
    texts, thinking = _extract_thinking_and_text_from_blocks(
        [
            {"type": "thinking", "thinking": "First thought"},
            {
                "type": "reasoning",
                "summary": [
                    {"text": "Step 1: analyze"},
                    {"text": "Step 2: conclude"},
                ],
            },
            {"type": "reasoning_content", "reasoning_content": "Deep reasoning"},
            {"type": "text", "text": "Final answer"},
        ]
    )

    assert texts == ["Final answer\n"]
    assert thinking == [
        "First thought",
        "Step 1: analyze\nStep 2: conclude",
        "Deep reasoning",
    ]


def test_extract_thinking_tags_only_from_content_prefix() -> None:
    """Only leading provider-style thinking tags should be stripped."""
    content = "   <think>Thought 1</think>\n<think>Thought 2</think>\nAnswer"

    cleaned, thinking = _extract_thinking_tags(content)

    assert cleaned == "Answer"
    assert thinking == "Thought 1\n\nThought 2"


def test_assistant_message_text_excludes_reasoning_and_tags() -> None:
    """Collected assistant text should exclude provider reasoning content."""
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Block reasoning"},
            {"type": "text", "text": "<think>Inline reasoning</think>\nVisible answer"},
        ],
        additional_kwargs={"thinking": {"text": "Metadata reasoning"}},
    )

    assert assistant_message_text(message) == "Visible answer"
    assert assistant_message_has_renderable_content(message) is True


def test_assistant_message_has_renderable_content_for_reasoning_only_message() -> None:
    """Reasoning-only messages should still be considered renderable."""
    message = AIMessage(
        content=[{"type": "reasoning_content", "reasoning_content": "Detailed reasoning"}]
    )

    assert assistant_message_has_renderable_content(message) is True
    assert assistant_message_text(message) == ""


def test_render_assistant_message_renders_reasoning_only_message() -> None:
    """Rendering should not drop assistant messages that only contain reasoning."""
    message = AIMessage(
        content=[{"type": "reasoning_content", "reasoning_content": "Detailed reasoning"}]
    )

    with patch("relay.cli.ui.renderer.console.print") as print_mock:
        render_assistant_message(message)

    assert print_mock.call_count == 1
    assert str(print_mock.call_args.args[0]) == "Detailed reasoning"
