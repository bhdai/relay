"""Tests for CLI graph streaming output handling."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from relay.cli.core.streaming import stream_response


class _FakeGraph:
    """Minimal async graph stub for streaming tests."""

    def __init__(self, events):
        self._events = events
        self.last_context = None

    async def astream(self, *_args, **_kwargs):
        self.last_context = _kwargs.get("context")
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_stream_response_extracts_text_from_chunk_blocks(capsys) -> None:
    """Chunk content blocks should be rendered like plain text chunks."""
    graph = _FakeGraph(
        [
            (
                "messages",
                (
                    AIMessageChunk(content=[{"type": "text", "text": "Hello"}]),
                    {},
                ),
            )
        ]
    )

    stats = await stream_response(graph, {"messages": []}, thread_id="thread-1")

    assert stats.collected_text == "Hello"
    assert capsys.readouterr().out == "Hello"


@pytest.mark.asyncio
async def test_stream_response_renders_final_ai_message_when_no_chunks() -> None:
    """Final AI messages in update events should still reach the CLI renderer."""
    graph = _FakeGraph(
        [
            (
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(content=[{"type": "text", "text": "Final answer"}])
                        ]
                    }
                },
            )
        ]
    )
    with patch("relay.cli.core.streaming.render_assistant_message") as render_message:
        stats = await stream_response(graph, {"messages": []}, thread_id="thread-1")

    render_message.assert_called_once_with("Final answer")
    assert stats.collected_text == "Final answer"


@pytest.mark.asyncio
async def test_stream_response_passes_pricing_into_agent_context() -> None:
    """Streaming should propagate configured pricing into AgentContext."""
    graph = _FakeGraph([])

    await stream_response(
        graph,
        {"messages": []},
        thread_id="thread-1",
        input_cost_per_mtok=0.4,
        output_cost_per_mtok=1.6,
    )

    assert graph.last_context is not None
    assert graph.last_context.input_cost_per_mtok == 0.4
    assert graph.last_context.output_cost_per_mtok == 1.6