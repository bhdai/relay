"""Tests for CLI graph streaming output handling."""

from __future__ import annotations

from unittest.mock import call
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from relay.cli.core.streaming import stream_response
from relay.configs.approval import ApprovalMode


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
async def test_stream_response_extracts_text_from_chunk_blocks() -> None:
    """Chunk content blocks should render as a finalized assistant message."""
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

    with patch("relay.cli.core.streaming.render_assistant_message") as render_message:
        stats = await stream_response(graph, {"messages": []}, thread_id="thread-1")

    assert stats.collected_text == "Hello"
    rendered = render_message.call_args.args[0]
    assert isinstance(rendered, AIMessage)
    assert rendered.content == [{"type": "text", "text": "Hello"}]


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

    rendered = render_message.call_args.args[0]
    assert isinstance(rendered, AIMessage)
    assert rendered.content == [{"type": "text", "text": "Final answer"}]
    assert stats.collected_text == "Final answer"


@pytest.mark.asyncio
async def test_stream_response_renders_reasoning_only_messages() -> None:
    """Reasoning-only messages should still reach the renderer."""
    graph = _FakeGraph(
        [
            (
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content=[
                                    {
                                        "type": "reasoning_content",
                                        "reasoning_content": "Detailed reasoning",
                                    }
                                ]
                            )
                        ]
                    }
                },
            )
        ]
    )

    with patch("relay.cli.core.streaming.render_assistant_message") as render_message:
        stats = await stream_response(graph, {"messages": []}, thread_id="thread-1")

    rendered = render_message.call_args.args[0]
    assert isinstance(rendered, AIMessage)
    assert rendered.content == [
        {
            "type": "reasoning_content",
            "reasoning_content": "Detailed reasoning",
        }
    ]
    assert stats.collected_text == ""


@pytest.mark.asyncio
async def test_stream_response_passes_pricing_into_agent_context() -> None:
    """Streaming should propagate configured pricing into AgentContext."""
    graph = _FakeGraph([])

    # approval_mode is still accepted as a parameter (for backward compatibility
    # and future Phase 5 translation), but is no longer stored on AgentContext.
    await stream_response(
        graph,
        {"messages": []},
        thread_id="thread-1",
        input_cost_per_mtok=0.4,
        output_cost_per_mtok=1.6,
        approval_mode=ApprovalMode.AGGRESSIVE,
    )

    assert graph.last_context is not None
    assert graph.last_context.input_cost_per_mtok == 0.4
    assert graph.last_context.output_cost_per_mtok == 1.6


@pytest.mark.asyncio
async def test_stream_response_renders_custom_subagent_updates() -> None:
    """Custom stream events should surface delegated subagent narration and tools."""
    graph = _FakeGraph(
        [
            (
                "custom",
                {
                    "relay_event": "subagent_start",
                    "subagent": "explorer",
                    "description": "Inspect the repository",
                },
            ),
            (
                "custom",
                {
                    "relay_event": "subagent_message",
                    "subagent": "explorer",
                    "message": AIMessageChunk(content=[{"type": "text", "text": "Inspecting files"}]),
                },
            ),
            (
                "custom",
                {
                    "relay_event": "subagent_update",
                    "subagent": "explorer",
                    "update": {
                        "agent": {
                            "messages": [
                                AIMessage(
                                    content="",
                                    tool_calls=[
                                        {
                                            "name": "read_file",
                                            "args": {"file_path": "relay/main.py"},
                                            "id": "call_123",
                                        }
                                    ],
                                )
                            ]
                        }
                    },
                },
            ),
        ]
    )

    with (
        patch("relay.cli.core.streaming.render_info") as render_info,
        patch("relay.cli.core.streaming.render_assistant_message") as render_message,
        patch("relay.cli.core.streaming.render_tool_call") as render_tool_call,
    ):
        await stream_response(graph, {"messages": []}, thread_id="thread-1")

    render_info.assert_called_once_with("  ↳ explorer: Inspect the repository")
    rendered = render_message.call_args.args[0]
    assert isinstance(rendered, AIMessage)
    assert rendered.content == [{"type": "text", "text": "Inspecting files"}]
    render_tool_call.assert_called_once_with(
        "read_file",
        {"file_path": "relay/main.py"},
        indent_level=1,
    )


@pytest.mark.asyncio
async def test_stream_response_renders_multiple_assistant_phases_in_one_turn() -> None:
    """Later assistant messages should still render after earlier streamed text."""
    graph = _FakeGraph(
        [
            (
                "messages",
                (
                    AIMessageChunk(content=[{"type": "text", "text": "Planning next step"}]),
                    {},
                ),
            ),
            (
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "read_file",
                                        "args": {"file_path": "relay/main.py"},
                                        "id": "call_123",
                                    }
                                ],
                            )
                        ]
                    }
                },
            ),
            (
                "updates",
                {
                    "agent": {
                        "messages": [AIMessage(content=[{"type": "text", "text": "Done investigating"}])]
                    }
                },
            ),
        ]
    )

    with (
        patch("relay.cli.core.streaming.render_assistant_message") as render_message,
        patch("relay.cli.core.streaming.render_tool_call") as render_tool_call,
    ):
        await stream_response(graph, {"messages": []}, thread_id="thread-1")

    assert render_message.call_count == 2
    first_message = render_message.call_args_list[0].args[0]
    second_message = render_message.call_args_list[1].args[0]
    assert isinstance(first_message, AIMessage)
    assert isinstance(second_message, AIMessage)
    assert first_message.content == [{"type": "text", "text": "Planning next step"}]
    assert second_message.content == [{"type": "text", "text": "Done investigating"}]
    render_tool_call.assert_called_once_with(
        "read_file",
        {"file_path": "relay/main.py"},
        indent_level=0,
    )


@pytest.mark.asyncio
async def test_stream_response_supports_namespaced_events() -> None:
    """Three-field stream events from subgraph-aware streams should be accepted."""
    graph = _FakeGraph(
        [
            (
                ("tools",),
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "ls",
                                        "args": {"path": "."},
                                        "id": "call_456",
                                    }
                                ],
                            )
                        ]
                    }
                },
            )
        ]
    )

    with patch("relay.cli.core.streaming.render_tool_call") as render_tool_call:
        await stream_response(graph, {"messages": []}, thread_id="thread-1")

    render_tool_call.assert_called_once_with("ls", {"path": "."}, indent_level=1)