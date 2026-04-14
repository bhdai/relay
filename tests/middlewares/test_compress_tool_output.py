"""Tests for CompressToolOutputMiddleware."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from relay.agents.context import AgentContext
from relay.middlewares.compress_tool_output import (
    CompressToolOutputMiddleware,
    _estimate_tokens,
)
from relay.tools.internal.memory import read_memory_file


class TestEstimateTokens:
    """Tests for the _estimate_tokens helper."""

    def test_fallback_character_estimate(self):
        """When no model tokenizer is available, use len // 4."""
        model = Mock()
        model.get_num_tokens_from_messages.side_effect = NotImplementedError
        result = _estimate_tokens("a" * 400, model)
        assert result == 100

    def test_uses_model_tokenizer(self):
        """When the model supports it, use get_num_tokens_from_messages."""
        model = Mock()
        model.get_num_tokens_from_messages.return_value = 42
        result = _estimate_tokens("hello world", model)
        assert result == 42


class TestCompressToolOutputMiddleware:
    """Tests for CompressToolOutputMiddleware class."""

    @pytest.mark.asyncio
    async def test_compresses_large_output(self):
        """Large tool output should be stored in files and replaced with a reference."""
        model = Mock()
        # Return a large token count to trigger compression.
        model.get_num_tokens_from_messages.return_value = 20_000
        middleware = CompressToolOutputMiddleware(model)

        large_content = "x" * 40_000  # Large string

        request = Mock(spec=ToolCallRequest)
        request.tool_call = {"id": "call_1", "name": "test_tool"}
        request.runtime = Mock()
        request.runtime.context = AgentContext(tool_output_max_tokens=10)

        handler = AsyncMock(
            return_value=ToolMessage(
                name="test_tool",
                content=large_content,
                tool_call_id="call_1",
            )
        )

        result = await middleware.awrap_tool_call(request, handler)

        # Should return a Command with compressed message and file.
        assert isinstance(result, Command)
        assert result.update is not None
        assert "messages" in result.update
        assert "files" in result.update

        compressed_msg = result.update["messages"][0]
        assert "tool_output_call_1.txt" in compressed_msg.content
        assert "stored in virtual file" in compressed_msg.content

        files = result.update["files"]
        assert "tool_output_call_1.txt" in files
        assert files["tool_output_call_1.txt"] == large_content

    @pytest.mark.asyncio
    async def test_does_not_compress_small_output(self):
        """Small tool output should pass through unchanged."""
        model = Mock()
        model.get_num_tokens_from_messages.return_value = 5
        middleware = CompressToolOutputMiddleware(model)

        small_content = "small output"

        request = Mock(spec=ToolCallRequest)
        request.tool_call = {"id": "call_1", "name": "test_tool"}
        request.runtime = Mock()
        request.runtime.context = AgentContext(tool_output_max_tokens=10_000)

        handler = AsyncMock(
            return_value=ToolMessage(
                name="test_tool",
                content=small_content,
                tool_call_id="call_1",
            )
        )

        result = await middleware.awrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert result.content == small_content

    @pytest.mark.asyncio
    async def test_skips_compression_for_errors(self):
        """Error messages should never be compressed."""
        model = Mock()
        model.get_num_tokens_from_messages.return_value = 20_000
        middleware = CompressToolOutputMiddleware(model)

        error_content = "Error: " + ("x" * 40_000)

        request = Mock(spec=ToolCallRequest)
        request.tool_call = {"id": "call_1", "name": "test_tool"}
        request.runtime = Mock()
        request.runtime.context = AgentContext(tool_output_max_tokens=10)

        error_msg = ToolMessage(
            name="test_tool",
            content=error_content,
            tool_call_id="call_1",
        )
        error_msg.status = "error"

        handler = AsyncMock(return_value=error_msg)

        result = await middleware.awrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert result.content == error_content

    @pytest.mark.asyncio
    async def test_skips_compression_for_read_memory_file(self):
        """Output from read_memory_file should not be compressed."""
        model = Mock()
        model.get_num_tokens_from_messages.return_value = 20_000
        middleware = CompressToolOutputMiddleware(model)

        large_content = "x" * 40_000

        request = Mock(spec=ToolCallRequest)
        request.tool_call = {"id": "call_1", "name": read_memory_file.name}
        request.runtime = Mock()
        request.runtime.context = AgentContext(tool_output_max_tokens=10)

        handler = AsyncMock(
            return_value=ToolMessage(
                name=read_memory_file.name,
                content=large_content,
                tool_call_id="call_1",
            )
        )

        result = await middleware.awrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert result.content == large_content

    @pytest.mark.asyncio
    async def test_passes_through_command_from_handler(self):
        """Commands returned by the handler should pass through unchanged."""
        model = Mock()
        middleware = CompressToolOutputMiddleware(model)

        request = Mock(spec=ToolCallRequest)
        request.tool_call = {"id": "call_1", "name": "test_tool"}
        request.runtime = Mock()
        request.runtime.context = AgentContext()

        cmd: Command = Command(update={"messages": []})
        handler = AsyncMock(return_value=cmd)

        result = await middleware.awrap_tool_call(request, handler)
        assert result is cmd

    @pytest.mark.asyncio
    async def test_no_compression_when_max_tokens_missing(self):
        """When tool_output_max_tokens is 0, skip compression."""
        model = Mock()
        model.get_num_tokens_from_messages.return_value = 20_000
        middleware = CompressToolOutputMiddleware(model)

        large_content = "x" * 40_000

        request = Mock(spec=ToolCallRequest)
        request.tool_call = {"id": "call_1", "name": "test_tool"}
        request.runtime = Mock()
        request.runtime.context = AgentContext(tool_output_max_tokens=0)

        handler = AsyncMock(
            return_value=ToolMessage(
                name="test_tool",
                content=large_content,
                tool_call_id="call_1",
            )
        )

        result = await middleware.awrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert result.content == large_content
