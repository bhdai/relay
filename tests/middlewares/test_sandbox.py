"""Tests for SandboxMiddleware."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from relay.agents.context import AgentContext
from relay.middlewares.sandbox import SandboxMiddleware
from relay.sandboxes.backend import SandboxBackend


class _FakeBackend(SandboxBackend):
    """Minimal in-process backend for tests."""

    def __init__(self, result: dict | None = None) -> None:
        self._result = result or {"success": True, "content": "sandboxed-output"}

    async def execute(self, *, module_path, tool_name, args, tool_runtime):
        return self._result


def _make_request(
    tool_name: str = "test_tool",
    call_id: str = "call_1",
    args: dict | None = None,
) -> Mock:
    mock_tool = Mock()
    mock_tool.metadata = {}
    mock_tool.__module__ = "relay.tools.impl.web"
    mock_tool.name = tool_name

    request = Mock(spec=ToolCallRequest)
    request.tool_call = {
        "id": call_id,
        "name": tool_name,
        "args": args or {},
    }
    request.tool = mock_tool
    request.runtime = Mock()
    request.runtime.context = AgentContext()
    return request


class TestSandboxMiddleware:
    """Tests for SandboxMiddleware.awrap_tool_call."""

    @pytest.mark.asyncio
    async def test_blocked_tool_returns_error(self):
        """A tool not in the map should be blocked."""
        mw = SandboxMiddleware(tool_sandbox_map={})
        request = _make_request()
        handler = AsyncMock()

        result = await mw.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "blocked" in result.content.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_backend_passes_through(self):
        """A tool mapped to None should execute normally (no sandbox)."""
        mw = SandboxMiddleware(tool_sandbox_map={"test_tool": None})
        request = _make_request()
        handler = AsyncMock(
            return_value=ToolMessage(
                name="test_tool", content="normal", tool_call_id="call_1"
            )
        )

        result = await mw.awrap_tool_call(request, handler)

        assert result.content == "normal"
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_sandboxed_execution_success(self):
        """Tool routed through backend should return backend content."""
        backend = _FakeBackend({"success": True, "content": "sandbox-result"})
        mw = SandboxMiddleware(tool_sandbox_map={"test_tool": backend})
        request = _make_request()
        handler = AsyncMock()

        result = await mw.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "sandbox-result"
        # Handler should NOT be called when sandbox takes over.
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_sandboxed_execution_failure(self):
        """Backend failure should produce an error ToolMessage."""
        backend = _FakeBackend({
            "success": False,
            "error": "permission denied",
            "traceback": "Traceback: ...",
        })
        mw = SandboxMiddleware(tool_sandbox_map={"test_tool": backend})
        request = _make_request()
        handler = AsyncMock()

        result = await mw.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "permission denied" in result.content
        assert "Traceback" in result.content

    @pytest.mark.asyncio
    async def test_no_tool_reference_falls_through(self):
        """If request.tool is None, fall through to handler."""
        backend = _FakeBackend()
        mw = SandboxMiddleware(tool_sandbox_map={"test_tool": backend})
        request = _make_request()
        request.tool = None
        handler = AsyncMock(
            return_value=ToolMessage(
                name="test_tool", content="fallback", tool_call_id="call_1"
            )
        )

        result = await mw.awrap_tool_call(request, handler)
        assert result.content == "fallback"
        handler.assert_called_once()
