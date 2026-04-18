"""Tests for SandboxMiddleware."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from relay.agents.context import AgentContext
from relay.middlewares.sandbox import SandboxMiddleware
from relay.sandboxes.backend import SandboxBackend, SandboxBinding


class _FakeBackend(SandboxBackend):
    """Minimal in-process backend for tests."""

    def __init__(self, result: dict | None = None) -> None:
        # Skip the base __init__ (requires a real config + working_dir).
        self._result = result or {"success": True, "content": "sandboxed-output"}

    async def execute(self, *, module_path, tool_name, args, tool_runtime=None, timeout=120.0):
        return self._result

    def build_command(self, command, extra_env=None):
        return command

    def validate_environment(self):
        pass


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
        """A tool not matching any binding should be blocked."""
        mw = SandboxMiddleware.from_tool_map({})
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
        mw = SandboxMiddleware.from_tool_map({"test_tool": None})
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
        mw = SandboxMiddleware.from_tool_map({"test_tool": backend})
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
        mw = SandboxMiddleware.from_tool_map({"test_tool": backend})
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
        mw = SandboxMiddleware.from_tool_map({"test_tool": backend})
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

    @pytest.mark.asyncio
    async def test_binding_based_resolution(self):
        """SandboxMiddleware constructed with bindings resolves via patterns."""
        backend = _FakeBackend({"success": True, "content": "via-binding"})
        bindings = [
            SandboxBinding(patterns=["impl:terminal:*"], backend=backend),
            SandboxBinding(patterns=["internal:*:*"], backend=None),
        ]
        mw = SandboxMiddleware(
            bindings,
            tool_module_map={"run_command": "terminal", "manage_memory": "memory"},
        )

        # run_command matches impl:terminal:* → sandboxed
        request = _make_request(tool_name="run_command")
        handler = AsyncMock()
        result = await mw.awrap_tool_call(request, handler)
        assert result.content == "via-binding"
        handler.assert_not_called()

        # manage_memory matches internal:*:* → passthrough (None backend)
        request2 = _make_request(tool_name="manage_memory")
        handler2 = AsyncMock(
            return_value=ToolMessage(
                name="manage_memory", content="memory-ok", tool_call_id="call_1"
            )
        )
        result2 = await mw.awrap_tool_call(request2, handler2)
        assert result2.content == "memory-ok"
        handler2.assert_called_once()
