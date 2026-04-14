"""Middleware for sandboxed tool execution.

When sandboxing is enabled, the middleware intercepts tool calls and
routes them through a ``SandboxBackend`` which runs the tool in a
restricted subprocess.  Tools not present in the ``tool_sandbox_map``
are blocked entirely (the agent receives an error message).  Tools
mapped to ``None`` execute normally (no sandbox).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.utils.messages import create_tool_message

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest

    from relay.sandboxes.backend import SandboxBackend

logger = logging.getLogger(__name__)


class SandboxMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Run tool calls inside an OS-level sandbox.

    Parameters
    ----------
    tool_sandbox_map:
        Mapping from tool name to the ``SandboxBackend`` that should
        execute it, or ``None`` for unsandboxed passthrough.  Tools
        **not** in the map are blocked.
    """

    def __init__(
        self,
        tool_sandbox_map: dict[str, SandboxBackend | None],
    ) -> None:
        super().__init__()
        self.tool_sandbox_map = tool_sandbox_map

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_sandbox_backend(
        self,
        tool_name: str,
    ) -> tuple[SandboxBackend | None, bool]:
        """Look up the backend for *tool_name*.

        Returns
        -------
        (backend, is_blocked)
            *backend* is ``None`` when the tool should run unsandboxed.
            *is_blocked* is ``True`` when *tool_name* is not in the map
            at all (the call should be rejected).
        """
        if tool_name not in self.tool_sandbox_map:
            return None, True  # blocked

        return self.tool_sandbox_map[tool_name], False

    # ------------------------------------------------------------------
    # Middleware hook
    # ------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[..., Any],
    ) -> ToolMessage | Command:
        """Intercept tool calls and sandbox if configured."""
        tool_name = request.tool_call["name"]
        tool_call_id = str(request.tool_call["id"])

        backend, is_blocked = self._get_sandbox_backend(tool_name)

        # ----- Blocked tool (not in the sandbox map) -----
        if is_blocked:
            return create_tool_message(
                result=f"Tool '{tool_name}' blocked: no sandbox pattern matched",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                is_error=True,
            )

        # ----- No sandbox — pass through to normal execution -----
        if backend is None:
            return await handler(request)

        # ----- Sandboxed execution -----
        tool = request.tool
        if tool is None:
            # Cannot sandbox without a concrete tool reference.
            return await handler(request)

        tool_args = request.tool_call.get("args", {})

        # Resolve the underlying Python function metadata so the
        # sandbox worker can import and invoke it.
        underlying_func = getattr(tool, "func", None) or getattr(
            tool, "coroutine", None
        )
        module_path = (
            getattr(underlying_func, "__module__", tool.__module__)
            if underlying_func
            else tool.__module__
        )
        func_name = (
            getattr(underlying_func, "__name__", tool.name)
            if underlying_func
            else tool.name
        )

        # Serialize the runtime for the subprocess.
        # TODO: Implement full runtime serialization once the sandbox
        # worker protocol is defined.
        tool_runtime: dict[str, Any] = {}
        if request.runtime and request.runtime.context:
            ctx = request.runtime.context
            if isinstance(ctx, AgentContext):
                tool_runtime["context"] = ctx.model_dump(mode="json")

        result = await backend.execute(
            module_path=module_path,
            tool_name=func_name,
            args=tool_args,
            tool_runtime=tool_runtime,
        )

        if not result.get("success"):
            error_msg = f"Sandbox error: {result.get('error')}"
            if tb := result.get("traceback"):
                error_msg += f"\n\nTraceback:\n{tb}"
            if stderr := result.get("stderr"):
                error_msg += f"\n\nStderr:\n{stderr}"
            return create_tool_message(
                result=error_msg,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                is_error=True,
            )

        return create_tool_message(
            result=result.get("content", str(result)),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )
