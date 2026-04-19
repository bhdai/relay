"""Middleware for the relay agent.

Middleware hooks into the agent lifecycle at specific points to add
cross-cutting behaviour (cost tracking, interrupt repair, output
compression, short-circuit on ``return_direct``, permission gating).
"""

from relay.middlewares.approval import ApprovalMiddleware
from relay.middlewares.compress_tool_output import CompressToolOutputMiddleware
from relay.middlewares.dynamic_prompt import create_dynamic_prompt_middleware
from relay.middlewares.pending_tool_result import PendingToolResultMiddleware
from relay.middlewares.permission import PermissionMiddleware
from relay.middlewares.return_direct import ReturnDirectMiddleware
from relay.middlewares.sandbox import SandboxMiddleware
from relay.middlewares.token_cost import TokenCostMiddleware

__all__ = [
    "ApprovalMiddleware",
    "CompressToolOutputMiddleware",
    "create_dynamic_prompt_middleware",
    "PendingToolResultMiddleware",
    "PermissionMiddleware",
    "ReturnDirectMiddleware",
    "SandboxMiddleware",
    "TokenCostMiddleware",
]
