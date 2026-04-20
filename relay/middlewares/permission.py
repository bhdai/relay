"""PermissionMiddleware — unified permission gate for all tool calls.

Replaces ``ApprovalMiddleware`` with a ruleset-driven, interrupt-based
permission system that mirrors Opencode's permission model.

Overview:
    1. Each tool call enters ``awrap_tool_call``.
    2. A ``PermissionRequest`` is built from the tool's
       ``permission_config`` metadata, or from sensible defaults.
    3. The request is evaluated by the session-scoped ``PermissionService``
       against the agent-config ruleset plus accumulated "always" rules.
    4. ``Allowed`` executes the tool immediately.
    5. ``Denied`` returns an error ``ToolMessage`` with ``return_direct``.
    6. ``NeedsAsk`` raises a LangGraph ``interrupt(payload)`` and feeds the
       user's ``Reply`` back into the service on resume.

Tool Metadata:
    Each tool may expose a ``"permission_config"`` key in ``.metadata``.

    ``"permission"``:
        Permission key string, for example ``"bash"``, ``"edit"``, or
        ``"read"``. Falls back to the tool's own name when absent.
    ``"patterns_fn"``:
        ``callable(args: dict) -> list[str]`` returning the concrete values
        being evaluated, such as a command string or file path. Defaults to
        ``["*"]``.
    ``"always_fn"``:
        ``callable(args: dict) -> list[str]`` returning the coarser patterns
        persisted when the user replies ``"always"``. Defaults to ``["*"]``.
    ``"metadata_fn"``:
        ``callable(args: dict) -> dict`` returning display context for the
        approval prompt, such as ``{"command": "git push --force"}``.
        Defaults to ``{}``.
    ``"is_catalog_proxy"``:
        When ``True``, the middleware looks through the proxy tool, such as
        ``run_tool``, to the underlying tool's ``permission_config``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.permission.schema import (
    Allowed,
    Denied,
    NeedsAsk,
    PermissionRequest,
    PermissionRule,
    Reply,
    Ruleset,
)
from relay.permission.service import PermissionService
from relay.utils.messages import create_tool_message

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)


# ==============================================================================
# Interrupt payload
# ==============================================================================


class PermissionInterruptPayload(BaseModel):
    """Payload transmitted with a LangGraph ``interrupt`` for permission prompts.

    The CLI reads this payload to render the approval prompt and sends back a
    ``Reply`` (``"once"``, ``"always"``, or ``"reject"``).
    """

    request_id: str
    """ID of the pending ``PermissionRequest`` in the service."""

    question: str
    """Human-readable prompt line shown to the user."""

    permission: str
    """Permission key being requested, e.g. ``"bash"``, ``"edit"``."""

    patterns: list[str]
    """Concrete patterns being evaluated, e.g. ``["git push --force"]``."""

    always_patterns: list[str]
    """Patterns that would be persisted when the user replies ``"always"``."""

    metadata: dict[str, Any] = {}
    """Tool-specific context (command text, file diff, etc.) for display."""

    options: list[str] = ["once", "always", "reject"]
    """Available reply choices shown in the prompt UI."""


# ==============================================================================
# PermissionMiddleware
# ==============================================================================


class PermissionMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Gate every tool call through the agent's permission ruleset.

    At graph-build time the middleware receives the agent's resolved
    ``Ruleset`` (``DEFAULT_PERMISSION`` merged with YAML config overrides).
    At runtime, per-session ``PermissionService`` instances are created
    lazily, keyed by the LangGraph ``thread_id`` extracted from the
    execution config.

    Accumulated "always" rules are written back into
    ``AgentContext.permission_ruleset`` after each ``"always"`` reply so
    they survive graph checkpointing across turns.
    """

    def __init__(self, ruleset: Ruleset) -> None:
        super().__init__()
        # Fixed agent-config-level ruleset, resolved at graph-build time.
        self._ruleset: Ruleset = ruleset

        # Per-session services: thread_id → PermissionService.
        # The compiled graph is typically reused across sessions, so session
        # isolation requires explicit keying by thread_id.
        self._services: dict[str, PermissionService] = {}

    # ------------------------------------------------------------------
    # Session-scoped service management
    # ------------------------------------------------------------------

    def _session_id(self, request: ToolCallRequest) -> str:
        """Extract the LangGraph ``thread_id`` from the runtime config.

        Falls back to ``"default"`` when no thread_id is present (e.g.
        in unit tests or stateless invocations).
        """
        runtime = request.runtime
        if runtime is None:
            return "default"
        config = getattr(runtime, "config", None)
        if isinstance(config, dict):
            return config.get("configurable", {}).get("thread_id", "default")
        return "default"

    def _get_service(
        self,
        session_id: str,
        context: AgentContext,
    ) -> PermissionService:
        """Return the session-scoped ``PermissionService``, creating it if needed.

        On first access for a session the service is seeded with any previously
        persisted "always" rules from ``context.permission_ruleset``.  These
        may include rules approved in earlier turns of the same session.
        """
        if session_id not in self._services:
            # Reconstruct persisted approved rules from the serialisable form
            # stored on AgentContext.  Pydantic validation on PermissionRule
            # will reject malformed entries early.
            initial_approved: Ruleset = [
                PermissionRule(**r) for r in context.permission_ruleset
            ]
            self._services[session_id] = PermissionService(
                initial_approved=initial_approved,
            )
        return self._services[session_id]

    # ------------------------------------------------------------------
    # Permission request construction
    # ------------------------------------------------------------------

    def _build_permission_request(
        self,
        request: ToolCallRequest,
        session_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        context: AgentContext,
    ) -> PermissionRequest:
        """Build a ``PermissionRequest`` from the tool's ``permission_config`` metadata.

        The method resolves the permission config in three steps:

        1. Read ``metadata["permission_config"]`` from the tool (if present).
        2. If ``"is_catalog_proxy"`` is set, look through to the underlying
           tool's config (used by the ``run_tool`` catalog proxy).
        3. Call the callable fields (``patterns_fn``, ``always_fn``,
           ``metadata_fn``) with the tool arguments to build the request.

        Any missing config field falls back to a safe default:
        - ``permission`` → tool name
        - ``patterns`` → ``["*"]``
        - ``always`` → ``["*"]``
        - ``metadata`` → ``{}``
        """
        perm_cfg: dict[str, Any] = {}
        if request.tool:
            tool_meta = request.tool.metadata or {}
            perm_cfg = tool_meta.get("permission_config", {})

        # Catalog-proxy look-through: run_tool wraps an underlying catalog tool.
        # Enforce approval using the underlying tool's permission config so that
        # the proxy transparently inherits the underlying tool's semantics.
        if perm_cfg.get("is_catalog_proxy"):
            underlying_tool_name = tool_args.get("tool_name")
            raw_underlying_args = tool_args.get("tool_args", {})
            underlying_tool_args: dict[str, Any] = (
                raw_underlying_args if isinstance(raw_underlying_args, dict) else {}
            )

            if isinstance(underlying_tool_name, str) and underlying_tool_name:
                underlying_tool = next(
                    (t for t in context.tool_catalog if t.name == underlying_tool_name),
                    None,
                )
                if underlying_tool is not None:
                    # Redirect evaluation to the underlying tool and its args.
                    tool_name = underlying_tool_name
                    tool_args = underlying_tool_args
                    tool_meta = underlying_tool.metadata or {}
                    perm_cfg = tool_meta.get("permission_config", {})

        # Extract config fields (all optional, with safe fallbacks).
        permission_key: str = perm_cfg.get("permission", tool_name)

        patterns_fn: Callable[[dict[str, Any]], list[str]] | None = perm_cfg.get(
            "patterns_fn"
        )
        always_fn: Callable[[dict[str, Any]], list[str]] | None = perm_cfg.get(
            "always_fn"
        )
        metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = perm_cfg.get(
            "metadata_fn"
        )

        patterns: list[str] = (
            patterns_fn(tool_args) if callable(patterns_fn) else ["*"]
        )
        always: list[str] = always_fn(tool_args) if callable(always_fn) else ["*"]
        meta: dict[str, Any] = (
            metadata_fn(tool_args) if callable(metadata_fn) else {}
        )

        return PermissionRequest(
            session_id=session_id,
            permission=permission_key,
            patterns=patterns,
            always=always,
            metadata=meta,
            tool_call_id=str(request.tool_call["id"]),
        )

    # ------------------------------------------------------------------
    # Execution helper
    # ------------------------------------------------------------------

    async def _execute(
        self,
        request: ToolCallRequest,
        handler: Callable[..., Any],
        tool_name: str,
        tool_call_id: str,
    ) -> ToolMessage | Command:
        """Invoke the tool handler and normalise the result to a ``ToolMessage``."""
        result = await handler(request)
        if isinstance(result, (Command, ToolMessage)):
            return result
        return create_tool_message(
            result=str(result),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_question(tool_name: str, perm_request: PermissionRequest) -> str:
        """Build a human-readable single-line permission prompt."""
        patterns_str = ", ".join(perm_request.patterns)
        return f"Allow {tool_name} [{perm_request.permission}]: {patterns_str}"

    # ------------------------------------------------------------------
    # Middleware hook
    # ------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[..., Any],
    ) -> ToolMessage | Command:
        """Intercept each tool call and enforce the permission ruleset.

        Decision outcomes:

        ``Allowed``
            Execute the tool immediately without prompting.
        ``Denied``
            Return a ``return_direct`` error ``ToolMessage`` without
            executing the tool.
        ``NeedsAsk``
            Raise a LangGraph ``interrupt`` with a
            ``PermissionInterruptPayload``.  The graph resumes with the
            user's ``Reply`` (``"once"``, ``"always"``, or ``"reject"``).
            - ``"once"``   → execute the tool; no rules persisted.
            - ``"always"`` → execute the tool; approved patterns written
                             back to ``AgentContext.permission_ruleset``.
            - ``"reject"`` → return ``return_direct`` error without
                             executing; all pending same-session requests
                             are also rejected (fan-out).
        """
        try:
            tool_call_id = str(request.tool_call["id"])
            tool_name = request.tool_call["name"]
            tool_args = request.tool_call.get("args", {})

            context = request.runtime.context
            assert isinstance(context, AgentContext), (
                "runtime context must be AgentContext"
            )

            session_id = self._session_id(request)
            service = self._get_service(session_id, context)

            perm_request = self._build_permission_request(
                request, session_id, tool_name, tool_args, context
            )
            decision = service.ask(perm_request, self._ruleset)

            # --- Denied: return an error immediately without executing. ---
            if isinstance(decision, Denied):
                logger.debug(
                    "Permission denied for tool %r: %s", tool_name, decision.reason
                )
                return create_tool_message(
                    result=decision.reason,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    is_error=True,
                    return_direct=True,
                )

            # --- Allowed: execute without prompting. ---
            if isinstance(decision, Allowed):
                logger.debug("Permission allowed for tool %r", tool_name)
                return await self._execute(request, handler, tool_name, tool_call_id)

            # --- NeedsAsk: suspend the graph for user input. ---
            assert isinstance(decision, NeedsAsk)
            perm_req = decision.request

            payload = PermissionInterruptPayload(
                request_id=perm_req.id,
                question=self._format_question(tool_name, perm_req),
                permission=perm_req.permission,
                patterns=perm_req.patterns,
                always_patterns=perm_req.always,
                metadata=perm_req.metadata,
            )

            # Suspend; resume with the user's Reply.
            user_reply: Reply = interrupt(payload)

            # Process the reply and persist "always" rules to the context so
            # they survive graph checkpointing.
            service.reply(perm_req.id, user_reply)
            context.permission_ruleset = [r.model_dump() for r in service.approved]

            if user_reply == "reject":
                logger.debug(
                    "Permission rejected by user for tool %r", tool_name
                )
                return create_tool_message(
                    result="Action rejected by user.",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    is_error=True,
                    return_direct=True,
                )

            # "once" or "always" → execute the tool.
            logger.debug(
                "Permission granted (%r) for tool %r", user_reply, tool_name
            )
            return await self._execute(request, handler, tool_name, tool_call_id)

        except GraphInterrupt:
            raise
        except Exception as exc:
            return create_tool_message(
                result=f"Failed to execute tool: {exc}",
                tool_name=request.tool_call["name"],
                tool_call_id=str(request.tool_call["id"]),
                is_error=True,
            )
