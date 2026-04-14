"""Middleware for tool approval flow.

Before each tool invocation the middleware checks:

1. Tool-level metadata (``always_approve``, ``format_args_fn``, etc.).
2. The persistent ``ToolApprovalConfig`` (allow/deny/ask rule lists).
3. The session ``ApprovalMode`` on the ``AgentContext``.

If none of the above yields a decision the middleware raises a
LangGraph ``interrupt`` so the CLI can prompt the user.  The user's
choice (allow / always-allow / deny / always-deny) is optionally
persisted back to the JSON config file.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.configs.approval import ApprovalMode, ToolApprovalConfig, ToolApprovalRule
from relay.utils.messages import create_tool_message

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

ALLOW = "allow"
ALWAYS_ALLOW = "always allow"
DENY = "deny"
ALWAYS_DENY = "always deny"

# Relative path inside the project-local config directory.
CONFIG_APPROVAL_FILE_NAME = Path(".relay/config.approval.json")


class InterruptPayload(BaseModel):
    """Payload sent with a LangGraph ``interrupt`` for approval prompts."""

    question: str
    options: list[str]


# ==============================================================================
# Middleware
# ==============================================================================


class ApprovalMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Intercept tool calls for user approval.

    Checks approval rules and mode, interrupts for user confirmation
    if needed, and persists always-allow / always-deny decisions.
    """

    def __init__(self) -> None:
        super().__init__()
        # Cache: tool_call_id → (user_response, tool_message)
        self._decision_cache: dict[str, tuple[str, ToolMessage]] = {}

    def clear_cache(self) -> None:
        """Clear the decision cache (useful at the start of a new turn)."""
        self._decision_cache.clear()

    # ------------------------------------------------------------------
    # Rule checking
    # ------------------------------------------------------------------

    @staticmethod
    def _check_approval_rules(
        config: ToolApprovalConfig,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> tuple[bool | None, bool]:
        """Check persistent rules for an automatic decision.

        Returns
        -------
        (decision, is_always_ask)
            *decision* is ``True`` (allow), ``False`` (deny), or
            ``None`` (prompt the user).  *is_always_ask* is ``True``
            when matched by an ``always_ask`` rule.
        """
        # Deny takes highest priority.
        for rule in config.always_deny:
            if rule.matches_call(tool_name, tool_args):
                return False, False

        for rule in config.always_allow:
            if rule.matches_call(tool_name, tool_args):
                return True, False

        for rule in config.always_ask:
            if rule.matches_call(tool_name, tool_args):
                return None, True

        return None, False

    @staticmethod
    def _check_approval_mode_bypass(
        approval_mode: ApprovalMode,
        config: ToolApprovalConfig,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> bool:
        """Return ``True`` if *approval_mode* allows skipping the prompt.

        - ``SEMI_ACTIVE``: never bypass.
        - ``ACTIVE``: bypass unless in ``always_deny`` or ``always_ask``.
        - ``AGGRESSIVE``: bypass unless in ``always_deny``.
        """
        if approval_mode == ApprovalMode.SEMI_ACTIVE:
            return False
        elif approval_mode == ApprovalMode.ACTIVE:
            for rule in config.always_deny:
                if rule.matches_call(tool_name, tool_args):
                    return False
            for rule in config.always_ask:
                if rule.matches_call(tool_name, tool_args):
                    return False
            return True
        elif approval_mode == ApprovalMode.AGGRESSIVE:
            for rule in config.always_deny:
                if rule.matches_call(tool_name, tool_args):
                    return False
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _save_approval_decision(
        config: ToolApprovalConfig,
        config_file: Path,
        tool_name: str,
        tool_args: dict[str, Any] | None,
        allow: bool,
        from_always_ask: bool = False,
    ) -> None:
        """Persist an always-allow / always-deny decision to the config file."""
        rule = ToolApprovalRule(name=tool_name, args=tool_args)

        # Remove existing entries for this tool+args from both lists.
        config.always_allow = [
            r
            for r in config.always_allow
            if not (r.name == tool_name and r.args == tool_args)
        ]
        config.always_deny = [
            r
            for r in config.always_deny
            if not (r.name == tool_name and r.args == tool_args)
        ]

        # Remove from always_ask only on permanent decisions from always_ask matches.
        if from_always_ask:
            config.always_ask = [
                r
                for r in config.always_ask
                if not (r.name == tool_name and r.args == tool_args)
            ]

        if allow:
            config.always_allow.append(rule)
            logger.info("Added '%s' to always-allow list", tool_name)
        else:
            config.always_deny.append(rule)
            logger.info("Added '%s' to always-deny list", tool_name)

        config.save_to_json_file(config_file)

    # ------------------------------------------------------------------
    # Core approval logic
    # ------------------------------------------------------------------

    def _handle_approval(self, request: ToolCallRequest) -> str:
        """Evaluate rules → mode bypass → interrupt for user input.

        Returns one of :data:`ALLOW`, :data:`ALWAYS_ALLOW`,
        :data:`DENY`, or :data:`ALWAYS_DENY`.
        """
        context = request.runtime.context
        assert isinstance(context, AgentContext), (
            "runtime context must be AgentContext"
        )

        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})

        # ----- Tool-level metadata overrides -----
        format_args_fn: Callable[..., Any] | None = None
        render_args_fn: Callable[..., Any] | None = None
        name_only = False
        always_approve = False

        if request.tool:
            tool_metadata = request.tool.metadata or {}
            tool_config = tool_metadata.get("approval_config", {})
            format_args_fn = tool_config.get("format_args_fn")
            render_args_fn = tool_config.get("render_args_fn")
            name_only = tool_config.get("name_only", False)
            always_approve = tool_config.get("always_approve", False)

        if always_approve:
            return ALLOW

        # ----- Load persistent config -----
        config_file = Path(context.working_dir) / CONFIG_APPROVAL_FILE_NAME
        approval_config = ToolApprovalConfig.from_json_file(config_file)

        formatted_args = format_args_fn(tool_args) if format_args_fn else tool_args

        # ----- Mode bypass check -----
        if self._check_approval_mode_bypass(
            context.approval_mode, approval_config, tool_name, formatted_args
        ):
            return ALLOW

        # ----- Rule-based decision -----
        approval_decision, is_always_ask = self._check_approval_rules(
            approval_config, tool_name, formatted_args
        )

        if approval_decision is True:
            return ALLOW
        elif approval_decision is False:
            return DENY

        # ----- Interrupt for user prompt -----
        question = f"Allow running {tool_name} ?"
        if render_args_fn:
            rendered_config = {"configurable": {"working_dir": context.working_dir}}
            rendered = render_args_fn(tool_args, rendered_config)
            question += f" : {rendered}"
        elif not name_only:
            question += f" : {tool_args}"

        interrupt_payload = InterruptPayload(
            question=question,
            options=[ALLOW, ALWAYS_ALLOW, DENY, ALWAYS_DENY],
        )
        user_response = interrupt(interrupt_payload)

        # ----- Persist "always" decisions -----
        args_to_save = None if name_only else formatted_args

        if user_response == ALWAYS_ALLOW:
            self._save_approval_decision(
                approval_config,
                config_file,
                tool_name,
                args_to_save,
                allow=True,
                from_always_ask=is_always_ask,
            )
        elif user_response == ALWAYS_DENY:
            self._save_approval_decision(
                approval_config,
                config_file,
                tool_name,
                args_to_save,
                allow=False,
                from_always_ask=is_always_ask,
            )

        return user_response

    # ------------------------------------------------------------------
    # Middleware hook
    # ------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[..., Any],
    ) -> ToolMessage | Command:
        """Intercept tool calls for approval before execution."""
        try:
            tool_call_id = str(request.tool_call["id"])
            tool_name = request.tool_call["name"]

            # Return cached result if we already processed this tool_call_id.
            if tool_call_id in self._decision_cache:
                _cached_response, cached_message = self._decision_cache[tool_call_id]
                return cached_message

            user_response = self._handle_approval(request)

            if user_response in (ALLOW, ALWAYS_ALLOW):
                result = await handler(request)
                if isinstance(result, Command):
                    return result

                # Handler returns a ToolMessage — pass it through directly.
                if isinstance(result, ToolMessage):
                    self._decision_cache[tool_call_id] = (user_response, result)
                    return result

                tool_msg = create_tool_message(
                    result=str(result),
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
                self._decision_cache[tool_call_id] = (user_response, tool_msg)
                return tool_msg
            else:
                tool_msg = create_tool_message(
                    result="Action denied by user.",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    is_error=True,
                    return_direct=True,
                )
                self._decision_cache[tool_call_id] = (user_response, tool_msg)
                return tool_msg
        except GraphInterrupt:
            raise
        except Exception as exc:
            return create_tool_message(
                result=f"Failed to execute tool: {exc}",
                tool_name=request.tool_call["name"],
                tool_call_id=str(request.tool_call["id"]),
                is_error=True,
            )


# ==============================================================================
# Arg Formatting Helpers
# ==============================================================================
#
# These factory functions produce callables suitable for
# ``tool.metadata["approval_config"]["format_args_fn"]``.  They
# transform raw tool arguments into a simpler form for rule matching
# (e.g. extracting just the base command from a shell invocation).


def create_field_extractor(
    field_patterns: dict[str, str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a function that extracts regex named-groups from argument fields.

    Example::

        extractor = create_field_extractor({
            "command": r"(?P<command>\\S+)",  # first word only
        })
    """

    def _extract(args: dict[str, Any]) -> dict[str, Any]:
        result = args.copy()
        for field, pattern in field_patterns.items():
            if field in args:
                match = re.search(pattern, str(args[field]))
                if match:
                    result.update(match.groupdict())
        return result

    return _extract


def create_field_transformer(
    field_transforms: dict[str, Callable[[str], str]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a function that applies transform callables to argument fields.

    Example::

        transformer = create_field_transformer({
            "command": lambda x: x.split()[0],  # first word only
        })
    """

    def _transform(args: dict[str, Any]) -> dict[str, Any]:
        result = args.copy()
        for field, func in field_transforms.items():
            if field in args:
                try:
                    result[field] = func(str(args[field]))
                except Exception:
                    pass  # Keep original value on failure.
        return result

    return _transform
