"""Middleware to repair unfinished tool calls after interruptions.

When a graph is interrupted (e.g. by ``interrupt()``), the last
``AIMessage`` may have ``tool_calls`` whose ``ToolMessage`` results are
missing or out-of-order.  This middleware runs *before* the agent loop
and:

1.  Injects ``"Interrupted."`` error ``ToolMessage``s for any missing
    results.
2.  Reorders existing ``ToolMessage``s to match the ``tool_calls``
    declaration order.
3.  Moves non-tool messages (e.g. ``HumanMessage`` from the interrupt
    resume) to after the tool results so the message list is valid for
    the next LLM call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.utils.messages import create_tool_message

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class PendingToolResultMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Inject error ``ToolMessage``s for tool calls that never returned."""

    state_schema = AgentState

    async def abefore_agent(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        # -------------------------------------------------------------------
        # Find the last AIMessage — the only one whose tool_calls matter.
        # -------------------------------------------------------------------

        last_ai_index = None
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], AIMessage):
                last_ai_index = idx
                break

        if last_ai_index is None:
            return None

        last_ai = messages[last_ai_index]
        tool_calls = getattr(last_ai, "tool_calls", None) or []
        if not tool_calls:
            return None

        expected_tool_call_ids = {
            call.get("id") for call in tool_calls if call.get("id")
        }
        if not expected_tool_call_ids:
            return None

        # -------------------------------------------------------------------
        # Gather existing ToolMessages that correspond to these calls.
        # -------------------------------------------------------------------

        existing_results: dict[str, tuple[int, ToolMessage]] = {}
        for idx in range(last_ai_index + 1, len(messages)):
            msg = messages[idx]
            if (
                isinstance(msg, ToolMessage)
                and msg.tool_call_id in expected_tool_call_ids
            ):
                existing_results[msg.tool_call_id] = (idx, msg)

        missing_call_ids = expected_tool_call_ids - existing_results.keys()

        # -------------------------------------------------------------------
        # Build placeholder messages for calls that never returned.
        # -------------------------------------------------------------------

        injected_lookup = {
            call["id"]: create_tool_message(
                result="Interrupted.",
                tool_name=call.get("name") or "unknown_tool",
                tool_call_id=call["id"],
                is_error=True,
            )
            for call in tool_calls
            if call.get("id") in missing_call_ids
        }

        if not injected_lookup and not existing_results:
            return None

        # -------------------------------------------------------------------
        # Detect whether existing results are out-of-order (e.g. a
        # HumanMessage sits between the AIMessage and its ToolMessages).
        # -------------------------------------------------------------------

        needs_repair = any(
            any(
                not isinstance(messages[check_idx], (AIMessage, ToolMessage))
                for check_idx in range(last_ai_index + 1, idx)
            )
            for idx, _ in existing_results.values()
        )

        if not injected_lookup and not needs_repair:
            return None

        # -------------------------------------------------------------------
        # Rebuild the message list with correct ordering:
        #   [… prior messages …, AIMessage, ToolMessage(s), rest …]
        # -------------------------------------------------------------------

        repaired = list(messages[: last_ai_index + 1])

        # Tool messages in the same order as tool_calls.
        for call in tool_calls:
            call_id = call.get("id")
            if call_id in existing_results:
                repaired.append(existing_results[call_id][1])
            elif call_id in injected_lookup:
                repaired.append(injected_lookup[call_id])

        # Append remaining non-tool messages (e.g. HumanMessage from resume).
        existing_result_indices = {idx for idx, _ in existing_results.values()}
        repaired.extend(
            messages[idx]
            for idx in range(last_ai_index + 1, len(messages))
            if idx not in existing_result_indices
        )

        logger.debug(
            "Repaired tool results: %d moved, %d interrupted",
            len(existing_results),
            len(injected_lookup),
        )
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *repaired]}
