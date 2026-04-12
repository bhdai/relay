"""Middleware for tracking token usage and calculating costs.

Extracts ``usage_metadata`` from the latest ``AIMessage`` after each
model call and writes per-call token counts plus an incremental cost
to the agent state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from relay.agents.context import AgentContext
from relay.agents.state import AgentState

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class TokenCostMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Extract token usage from model responses and calculate cost.

    Updates state with:
    - ``current_input_tokens``: input tokens for this call
    - ``current_output_tokens``: output tokens for this call
    - ``total_cost``: incremental cost (accumulated via ``sum_reducer``)
    """

    state_schema = AgentState

    async def aafter_model(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        latest_message = messages[-1]
        if not isinstance(latest_message, AIMessage):
            return None

        usage_metadata = getattr(latest_message, "usage_metadata", None)
        if not usage_metadata:
            return None

        input_tokens = usage_metadata.get("input_tokens", 0)
        output_tokens = usage_metadata.get("output_tokens", 0)

        update: dict[str, Any] = {
            "current_input_tokens": input_tokens,
            "current_output_tokens": output_tokens,
        }

        context: AgentContext = runtime.context
        call_cost = (
            input_tokens / 1_000_000 * context.input_cost_per_mtok
            + output_tokens / 1_000_000 * context.output_cost_per_mtok
        )
        update["total_cost"] = call_cost

        return update
