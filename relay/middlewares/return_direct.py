"""Middleware for handling ``return_direct`` behaviour in tools.

When a ``ToolMessage`` has ``return_direct=True`` (e.g. a denied action),
this middleware short-circuits the agent loop by jumping to ``"end"``
before the model is called again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import ToolMessage

from relay.agents.context import AgentContext
from relay.agents.state import AgentState

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class ReturnDirectMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Jump to ``"end"`` when the most recent tool messages include one
    with ``return_direct=True``.

    The scan walks backwards from the end of the message list and stops
    at the first non-``ToolMessage``, so only the latest batch of tool
    results is inspected.
    """

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: AgentState, runtime: Runtime[AgentContext]
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                if getattr(msg, "return_direct", False):
                    return {"jump_to": "end"}
            elif not isinstance(msg, ToolMessage):
                break

        return None
