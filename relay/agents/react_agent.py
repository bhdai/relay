"""Standard ReAct agent construction with middleware ordering.

This is the canonical place for middleware stack assembly.  All relay
agents (coordinator, subagents) should be built through
``create_react_agent`` so that lifecycle hooks are applied consistently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent

from relay.middlewares import (
    CompressToolOutputMiddleware,
    PendingToolResultMiddleware,
    ReturnDirectMiddleware,
    TokenCostMiddleware,
    create_dynamic_prompt_middleware,
)

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

    from relay.agents.context import AgentContext
    from relay.agents.state import AgentState


def create_react_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    prompt: str,
    state_schema: type[AgentState] | None = None,
    context_schema: type[AgentContext] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    name: str | None = None,
):
    """Create a ReAct agent with relay's standard middleware stack.

    Middleware execution order:

    - ``before_*`` hooks: first to last
    - ``after_*`` hooks: last to first (reverse)
    - ``wrap_*`` hooks: nested (first middleware wraps all others)
    """

    # Group 0: Dynamic prompt — render template with runtime context.
    dynamic_prompt: list[AgentMiddleware[Any, Any]] = [
        create_dynamic_prompt_middleware(prompt),
    ]

    # Group 1: afterModel — after each model response.
    after_model: list[AgentMiddleware[Any, Any]] = [
        TokenCostMiddleware(),
    ]

    # Group 2: beforeAgent — before each agent invocation.
    before_agent: list[AgentMiddleware[Any, Any]] = [
        PendingToolResultMiddleware(),
    ]

    # Group 3: beforeModel — before each model call.
    before_model: list[AgentMiddleware[Any, Any]] = [
        ReturnDirectMiddleware(),
    ]

    # Group 4: wrapToolCall — wraps tool execution.
    # CompressToolOutputMiddleware intercepts large tool outputs and
    # stores them in virtual files to avoid token budget overruns.
    wrap_tool_call: list[AgentMiddleware[Any, Any]] = [
        CompressToolOutputMiddleware(model),
    ]

    middlewares: list[AgentMiddleware[Any, Any]] = (
        dynamic_prompt + after_model + before_agent + before_model + wrap_tool_call
    )

    return create_agent(
        model=model,
        tools=tools,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        name=name,
        middleware=middlewares,
    )
