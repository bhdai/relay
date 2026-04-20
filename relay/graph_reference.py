"""Reference: building the ReAct agent graph the "hard way".

This file is a **learning reference** that shows how to build the same
agent from ``relay.graph`` using the low-level LangGraph ``StateGraph``
API instead of the ``langchain.agents.create_agent`` helper.

It is *not* imported by the application — its only purpose is to make
the graph construction visible and inspectable.

The production graph in ``relay.graph`` uses the ``middleware=`` parameter
of ``create_agent`` to hook into the lifecycle.  Here we inline the
equivalent logic to show what the middleware hooks actually *do* at the
graph level:

- **TokenCostMiddleware** → post-processing after the LLM call inside
  the ``agent`` node.
- **PendingToolResultMiddleware** → pre-processing at the start of
  the ``agent`` node (before the LLM call).
- **ReturnDirectMiddleware** → routing logic that can short-circuit the
  loop to ``END`` before calling the model again.

Example:
    from dotenv import load_dotenv; load_dotenv()
    from relay.graph_reference import build_graph

    graph = build_graph()
    # ... use exactly like the production graph ...
"""

import json
import logging

from langchain_core.messages import (
    AIMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from relay.agents.state import AgentState
from relay.settings import get_settings
from relay.tools.impl.filesystem import FILE_SYSTEM_TOOLS
from relay.tools.internal.memory import MEMORY_TOOLS
from relay.tools.impl.terminal import TERMINAL_TOOLS
from relay.tools.internal.todo import TODO_TOOLS
from relay.tools.impl.web import WEB_TOOLS
from relay.utils.messages import create_tool_message

# ==============================================================================
# 1. Define the Agent State
# ==============================================================================
#
# The production code uses ``relay.agents.state.AgentState`` which extends
# ``langchain.agents.AgentState`` with custom reducers for files, todos,
# and cost tracking.  Here we re-use exactly that same schema so this
# reference graph stays functionally equivalent.


# ==============================================================================
# 2. Collect Tools
# ==============================================================================
#
# We aggregate every tool the agent can call and build a lookup dict so
# the tool-execution node can dispatch by name.

ALL_TOOLS = [*FILE_SYSTEM_TOOLS, *TERMINAL_TOOLS, *WEB_TOOLS, *MEMORY_TOOLS, *TODO_TOOLS]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ==============================================================================
# 3. Define the Graph Nodes
# ==============================================================================
#
# A ReAct loop has exactly two nodes:
#
#   agent  →  calls the LLM with the current messages
#   tools  →  executes whatever tool calls the LLM requested
#
# The production code uses middleware hooks to add cross-cutting concerns.
# Here we inline the same logic so you can see exactly where each hook
# fires within the graph.

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Inline middleware: PendingToolResultMiddleware (abefore_agent)
# ------------------------------------------------------------------------------
#
# When the graph resumes after an interrupt, tool calls from the last
# AIMessage may be missing their ToolMessage results or have them in the
# wrong position.  This function repairs the message list before the LLM
# sees it.


def _repair_pending_tool_results(state: AgentState) -> dict | None:
    """Inject error ToolMessages for tool calls that never returned."""
    messages = state.get("messages", [])
    if not messages:
        return None

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

    expected_ids = {c.get("id") for c in tool_calls if c.get("id")}
    if not expected_ids:
        return None

    existing: dict[str, tuple[int, ToolMessage]] = {}
    for idx in range(last_ai_index + 1, len(messages)):
        msg = messages[idx]
        if isinstance(msg, ToolMessage) and msg.tool_call_id in expected_ids:
            existing[msg.tool_call_id] = (idx, msg)

    missing_ids = expected_ids - existing.keys()
    injected = {
        c["id"]: create_tool_message(
            result="Interrupted.",
            tool_name=c.get("name") or "unknown_tool",
            tool_call_id=c["id"],
            is_error=True,
        )
        for c in tool_calls
        if c.get("id") in missing_ids
    }

    if not injected and not existing:
        return None

    needs_repair = any(
        any(
            not isinstance(messages[ci], (AIMessage, ToolMessage))
            for ci in range(last_ai_index + 1, idx)
        )
        for idx, _ in existing.values()
    )

    if not injected and not needs_repair:
        return None

    repaired = list(messages[: last_ai_index + 1])
    for c in tool_calls:
        cid = c.get("id")
        if cid in existing:
            repaired.append(existing[cid][1])
        elif cid in injected:
            repaired.append(injected[cid])

    existing_indices = {idx for idx, _ in existing.values()}
    repaired.extend(
        messages[idx]
        for idx in range(last_ai_index + 1, len(messages))
        if idx not in existing_indices
    )

    logger.debug(
        "Repaired tool results: %d moved, %d interrupted",
        len(existing),
        len(injected),
    )
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *repaired]}


# ------------------------------------------------------------------------------
# Inline middleware: TokenCostMiddleware (aafter_model)
# ------------------------------------------------------------------------------


def _extract_token_cost(response: AIMessage) -> dict:
    """Extract token usage and compute cost from the model response."""
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return {
        "current_input_tokens": input_tokens,
        "current_output_tokens": output_tokens,
        # Cost rates come from AgentContext at runtime; here we use 0.0
        # since the low-level graph has no context wiring yet.
        "total_cost": 0.0,
    }


def _build_agent_node(llm):
    """Return a node function that invokes the LLM with bound tools.

    This node inlines the middleware hooks:
    - **abefore_agent**: ``_repair_pending_tool_results`` fixes the
      message list before the LLM call.
    - **aafter_model**: ``_extract_token_cost`` records token usage
      after the LLM responds.
    """

    # Bind the tool schemas so the LLM knows what it can call.
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    async def agent_node(state: AgentState) -> dict:
        """Call the LLM and return the response message."""

        # --- abefore_agent: PendingToolResultMiddleware ---
        #
        # In the production graph this runs as a middleware hook before
        # the agent node.  Here we call it inline and merge any repair
        # into the state before invoking the LLM.
        repair = _repair_pending_tool_results(state)
        messages = state["messages"]
        if repair is not None:
            # The repair replaces the entire message list (via
            # RemoveMessage + re-add).  For the LLM call we use the
            # repaired list directly (skip the RemoveMessage sentinel).
            messages = [m for m in repair["messages"] if not isinstance(m, RemoveMessage)]

        response = await llm_with_tools.ainvoke(messages)

        # --- aafter_model: TokenCostMiddleware ---
        cost_update = _extract_token_cost(response)

        update: dict = {"messages": [response]}
        if repair is not None:
            # Propagate the full repair (including RemoveMessage) so
            # LangGraph actually replaces the stored messages.
            update["messages"] = repair["messages"] + [response]
        update.update(cost_update)

        return update

    return agent_node


async def tool_node(state: AgentState) -> dict:
    """Execute every tool call in the last AI message.

    Each call is dispatched to the matching tool from ``TOOL_MAP`` and a
    ``ToolMessage`` with the result (or error) is appended.
    """
    last_message: AIMessage = state["messages"][-1]
    outputs: list[ToolMessage] = []

    for call in last_message.tool_calls:
        tool = TOOL_MAP.get(call["name"])
        if tool is None:
            content = f"Unknown tool: {call['name']}"
        else:
            try:
                content = await tool.ainvoke(call["args"])
            except Exception as exc:
                content = f"Error: {exc}"

        outputs.append(
            ToolMessage(
                content=content,
                tool_call_id=call["id"],
                name=call["name"],
            )
        )

    return {"messages": outputs}


# ==============================================================================
# 4. Routing Logic
# ==============================================================================
#
# After every ``agent`` step we inspect the last message.  If the LLM
# requested tool calls we route to ``tools``; otherwise the conversation
# turn is complete and we go to ``END``.
#
# After every ``tools`` step we check whether any of the tool results
# have ``return_direct=True`` — if so we skip the next model call and
# go straight to ``END`` (inlined ReturnDirectMiddleware).


def should_continue(state: AgentState) -> str:
    """Return ``'tools'`` if the LLM wants to call tools, else ``'end'``."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def _should_return_direct(state: AgentState) -> str:
    """Check recent ToolMessages for ``return_direct=True``.

    This is the inline equivalent of ``ReturnDirectMiddleware.abefore_model``
    — it scans backwards through the recent ToolMessages and short-circuits
    the loop to ``END`` if any has ``return_direct=True``.
    """
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, ToolMessage):
            if getattr(msg, "return_direct", False):
                return "end"
        elif not isinstance(msg, ToolMessage):
            break
    return "agent"


# ==============================================================================
# 5. Assemble the Graph
# ==============================================================================


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Construct the ReAct agent graph using the low-level StateGraph API.

    This is functionally equivalent to ``relay.graph.build_graph`` but
    shows every step explicitly.
    """
    settings = get_settings()
    rl = settings.rate_limit
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=rl.requests_per_second,
        check_every_n_seconds=rl.check_every_n_seconds,
        max_bucket_size=rl.max_bucket_size,
    )
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=settings.llm.openai_api_key,
        rate_limiter=rate_limiter,
    )

    # --- Build the graph -------------------------------------------------------

    graph = StateGraph(AgentState)

    # Add nodes.
    graph.add_node("agent", _build_agent_node(llm))
    graph.add_node("tools", tool_node)

    # Set the entry point — every new turn starts at the agent.
    graph.set_entry_point("agent")

    # Conditional edge: after the agent runs, decide whether to call tools
    # or finish.
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools run, check whether any result has return_direct=True.
    # If so, skip the next LLM call and go straight to END.
    # Otherwise loop back to the agent for the next model call.
    graph.add_conditional_edges(
        "tools",
        _should_return_direct,
        {
            "agent": "agent",
            "end": END,
        },
    )

    # --- Compile -----------------------------------------------------------------
    #
    # ``compile()`` freezes the graph definition and returns a runnable that
    # can be invoked with ``ainvoke``, ``astream``, etc.

    return graph.compile(checkpointer=checkpointer)
