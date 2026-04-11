"""Reference: building the ReAct agent graph the "hard way".

This file is a **learning reference** that shows how to build the same
agent from ``relay.graph`` using the low-level LangGraph ``StateGraph``
API instead of the ``create_react_agent`` prebuilt helper.

It is *not* imported by the application — its only purpose is to make
the graph construction visible and inspectable.

Usage::

    from dotenv import load_dotenv; load_dotenv()
    from relay.graph_reference import build_graph

    graph = build_graph()
    # … use exactly like the production graph …
"""

import json
from typing import Annotated

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from relay.settings import get_settings
from relay.tools.filesystem import FILE_SYSTEM_TOOLS
from relay.tools.terminal import TERMINAL_TOOLS
from relay.tools.web import WEB_TOOLS

# ==============================================================================
# 1. Define the Agent State
# ==============================================================================
#
# The state is the data structure that flows along the graph edges. In a
# ReAct agent the only thing we need to track is the conversation message
# list.  ``add_messages`` is a LangGraph *reducer* that appends new
# messages rather than overwriting them.


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==============================================================================
# 2. Collect Tools
# ==============================================================================
#
# We aggregate every tool the agent can call and build a lookup dict so
# the tool-execution node can dispatch by name.

ALL_TOOLS = [*FILE_SYSTEM_TOOLS, *TERMINAL_TOOLS, *WEB_TOOLS]
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
# The ``agent`` node's output may contain ``tool_calls``.  If it does, the
# conditional edge routes to ``tools``; otherwise the loop ends.


def _build_agent_node(llm):
    """Return a node function that invokes the LLM with bound tools."""

    # Bind the tool schemas so the LLM knows what it can call.
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    async def agent_node(state: AgentState) -> dict:
        """Call the LLM and return the response message."""
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

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


def should_continue(state: AgentState) -> str:
    """Return ``'tools'`` if the LLM wants to call tools, else ``'end'``."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


# ==============================================================================
# 5. Assemble the Graph
# ==============================================================================


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Construct the ReAct agent graph using the low-level StateGraph API.

    This is functionally equivalent to ``relay.graph.build_graph`` but
    shows every step explicitly.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=settings.llm.openai_api_key,
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

    # After tools run, always loop back to the agent so it can inspect
    # the results and decide what to do next.
    graph.add_edge("tools", "agent")

    # --- Compile -----------------------------------------------------------------
    #
    # ``compile()`` freezes the graph definition and returns a runnable that
    # can be invoked with ``ainvoke``, ``astream``, etc.

    return graph.compile(checkpointer=checkpointer)
