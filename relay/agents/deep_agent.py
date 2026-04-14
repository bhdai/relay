"""Coordinator + subagent agent construction.

A "deep agent" is a ReAct agent augmented with a ``task`` delegation
tool that dispatches work to named subagents.  This module owns the
assembly of the coordinator tool surface and the ``task`` tool wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from relay.agents.react_agent import create_react_agent
from relay.tools.subagents.task import SubAgentConfig, create_task_tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore

    from relay.agents.context import AgentContext
    from relay.agents.state import AgentState


def create_deep_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    prompt: str,
    subagent_configs: list[SubAgentConfig] | None = None,
    state_schema: type[AgentState] | None = None,
    context_schema: type[AgentContext] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    name: str | None = None,
) -> CompiledStateGraph:
    """Create a coordinator agent that can delegate to subagents.

    If *subagent_configs* is provided, a ``task`` tool is built and
    appended to *tools*.  Otherwise, this is equivalent to calling
    ``create_react_agent`` directly.
    """

    all_tools = list(tools)
    if subagent_configs:
        task_tool = create_task_tool(
            subagent_configs=subagent_configs,
            llm=model,
        )
        all_tools.append(task_tool)

    return create_react_agent(
        model,
        tools=all_tools,
        prompt=prompt,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        name=name,
    )
