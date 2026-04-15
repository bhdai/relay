"""Task delegation tool for coordinator-to-subagent handoff.

The ``think`` tool lives here alongside ``task`` because it is only
used inside subagent contexts — every subagent gets ``think`` appended
automatically by ``create_task_tool``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict

from relay.agents.context import AgentContext
from relay.agents.react_agent import create_react_agent
from relay.agents.state import AgentState
from relay.utils.messages import create_tool_message

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# ==============================================================================
# Subagent configuration
# ==============================================================================


class SubAgentRuntime(BaseModel):
    """Runtime configuration for a delegated subagent.

    This is the *resolved* runtime object with actual tool instances and
    prompt text.  The declarative counterpart lives in
    ``relay.configs.agent.SubAgentConfig``.
    """

    name: str
    description: str
    tools: list[BaseTool]
    prompt: str
    recursion_limit: int = 25

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ==============================================================================
# Think tool — strategic reflection for subagents
# ==============================================================================


@tool(return_direct=True)
def think(reflection: str) -> str:
    """Tool for strategic reflection on progress and decision-making.

    Always use this tool after each search to analyze results and plan
    next steps systematically.  This creates a deliberate pause in the
    workflow for quality decision-making.

    When to use:

    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing gaps: What specific information am I still missing?
    - Before concluding: Can I provide a complete answer now?

    Reflection should address:

    1. Analysis of current findings — what concrete information have I gathered?
    2. Gap assessment — what crucial information is still missing?
    3. Quality evaluation — do I have sufficient evidence/examples?
    4. Strategic decision — should I continue searching or provide my answer?
    """
    return f"Reflection recorded: {reflection}"


think.metadata = {"approval_config": {"always_approve": True}}


# ==============================================================================
# Message rendering
# ==============================================================================


def _render_message_content(message: AnyMessage | None) -> str:
    """Extract readable text from the last subagent message."""
    if message is None:
        return "Subagent completed without returning a final message."

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content or "Subagent completed."

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    return str(content) if content else "Subagent completed."


# ==============================================================================
# Task tool
# ==============================================================================


def create_task_tool(
    subagent_configs: list[SubAgentRuntime],
    llm: BaseChatModel,
):
    """Create a ``task`` tool that delegates work to named subagents.

    Each subagent is lazily compiled on first use via
    ``create_react_agent``.  The ``think`` tool is appended to every
    subagent's tool surface automatically.
    """

    agents: dict[str, CompiledStateGraph] = {}
    config_by_name = {config.name: config for config in subagent_configs}
    descriptions = "\n".join(
        f"- {config.name}: {config.description}" for config in subagent_configs
    )

    @tool(
        description=(
            "Delegate a task to a specialized sub-agent with isolated context. "
            f"Available agents:\n{descriptions}"
        )
    )
    async def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime[AgentContext, AgentState],
    ):
        if subagent_type not in config_by_name:
            allowed = ", ".join(sorted(config_by_name))
            raise ToolException(
                f"Unknown subagent '{subagent_type}'. Allowed values: {allowed}"
            )

        config = config_by_name[subagent_type]

        # Lazily compile the subagent graph on first delegation.
        if subagent_type not in agents:
            agents[subagent_type] = create_react_agent(
                llm,
                tools=[*config.tools, think],
                prompt=config.prompt,
                state_schema=AgentState,
                context_schema=AgentContext,
                name=config.name,
            )

        subagent = agents[subagent_type]

        state = dict(runtime.state)
        state["messages"] = [HumanMessage(content=description)]

        try:
            result = await subagent.ainvoke(
                state,
                context=(
                    runtime.context.model_copy(deep=True)
                    if runtime.context is not None
                    else AgentContext()
                ),
                config={"recursion_limit": config.recursion_limit},
            )
        except Exception as exc:
            raise ToolException(
                f"delegate task to subagent '{subagent_type}'"
            ) from exc

        messages = result.get("messages") or []
        last_message = messages[-1] if messages else None
        is_error = isinstance(last_message, ToolMessage) and (
            getattr(last_message, "is_error", False)
            or getattr(last_message, "status", None) == "error"
        )

        update: dict[str, Any] = {
            "messages": [
                create_tool_message(
                    result=_render_message_content(last_message),
                    tool_name=task.name,
                    tool_call_id=runtime.tool_call_id or "",
                    is_error=is_error,
                )
            ]
        }
        if "files" in result:
            update["files"] = result["files"]
        if "todos" in result:
            update["todos"] = result["todos"]

        return Command(update=update)

    task.metadata = {"approval_config": {"always_approve": True}}
    return task