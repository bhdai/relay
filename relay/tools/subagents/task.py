"""Task delegation tool for coordinator-to-subagent handoff.

The ``think`` tool lives here alongside ``task`` because it is only
used inside subagent contexts — every subagent gets ``think`` appended
automatically by ``create_task_tool``.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict

from relay.agents.context import AgentContext
from relay.agents.react_agent import create_react_agent
from relay.agents.state import AgentState
from relay.configs.llm import LLMConfig
from relay.utils.messages import create_tool_message

if TYPE_CHECKING:
    from collections.abc import Callable

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
    llm_config: LLMConfig | None = None
    recursion_limit: int = 100

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


think.metadata = {"permission_config": {"permission": "think"}}


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


def _iter_update_payloads(data: Any) -> list[dict[str, Any]]:
    """Extract node payload dicts from a LangGraph ``updates`` event."""
    if not isinstance(data, dict):
        return []

    if "messages" in data:
        return [data]

    payloads: list[dict[str, Any]] = []
    for value in data.values():
        if isinstance(value, dict):
            payloads.append(value)
    return payloads


def _stream_subagent_event(
    runtime: ToolRuntime[AgentContext, AgentState],
    payload: dict[str, Any],
) -> None:
    """Forward delegated subagent activity to the parent graph stream."""
    runtime.stream_writer(deepcopy(payload))


def _unpack_subagent_event(raw_event: Any) -> tuple[tuple[str, ...], str, Any] | None:
    """Normalize a subagent stream event with optional subgraph metadata."""
    if not isinstance(raw_event, tuple):
        return None

    if len(raw_event) == 3:
        namespace, mode, data = raw_event
        if isinstance(namespace, tuple):
            return namespace, mode, data
        if isinstance(namespace, str):
            return (namespace,), mode, data
        return (), mode, data

    if len(raw_event) == 2:
        mode, data = raw_event
        return (), mode, data

    return None


# ==============================================================================
# Task tool
# ==============================================================================


def create_task_tool(
    subagent_configs: list[SubAgentRuntime],
    model_provider: Callable[[LLMConfig | None], BaseChatModel],
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
            model = model_provider(config.llm_config)
            agents[subagent_type] = create_react_agent(
                model,
                tools=[*config.tools, think],
                prompt=config.prompt,
                state_schema=AgentState,
                context_schema=AgentContext,
                name=config.name,
            )

        subagent = agents[subagent_type]

        state = dict(runtime.state)
        state["messages"] = [HumanMessage(content=description)]

        files_state = dict(state.get("files") or {})
        todos_state = state.get("todos")
        has_files = "files" in state
        has_todos = "todos" in state
        latest_messages: list[AnyMessage] | None = None

        _stream_subagent_event(
            runtime,
            {
                "relay_event": "subagent_start",
                "subagent": subagent_type,
                "description": description,
            },
        )

        try:
            async for raw_event in subagent.astream(
                state,
                context=(
                    runtime.context.model_copy(deep=True)
                    if runtime.context is not None
                    else AgentContext()
                ),
                config={"recursion_limit": config.recursion_limit},
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                event = _unpack_subagent_event(raw_event)
                if event is None:
                    continue

                child_namespace, mode, data = event

                if mode == "messages":
                    if not isinstance(data, Sequence) or len(data) != 2:
                        continue

                    msg_chunk, _metadata = data
                    if not isinstance(msg_chunk, AIMessageChunk):
                        continue

                    _stream_subagent_event(
                        runtime,
                        {
                            "relay_event": "subagent_message",
                            "subagent": subagent_type,
                            "namespace": list(child_namespace),
                            "message": msg_chunk,
                        },
                    )
                    continue

                if mode != "updates" or not isinstance(data, dict):
                    continue

                _stream_subagent_event(
                    runtime,
                    {
                        "relay_event": "subagent_update",
                        "subagent": subagent_type,
                        "namespace": list(child_namespace),
                        "update": data,
                    },
                )

                for payload in _iter_update_payloads(data):
                    if "messages" in payload:
                        latest_messages = payload["messages"]
                    if "files" in payload:
                        has_files = True
                        files_state = {**files_state, **(payload["files"] or {})}
                    if "todos" in payload:
                        has_todos = True
                        todos_state = payload["todos"]
        except GraphRecursionError as exc:
            _stream_subagent_event(
                runtime,
                {
                    "relay_event": "subagent_finish",
                    "subagent": subagent_type,
                    "status": "error",
                },
            )
            raise ToolException(
                "Subagent "
                f"'{subagent_type}' exceeded recursion limit "
                f"({config.recursion_limit}) while handling the delegated task. "
                "The task likely stayed in an exploration loop instead of reaching "
                "a final answer."
            ) from exc
        except Exception as exc:
            _stream_subagent_event(
                runtime,
                {
                    "relay_event": "subagent_finish",
                    "subagent": subagent_type,
                    "status": "error",
                },
            )
            raise ToolException(
                f"delegate task to subagent '{subagent_type}'"
            ) from exc

        _stream_subagent_event(
            runtime,
            {
                "relay_event": "subagent_finish",
                "subagent": subagent_type,
                "status": "completed",
            },
        )

        last_message = latest_messages[-1] if latest_messages else None
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
        if has_files:
            update["files"] = files_state
        if has_todos:
            update["todos"] = todos_state

        return Command(update=update)

    task.metadata = {
        "permission_config": {
            "permission": "task",
            # The subagent type is the concrete value being evaluated.
            "patterns_fn": lambda args: [args.get("subagent_type", "*")],
            # Approving one subagent type with "always" covers future calls
            # to the same subagent without re-prompting.
            "always_fn": lambda args: [args.get("subagent_type", "*")],
            "metadata_fn": lambda args: {
                "subagent": args.get("subagent_type", ""),
                "description": args.get("description", ""),
            },
        }
    }
    return task