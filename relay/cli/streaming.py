"""Graph streaming with interrupt/resume support.

The ``stream_response`` coroutine owns the inner loop that streams
from a LangGraph ``CompiledGraph``.  When an ``__interrupt__`` event
is detected (e.g. a tool requesting approval), it prompts the user
and re-enters the stream with ``Command(resume=...)``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.types import Command, Interrupt
from prompt_toolkit import PromptSession

from relay.agents.context import AgentContext
from relay.cli.renderer import (
    console,
    render_cost_summary,
    render_info,
    render_tool_call,
)


# ==============================================================================
# Interrupt prompting
# ==============================================================================


async def prompt_for_interrupt(
    interrupts: list[Interrupt],
) -> dict[str, Any] | None:
    """Prompt the user for each pending interrupt.

    Returns a ``{interrupt_id: user_choice}`` dict suitable for
    ``Command(resume=...)``, or ``None`` if the user cancels.
    """
    resume_map: dict[str, Any] = {}

    for intr in interrupts:
        payload = intr.value

        # Display the interrupt question.
        if hasattr(payload, "question"):
            console.print(f"  ? {payload.question}", style="bold yellow")
        else:
            console.print(f"  ? {payload}", style="bold yellow")

        # Show options if available.
        options: list[str] = []
        if hasattr(payload, "options") and payload.options:
            options = payload.options
            for j, opt in enumerate(options, 1):
                console.print(f"    {j}. {opt}", style="dim")

        # Collect user response.
        try:
            session = PromptSession()
            answer = await session.prompt_async("  → ")
        except (EOFError, KeyboardInterrupt):
            return None

        answer = answer.strip()

        # If the user typed a number and we have options, map it.
        if options and answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(options):
                answer = options[idx]

        resume_map[intr.id] = answer

    return resume_map


# ==============================================================================
# Streaming
# ==============================================================================


class _TurnStats:
    """Mutable accumulator for per-turn token/cost data."""

    __slots__ = ("input_tokens", "output_tokens", "cost", "collected_text")

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cost: float = 0.0
        self.collected_text: str = ""


async def stream_response(
    graph: Any,
    input_value: Any,
    *,
    thread_id: str,
) -> _TurnStats:
    """Stream the graph response, handling interrupts automatically.

    Parameters
    ----------
    graph:
        Compiled LangGraph graph.
    input_value:
        Either a ``dict`` with a ``HumanMessage`` for normal turns, or
        a ``Command(resume=...)`` when resuming after an interrupt.
    thread_id:
        The conversation thread to operate on.

    Returns
    -------
    _TurnStats:
        Token counts and accumulated text for this turn.
    """
    config = {"configurable": {"thread_id": thread_id}}
    context = AgentContext()
    stats = _TurnStats()
    current_input = input_value

    # ==================================================================
    # Interrupt/resume loop
    #
    # Each iteration streams from the graph.  If the graph raises an
    # interrupt (e.g. for approval), we prompt the user and loop back
    # with Command(resume=...).  If no interrupt, we break out after
    # the stream completes.
    # ==================================================================

    while True:
        interrupted = False

        stream = graph.astream(
            current_input,
            config=config,
            stream_mode=["messages", "updates"],
            context=context,
        )

        async for namespace, chunk in stream:
            # -- Token-level chunks ("messages" mode) --
            if namespace == "messages":
                msg, _metadata = chunk
                if isinstance(msg, AIMessageChunk) and msg.content:
                    print(msg.content, end="", flush=True)
                    stats.collected_text += msg.content

            # -- Node-level updates ("updates" mode) --
            elif namespace == "updates":
                if not isinstance(chunk, dict):
                    continue

                # ---- Interrupt detection ----
                raw_interrupts = chunk.get("__interrupt__")
                if raw_interrupts:
                    # Finish any partial output line.
                    if stats.collected_text and not stats.collected_text.endswith(
                        "\n"
                    ):
                        print()
                        stats.collected_text = ""

                    resume_value = await prompt_for_interrupt(raw_interrupts)
                    if resume_value is not None:
                        current_input = Command(resume=resume_value)
                        interrupted = True
                        break
                    else:
                        # User cancelled — stop the loop.
                        break

                # ---- Normal node output ----
                for _node_name, node_output in chunk.items():
                    if not isinstance(node_output, dict):
                        continue

                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if hasattr(msg, "tool_calls"):
                            for tc in msg.tool_calls:
                                render_tool_call(tc["name"], tc.get("args", {}))

                        if isinstance(msg, ToolMessage) and msg.status == "error":
                            console.print(
                                f"  ✗ {msg.name}: {msg.content[:200]}",
                                style="bold red",
                            )

                    # Extract cost data from state updates.
                    if "current_input_tokens" in node_output:
                        stats.input_tokens = (
                            node_output["current_input_tokens"] or 0
                        )
                    if "current_output_tokens" in node_output:
                        stats.output_tokens = (
                            node_output["current_output_tokens"] or 0
                        )
                    if "total_cost" in node_output:
                        stats.cost = node_output["total_cost"] or 0.0

        if not interrupted:
            break

    return stats
