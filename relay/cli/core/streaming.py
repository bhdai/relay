"""Graph streaming with interrupt/resume support.

The ``stream_response`` coroutine owns the inner loop that streams
from a LangGraph ``CompiledGraph``.  When an ``__interrupt__`` event
is detected (e.g. a tool requesting approval), it prompts the user
and re-enters the stream with ``Command(resume=...)``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command, Interrupt
from prompt_toolkit import PromptSession

from relay.agents.context import AgentContext
from relay.cli.theme import console
from relay.cli.ui.renderer import (
    render_assistant_message,
    render_cost_summary,
    render_info,
    render_tool_call,
    render_tool_error,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# User Memory Loading
# ==============================================================================
#
# If a ``memory.md`` file exists in the ``.relay/`` directory, its
# content is injected into ``AgentContext.user_memory`` so the system
# prompt can reference it.  This mirrors langrepl's approach of loading
# persistent user/project memory into context.

_RELAY_DIR = ".relay"
_MEMORY_FILENAME = "memory.md"


def _message_text(content: Any) -> str:
    """Extract printable text from LangChain message content values."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)

    return ""


def _close_open_text_line(stats: "_TurnStats") -> None:
    """Finish the current streamed line before structured output."""
    if stats.line_open:
        print()
        stats.line_open = False


def _load_user_memory(working_dir: str | None = None) -> str:
    """Read ``.relay/memory.md`` and return its content, or ``""``."""
    base = Path(working_dir) if working_dir else Path.cwd()
    memory_file = base / _RELAY_DIR / _MEMORY_FILENAME
    try:
        if memory_file.is_file():
            content = memory_file.read_text(encoding="utf-8")
            logger.debug("Loaded user memory from %s (%d chars)", memory_file, len(content))
            return content
    except OSError as exc:
        logger.warning("Failed to read user memory: %s", exc)
    return ""


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
            console.print(
                f"  ? {payload.question}",
                style=console.get_style("warning", bold=True),
            )
        else:
            console.print(
                f"  ? {payload}",
                style=console.get_style("warning", bold=True),
            )

        # Show options if available.
        options: list[str] = []
        if hasattr(payload, "options") and payload.options:
            options = payload.options
            for j, opt in enumerate(options, 1):
                console.print(f"    {j}. {opt}", style="muted")

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

    __slots__ = (
        "input_tokens",
        "output_tokens",
        "cost",
        "collected_text",
        "line_open",
    )

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cost: float = 0.0
        self.collected_text: str = ""
        self.line_open: bool = False


class _DisplayState:
    """Track structured output already rendered for the current turn."""

    __slots__ = (
        "buffered_message_chunks",
        "rendered_ai_messages",
        "rendered_tool_calls",
        "rendered_tool_errors",
        "announced_subagents",
    )

    def __init__(self) -> None:
        self.buffered_message_chunks: dict[tuple[str, ...], list[AIMessageChunk]] = {}
        self.rendered_ai_messages: set[str] = set()
        self.rendered_tool_calls: set[str] = set()
        self.rendered_tool_errors: set[str] = set()
        self.announced_subagents: set[str] = set()


def _unpack_stream_event(raw_event: Any) -> tuple[tuple[str, ...], str, Any] | None:
    """Normalize graph stream events with or without subgraph metadata."""
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


def _tool_call_key(tool_call: dict[str, Any], *, fallback_index: int) -> str:
    """Build a stable dedupe key for a rendered tool call."""
    call_id = tool_call.get("id")
    if isinstance(call_id, str) and call_id:
        return call_id

    name = tool_call.get("name") or "tool"
    args = tool_call.get("args", {})
    return f"{name}:{args!r}:{fallback_index}"


def _tool_error_key(message: ToolMessage, *, fallback_index: int) -> str:
    """Build a stable dedupe key for a rendered tool error."""
    if message.tool_call_id:
        return message.tool_call_id
    if message.id:
        return str(message.id)
    return f"{message.name}:{message.content!r}:{fallback_index}"


def _message_key(message: AIMessage, *, fallback_index: int) -> str:
    """Build a stable dedupe key for rendered assistant messages."""
    if message.id:
        return str(message.id)

    fingerprint = hashlib.sha256(
        f"{message.content!r}:{getattr(message, 'tool_calls', None)!r}:{fallback_index}".encode()
    ).hexdigest()
    return fingerprint


def _merge_message_chunks(chunks: list[AIMessageChunk]) -> AIMessage:
    """Merge streamed chunks into a final AIMessage."""
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk

    return AIMessage(
        content=merged.content,
        additional_kwargs=merged.additional_kwargs,
        response_metadata=merged.response_metadata,
        tool_calls=merged.tool_calls,
        id=merged.id,
        name=merged.name,
    )


def _iter_node_outputs(update: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract node output payloads from a streamed update event."""
    if "messages" in update:
        return [update]

    outputs: list[dict[str, Any]] = []
    for value in update.values():
        if isinstance(value, dict):
            outputs.append(value)
    return outputs


def _render_ai_message(
    message: AIMessage,
    *,
    stats: _TurnStats,
    display_state: _DisplayState,
    indent_level: int = 0,
    fallback_index: int = 0,
) -> None:
    """Render a full assistant message once per turn."""
    text = _message_text(message.content)
    if not text.strip():
        return

    message_key = _message_key(message, fallback_index=fallback_index)
    if message_key in display_state.rendered_ai_messages:
        return

    display_state.rendered_ai_messages.add(message_key)
    _close_open_text_line(stats)
    render_assistant_message(message, indent_level=indent_level)
    stats.collected_text += text


def _buffer_message_chunk(
    message: AIMessageChunk,
    *,
    display_namespace: tuple[str, ...],
    display_state: _DisplayState,
) -> None:
    """Accumulate streamed assistant chunks until a stable render point."""
    display_state.buffered_message_chunks.setdefault(display_namespace, []).append(message)


def _finalize_buffered_message(
    display_namespace: tuple[str, ...],
    *,
    stats: _TurnStats,
    display_state: _DisplayState,
) -> None:
    """Render and clear any buffered assistant chunks for one namespace."""
    chunks = display_state.buffered_message_chunks.pop(display_namespace, None)
    if not chunks:
        return

    _render_ai_message(
        _merge_message_chunks(chunks),
        stats=stats,
        display_state=display_state,
        indent_level=len(display_namespace),
    )


def _finalize_all_buffered_messages(
    *,
    stats: _TurnStats,
    display_state: _DisplayState,
) -> None:
    """Render any buffered assistant chunks that never received updates."""
    for display_namespace in list(display_state.buffered_message_chunks):
        _finalize_buffered_message(
            display_namespace,
            stats=stats,
            display_state=display_state,
        )


def _handle_node_output(
    node_output: dict[str, Any],
    *,
    stats: _TurnStats,
    display_state: _DisplayState,
    indent_level: int = 0,
) -> None:
    """Render a node update from the graph stream."""
    messages = node_output.get("messages", [])
    for index, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            _render_ai_message(
                msg,
                stats=stats,
                display_state=display_state,
                indent_level=indent_level,
                fallback_index=index,
            )

        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                tool_key = _tool_call_key(tc, fallback_index=index)
                if tool_key in display_state.rendered_tool_calls:
                    continue

                display_state.rendered_tool_calls.add(tool_key)
                _close_open_text_line(stats)
                render_tool_call(
                    tc["name"],
                    tc.get("args", {}),
                    indent_level=indent_level,
                )

        if isinstance(msg, ToolMessage) and msg.status == "error":
            error_key = _tool_error_key(msg, fallback_index=index)
            if error_key in display_state.rendered_tool_errors:
                continue

            display_state.rendered_tool_errors.add(error_key)
            _close_open_text_line(stats)
            render_tool_error(
                msg.name or "tool",
                msg.content[:200],
                indent_level=indent_level,
            )

    if "current_input_tokens" in node_output:
        stats.input_tokens = node_output["current_input_tokens"] or 0
    if "current_output_tokens" in node_output:
        stats.output_tokens = node_output["current_output_tokens"] or 0
    if "total_cost" in node_output:
        stats.cost = node_output["total_cost"] or 0.0


def _handle_custom_event(
    chunk: Any,
    *,
    namespace: tuple[str, ...],
    stats: _TurnStats,
    display_state: _DisplayState,
) -> None:
    """Render delegated subagent activity forwarded by the task tool."""
    if not isinstance(chunk, dict):
        return

    event_type = chunk.get("relay_event")
    subagent = chunk.get("subagent") or "subagent"
    child_namespace_raw = chunk.get("namespace") or []
    child_namespace = tuple(str(part) for part in child_namespace_raw)
    display_namespace = (*namespace, "subagent", *child_namespace)
    indent_level = len(display_namespace)

    if event_type == "subagent_start":
        description = str(chunk.get("description", "")).strip()
        dedupe_key = f"{subagent}:{description}"
        if dedupe_key in display_state.announced_subagents:
            return

        display_state.announced_subagents.add(dedupe_key)
        _close_open_text_line(stats)
        prefix = "  " * indent_level
        render_info(f"{prefix}↳ {subagent}: {description}")
        return

    if event_type == "subagent_message":
        message = chunk.get("message")
        if isinstance(message, AIMessageChunk):
            _buffer_message_chunk(
                message,
                display_namespace=display_namespace,
                display_state=display_state,
            )
        return

    if event_type != "subagent_update":
        return

    update = chunk.get("update")
    if not isinstance(update, dict):
        return

    _finalize_buffered_message(
        display_namespace,
        stats=stats,
        display_state=display_state,
    )

    for node_output in _iter_node_outputs(update):
        _handle_node_output(
            node_output,
            stats=stats,
            display_state=display_state,
            indent_level=indent_level,
        )


async def stream_response(
    graph: Any,
    input_value: Any,
    *,
    thread_id: str,
    working_dir: str | None = None,
    input_cost_per_mtok: float = 0.0,
    output_cost_per_mtok: float = 0.0,
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
    context = AgentContext(
        working_dir=working_dir or str(Path.cwd()),
        user_memory=_load_user_memory(working_dir),
        input_cost_per_mtok=input_cost_per_mtok,
        output_cost_per_mtok=output_cost_per_mtok,
    )
    stats = _TurnStats()
    display_state = _DisplayState()
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
            stream_mode=["messages", "updates", "custom"],
            context=context,
            subgraphs=True,
        )

        async for raw_event in stream:
            event = _unpack_stream_event(raw_event)
            if event is None:
                continue

            namespace, mode, chunk = event

            # -- Token-level chunks ("messages" mode) --
            if mode == "messages":
                if not isinstance(chunk, Sequence) or len(chunk) != 2:
                    continue

                msg, _metadata = chunk
                if isinstance(msg, AIMessageChunk):
                    _buffer_message_chunk(
                        msg,
                        display_namespace=namespace,
                        display_state=display_state,
                    )

            # -- Node-level updates ("updates" mode) --
            elif mode == "updates":
                if not isinstance(chunk, dict):
                    continue

                # ---- Interrupt detection ----
                raw_interrupts = chunk.get("__interrupt__")
                if raw_interrupts:
                    _close_open_text_line(stats)

                    resume_value = await prompt_for_interrupt(raw_interrupts)
                    if resume_value is not None:
                        current_input = Command(resume=resume_value)
                        interrupted = True
                        break

                    break

                # ---- Normal node output ----
                _finalize_buffered_message(
                    namespace,
                    stats=stats,
                    display_state=display_state,
                )

                for node_output in _iter_node_outputs(chunk):
                    _handle_node_output(
                        node_output,
                        stats=stats,
                        display_state=display_state,
                        indent_level=len(namespace),
                    )

            elif mode == "custom":
                _handle_custom_event(
                    chunk,
                    namespace=namespace,
                    stats=stats,
                    display_state=display_state,
                )

        if not interrupted:
            _finalize_all_buffered_messages(stats=stats, display_state=display_state)
            break

    return stats
