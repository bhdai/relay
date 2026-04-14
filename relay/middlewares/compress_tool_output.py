"""Middleware for compressing large tool outputs to virtual filesystem.

When a tool returns more text than the token budget allows, this
middleware stores the full content in ``state["files"]`` and replaces
the message with a short reference.  The agent can later read the full
content via ``read_memory_file()``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.tools.internal.memory import read_memory_file

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


# ==============================================================================
# Token Counting
# ==============================================================================
#
# We need a lightweight way to estimate how many tokens a tool output
# consumes.  The model's own tokenizer is the most accurate, but we
# fall back to a character-based estimate when it's unavailable.


def _estimate_tokens(text: str, model: BaseChatModel) -> int:
    """Estimate token count for *text* using *model*'s tokenizer.

    Falls back to a character-based heuristic (4 chars ≈ 1 token) when
    the model does not support ``get_num_tokens_from_messages``.
    """
    try:
        msgs = [HumanMessage(content=text)]
        return model.get_num_tokens_from_messages(msgs)
    except (NotImplementedError, ImportError, Exception):
        return len(text) // 4


# ==============================================================================
# Middleware
# ==============================================================================


class CompressToolOutputMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Compress large tool outputs into virtual scratchpad files.

    When tool output exceeds the token limit stored in
    ``AgentContext.tool_output_max_tokens``:

    1. Stores full content in ``state["files"]``
    2. Replaces the message body with a short reference
    3. The agent can call ``read_memory_file()`` to retrieve the content
    """

    def __init__(self, model: BaseChatModel):
        super().__init__()
        self.model = model

    def _compress_if_needed(
        self, tool_msg: ToolMessage, request: ToolCallRequest
    ) -> ToolMessage | Command:
        """Compress *tool_msg* if it exceeds the token budget."""

        # Never compress error messages — they are usually short and the
        # agent needs to see them verbatim.
        if getattr(tool_msg, "status", None) == "error" or getattr(
            tool_msg, "is_error", False
        ):
            return tool_msg

        # Don't compress output from read_memory_file — that would create
        # an infinite loop of compression and retrieval.
        if tool_msg.name == read_memory_file.name:
            return tool_msg

        # Get the token budget from runtime context.
        max_tokens = (
            request.runtime.context.tool_output_max_tokens
            if request.runtime.context
            and hasattr(request.runtime.context, "tool_output_max_tokens")
            else None
        )

        if not max_tokens:
            return tool_msg

        text_content = tool_msg.text
        if not text_content or not text_content.strip():
            return tool_msg

        token_count = _estimate_tokens(text_content, self.model)

        if token_count <= max_tokens:
            return tool_msg

        # Store the full content in a virtual file and replace the message
        # with a short reference.
        file_id = f"tool_output_{tool_msg.tool_call_id}.txt"

        ref_content = (
            f"Tool output too large ({token_count} tokens), "
            f"stored in virtual file: {file_id}\n"
            f"Use read_memory_file('{file_id}') to access full content."
        )

        compressed_msg = ToolMessage(
            id=tool_msg.id,
            name=tool_msg.name,
            content=ref_content,
            tool_call_id=tool_msg.tool_call_id,
        )

        return Command(
            update={
                "messages": [compressed_msg],
                "files": {file_id: text_content},
            }
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)

        # If handler already returned a Command, pass through.
        if isinstance(result, Command):
            return result

        # If handler returned a ToolMessage, check if compression is needed.
        if isinstance(result, ToolMessage):
            return self._compress_if_needed(result, request)

        return result
