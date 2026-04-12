"""Utility for constructing ToolMessage instances with metadata."""

from langchain_core.messages import ToolMessage


def create_tool_message(
    *,
    result: str,
    tool_name: str,
    tool_call_id: str,
    is_error: bool = False,
    return_direct: bool = False,
) -> ToolMessage:
    """Build a ``ToolMessage`` with optional error / return_direct flags.

    These extra attributes are inspected by middlewares (e.g.
    ``ReturnDirectMiddleware``) to alter control flow.
    """
    msg = ToolMessage(
        content=result,
        name=tool_name,
        tool_call_id=tool_call_id,
        is_error=is_error,
        return_direct=return_direct,
    )
    if is_error:
        msg.status = "error"
    return msg
