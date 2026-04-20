"""Virtual file-system tools for the agent's in-memory scratchpad.

These tools let the agent persist notes, plans, and other working text
inside ``state["files"]``, a ``dict[str, str]`` managed by the
``file_reducer`` in relay state.

Each mutating tool returns a ``Command(update=...)`` so LangGraph merges
the change through the reducer rather than clobbering the whole dict.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, ToolException, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from relay.agents.state import AgentState


@tool
def list_memory_files(state: Annotated[dict, InjectedState]) -> str:
    """List all files in the agent's in-memory scratchpad.

    Call this to orient yourself before any other memory-file operation.
    Use the scratchpad to persist plans, intermediate results, and notes
    that need to survive across many tool calls in the same session.
    """
    files = state.get("files") or {}
    if not files:
        return "(empty)"
    return "\n".join(sorted(files.keys()))


@tool
def read_memory_file(
    filename: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Read the contents of an in-memory scratchpad file.

    Args:
        filename: Key of the file to read.
    """
    files = state.get("files") or {}
    if filename not in files:
        raise ToolException(f"File not found: {filename}")
    return files[filename]


@tool
def write_memory_file(
    filename: str,
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Create or overwrite an in-memory scratchpad file.

    Use for storing plans, checklists, intermediate results, or any
    context you want to re-read later in the same session.  Prefer
    ``edit_memory_file`` when updating an existing file to avoid
    accidentally discarding content.

    Args:
        filename: Key for the file (e.g. ``"plan.md"``, ``"notes.txt"``).
        content: Full text to store.
    """
    return Command(
        update={
            "files": {filename: content},
            "messages": [
                ToolMessage(
                    content=f"Written: {filename}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def edit_memory_file(
    filename: str,
    old_content: str,
    new_content: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Replace a substring inside an existing in-memory scratchpad file.

    Args:
        filename: Key of the file to edit.
        old_content: Exact substring to find and replace.
        new_content: Replacement text.
    """
    files = dict(state.get("files") or {})
    if filename not in files:
        raise ToolException(f"File not found: {filename}")
    if old_content not in files[filename]:
        raise ToolException(
            f"old_content not found in {filename}. "
            "Read the file first to get the exact text."
        )
    files[filename] = files[filename].replace(old_content, new_content, 1)
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"Edited: {filename}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


MEMORY_TOOLS = [
    list_memory_files,
    read_memory_file,
    write_memory_file,
    edit_memory_file,
]
