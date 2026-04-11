"""File reading, writing, and editing tools.

Provides paginated reading, guarded writes (no overwriting), and
multi-edit support with both exact and whitespace-normalised matching.
"""

import shutil
from pathlib import Path

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, Field

from relay.utils.paths import resolve


class EditOperation(BaseModel):
    """A single find-and-replace edit within a file."""

    old_content: str = Field(..., description="The exact text to be replaced")
    new_content: str = Field(..., description="The replacement text")


class MoveOperation(BaseModel):
    """A single file move from source to destination."""

    source: str = Field(..., description="Source file path")
    destination: str = Field(..., description="Destination file path")


# ==============================================================================
# File Content Pagination
# ==============================================================================
#
# ``_paginate_file`` is the pure-logic core of ``read_file``.  It slices a
# list of lines and formats them with line numbers so the agent can
# reference specific locations when editing.


def _paginate_file(
    lines: list[str],
    start_line: int = 0,
    limit: int = 500,
) -> str:
    """Return a numbered slice of *lines* with a summary footer.

    Args:
        lines: All lines of the file (including trailing newlines).
        start_line: 0-based index of the first line to include.
        limit: Maximum number of lines to return.

    Returns:
        A string with numbered lines and a ``[start-end, n/total lines]``
        footer suitable for display to the agent.
    """
    total = len(lines)
    start = max(0, start_line)
    end = min(total, start + limit)
    selected = lines[start:end]

    numbered = "\n".join(
        f"{i + start:4d} - {line.rstrip()}" for i, line in enumerate(selected)
    )

    actual_end = start + len(selected) - 1 if selected else start
    return f"{numbered}\n\n[{start}-{actual_end}, {len(selected)}/{total} lines]"


# ==============================================================================
# Edit Helpers
# ==============================================================================


def _find_match(content: str, search: str) -> tuple[bool, int, int]:
    """Locate *search* inside *content*, returning ``(found, start, end)``.

    Falls back to whitespace-normalised matching when an exact lookup fails
    so that minor indentation differences from the LLM don't cause errors.
    """
    # Exact match — fast path.
    idx = content.find(search)
    if idx != -1:
        return True, idx, idx + len(search)

    # Whitespace-normalised fallback: collapse runs of whitespace on each
    # line while preserving line structure, then search again.
    def _normalise(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines())

    norm_content = _normalise(content)
    norm_search = _normalise(search)
    idx = norm_content.find(norm_search)
    if idx != -1:
        # Map the normalised offset back to the original content.  We walk
        # through the original lines to find the byte range that corresponds
        # to the matched normalised region.
        orig_lines = content.splitlines(keepends=True)
        norm_lines = norm_content.splitlines(keepends=True)

        # Build a mapping from normalised-offset → original-offset at line
        # boundaries so we can bracket the match.
        norm_pos = 0
        orig_pos = 0
        line_map: list[tuple[int, int]] = []  # (norm_start, orig_start)
        for nl, ol in zip(norm_lines, orig_lines):
            line_map.append((norm_pos, orig_pos))
            norm_pos += len(nl)
            orig_pos += len(ol)
        line_map.append((norm_pos, orig_pos))  # sentinel for end

        # Find the original start/end via the line map.
        orig_start = 0
        orig_end = len(content)
        for i, (ns, os_) in enumerate(line_map):
            if ns <= idx:
                orig_start = os_
            if ns >= idx + len(norm_search):
                orig_end = os_
                break

        return True, orig_start, orig_end

    return False, -1, -1


def _apply_edits(content: str, edits: list[EditOperation]) -> str:
    """Apply a sequence of non-overlapping edits to *content*.

    Raises ``ToolException`` when a match cannot be found or edits overlap.
    """
    matches: list[tuple[int, int, int, str]] = []  # (idx, start, end, new)
    for i, edit in enumerate(edits):
        found, start, end = _find_match(content, edit.old_content)
        if not found:
            raise ToolException(
                f"Edit #{i + 1}: could not find the specified old_content in the file."
            )
        matches.append((i, start, end, edit.new_content))

    # Check for overlapping ranges.
    sorted_matches = sorted(matches, key=lambda m: m[1])
    for j in range(len(sorted_matches) - 1):
        curr_idx, _, curr_end, _ = sorted_matches[j]
        next_idx, next_start, _, _ = sorted_matches[j + 1]
        if next_start < curr_end:
            raise ToolException(
                f"Overlapping edits: edit #{curr_idx + 1} and #{next_idx + 1}"
            )

    # Apply in reverse order so earlier offsets stay valid.
    result = content
    for _, start, end, new in sorted(matches, key=lambda m: m[1], reverse=True):
        result = result[:start] + new + result[end:]
    return result


# ==============================================================================
# Tools
# ==============================================================================


@tool
async def read_file(file_path: str, start_line: int = 0, limit: int = 500) -> str:
    """Read a file with line-based pagination.

    Args:
        file_path: Path to the file to read.
        start_line: 0-based starting line number (default 0).
        limit: Maximum number of lines to return (default 500).
    """
    path = resolve(file_path)
    try:
        with open(path, encoding="utf-8") as f:
            all_lines = f.readlines()
    except FileNotFoundError as exc:
        raise ToolException(f"File not found: {path}") from exc
    return _paginate_file(all_lines, start_line, limit)


@tool
async def write_file(file_path: str, content: str) -> str:
    """Create a new file with the given content.

    Use this only for files that **do not yet exist**; use ``edit_file``
    to modify existing files.

    Args:
        file_path: Path to the file to create.
        content: Content to write.
    """
    path = resolve(file_path)
    if path.exists():
        raise ToolException(f"File already exists: {path}. Use edit_file instead.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"File written: {path}"


@tool
async def edit_file(file_path: str, edits: list[EditOperation]) -> str:
    """Edit an existing file by applying one or more find-and-replace operations.

    Args:
        file_path: Path to the file to edit.
        edits: A list of ``{old_content, new_content}`` replacements.
    """
    path = resolve(file_path)
    if not path.exists():
        raise ToolException(f"File does not exist: {path}")

    original = path.read_text(encoding="utf-8")
    updated = _apply_edits(original, edits)
    path.write_text(updated, encoding="utf-8")
    return f"File edited: {path}"


@tool
async def create_dir(dir_path: str) -> str:
    """Create a directory (and any missing parents).

    Args:
        dir_path: Path to the directory to create.
    """
    path = resolve(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return f"Directory created: {path}"


@tool
async def move_file(source_path: str, destination_path: str) -> str:
    """Move a file from *source_path* to *destination_path*.

    Args:
        source_path: Path to the source file.
        destination_path: Path to the destination.
    """
    src = resolve(source_path)
    dst = resolve(destination_path)
    if not src.exists():
        raise ToolException(f"Source does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"File moved: {src} -> {dst}"


@tool
async def delete_file(file_path: str) -> str:
    """Delete a file.

    Args:
        file_path: Path to the file to delete.
    """
    path = resolve(file_path)
    if not path.exists():
        raise ToolException(f"File does not exist: {path}")
    path.unlink()
    return f"File deleted: {path}"
