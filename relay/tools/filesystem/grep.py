"""Grep file search tool.

Searches file contents for a literal string or regex pattern while
pruning ``IGNORE_DIRS`` and skipping binary files.
"""

import re
from pathlib import Path

from langchain_core.tools import ToolException, tool

from relay.tools import walk_files
from relay.utils.paths import resolve


def _grep_match(
    root: Path,
    file_paths,
    pattern: str,
    *,
    is_regex: bool,
    max_results: int,
) -> tuple[list[str], bool]:
    """Search file contents for *pattern* and return matching lines.

    Each match is formatted as ``path:line_number: content``.
    Binary files and encoding errors are silently skipped.
    """
    if is_regex:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolException(f"Invalid regex: {exc}") from exc
        matcher = rx.search
    else:
        matcher = lambda line: pattern in line  # noqa: E731

    matches: list[str] = []
    for rel in file_paths:
        try:
            with open(root / rel, encoding="utf-8", errors="strict") as f:
                for lineno, line in enumerate(f, 1):
                    if matcher(line):
                        matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                        if len(matches) >= max_results:
                            return matches, True
        except (UnicodeDecodeError, OSError):
            # Skip binary files and unreadable paths.
            continue
    return matches, False


@tool
async def grep_files(
    pattern: str,
    dir_path: str = ".",
    is_regex: bool = False,
    max_results: int = 100,
) -> str:
    """Search file contents for a text pattern or regex.

    Automatically ignores common noisy directories (e.g. ``.venv``,
    ``node_modules``, ``__pycache__``).  Binary files are skipped.

    Args:
        pattern: The literal string or regex to search for.
        dir_path: Root directory to search (default: current directory).
        is_regex: Whether *pattern* should be treated as a regex (default False).
        max_results: Maximum number of matching lines to return (default 100).
    """
    root = resolve(dir_path)
    if not root.is_dir():
        raise ToolException(f"Not a directory: {root}")

    matches, truncated = _grep_match(
        root, walk_files(root), pattern, is_regex=is_regex, max_results=max_results,
    )

    if not matches:
        return f"No matches for '{pattern}'"

    output = "\n".join(matches)
    if truncated:
        output += f"\n\n... (truncated, showing first {max_results} matches)"
    return output
