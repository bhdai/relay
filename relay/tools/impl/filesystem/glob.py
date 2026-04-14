"""Glob file search tool.

Finds files by matching relative paths against a glob pattern while
pruning ``IGNORE_DIRS`` at every level.
"""

import fnmatch

from langchain_core.tools import ToolException, tool

from relay.tools import walk_files
from relay.utils.paths import resolve


def _glob_match(file_paths, pattern: str, max_results: int) -> tuple[list[str], bool]:
    """Filter *file_paths* with ``fnmatch`` and cap at *max_results*."""
    matches: list[str] = []
    for rel in file_paths:
        if fnmatch.fnmatch(rel, pattern):
            matches.append(rel)
            if len(matches) >= max_results:
                return matches, True
    return matches, False


@tool
async def glob_files(
    pattern: str,
    dir_path: str = ".",
    max_results: int = 100,
) -> str:
    """Find files whose relative path matches a glob pattern.

    Use this instead of ``ls`` when you need to filter by file type or
    naming pattern — faster and more precise than scanning the whole
    tree.  Prefer this over ``run_command("find …")``  for file-name
    searches; reserve ``grep_files`` for content searches.

    Automatically ignores common noisy directories (e.g. ``.venv``,
    ``node_modules``, ``__pycache__``).

    Typical patterns::

        "*.py"          – all Python files in dir_path
        "**/*.ts"       – all TypeScript files at any depth
        "tests/**/test_*.py"  – test modules under tests/
        "src/**/index.*"      – entry-point files under src/

    Args:
        pattern: Glob pattern to match against relative file paths.
        dir_path: Root directory to search (default: current directory).
        max_results: Maximum number of matches to return (default 100).
    """
    root = resolve(dir_path)
    if not root.is_dir():
        raise ToolException(f"Not a directory: {root}")

    matches, truncated = _glob_match(walk_files(root), pattern, max_results)

    if not matches:
        return f"No files matching '{pattern}'"

    output = "\n".join(matches)
    if truncated:
        output += f"\n\n... (truncated, showing first {max_results} matches)"
    return output
