"""Directory listing tool — tree-view of a directory structure.

Walks the filesystem in Python with an explicit ignore set (modelled
after OpenCode's ``list`` tool) to avoid leaking noisy directories like
``.venv/`` or ``node_modules/``.
"""

import fnmatch
import os
from pathlib import Path

from langchain_core.tools import ToolException, tool

from relay.tools import IGNORE_DIRS


def _collect_files(
    root: Path,
    max_files: int,
    extra_ignore: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Walk *root* and return up to *max_files* relative paths.

    Directories whose name appears in ``IGNORE_DIRS`` are pruned at every
    level so they never contribute entries or slow the walk.  Additional
    *extra_ignore* glob patterns (e.g. ``["tests/", "*.bak"]``) are
    applied to relative paths after the directory pruning step.

    Returns ``(paths, truncated)`` where *truncated* is ``True`` when the
    walk was stopped early because the limit was reached.
    """
    files: list[str] = []
    truncated = False

    def _ignored_by_extra(rel: str) -> bool:
        if not extra_ignore:
            return False
        return any(fnmatch.fnmatch(rel, pat) for pat in extra_ignore)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories *in-place* so os.walk skips them.
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORE_DIRS
        )

        rel_dir = Path(dirpath).relative_to(root)

        # Also prune directories that match extra_ignore patterns.
        if extra_ignore:
            dirnames[:] = [
                d for d in dirnames
                if not _ignored_by_extra(
                    str(rel_dir / d) + "/" if str(rel_dir) != "." else d + "/"
                )
            ]

        for fname in sorted(filenames):
            if len(files) >= max_files:
                truncated = True
                break
            rel = str(rel_dir / fname) if str(rel_dir) != "." else fname
            if _ignored_by_extra(rel):
                continue
            files.append(rel)

        if truncated:
            break

    return files, truncated


def _render_tree(file_paths: list[str]) -> str:
    """Build a tree-style string from a sorted list of relative paths.

    Example:
        src/
          foo.py
          bar/
            baz.ts
        README.md
    """
    # Build a nested dict representing the directory tree.
    tree: dict = {}
    for p in file_paths:
        parts = Path(p).parts
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        # Files are stored as leaves with ``None`` value.
        node[parts[-1]] = None

    lines: list[str] = []

    def _walk(node: dict, indent: int) -> None:
        prefix = "  " * indent
        for key in sorted(node.keys(), key=lambda k: (not k.endswith("/"), k)):
            lines.append(f"{prefix}{key}")
            child = node[key]
            if isinstance(child, dict):
                _walk(child, indent + 1)

    _walk(tree, 0)
    return "\n".join(lines)


@tool
async def ls(
    dir_path: str = ".",
    ignore: list[str] | None = None,
    max_files: int = 200,
) -> str:
    """List files and directories as a tree, skipping common noisy folders.

    Always ignores built-in patterns (e.g. ``.venv``, ``node_modules``,
    ``__pycache__``).  Supply additional patterns via *ignore*.

    Args:
        dir_path: Path to the directory (default: current directory).
        ignore: Optional list of extra glob patterns to skip
                (e.g. ``["tests/", "*.bak"]``).
        max_files: Maximum number of files to include (default 200).
    """
    resolved = Path(dir_path).resolve()
    if not resolved.is_dir():
        raise ToolException(f"Not a directory: {resolved}")

    file_paths, truncated = _collect_files(resolved, max_files, extra_ignore=ignore)

    if not file_paths:
        return f"(empty directory: {resolved})"

    output = _render_tree(file_paths)
    if truncated:
        output += f"\n\n... (truncated, showing first {max_files} files)"
    return output
