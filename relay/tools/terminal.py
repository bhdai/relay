"""Terminal tools for executing shell commands and inspecting directory structure.

Provides the agent with controlled access to the system shell and a
tree-view directory listing utility.
"""

import asyncio
import os
import re
from pathlib import Path

from langchain_core.tools import ToolException, tool


# ==============================================================================
# Command Parsing Helpers
# ==============================================================================
#
# These helpers break compound shell commands into individual parts so that
# each segment can be inspected independently (e.g. for logging or policy
# checks).  They handle ``&&``, ``||``, ``;``, ``|``, and nested ``$(…)``
# / backtick substitutions.

_CHAIN_OPS = re.compile(r"\s*(&&|\|\||;|\|)\s*")
_SUBST_DOLLAR = re.compile(r"\$\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
_SUBST_BACKTICK = re.compile(r"`([^`]+)`")


def _extract_command_parts(command: str) -> list[str]:
    """Return every atomic command segment in *command*.

    Handles chained operators (``&&``, ``||``, ``;``, ``|``) and nested
    command substitutions (``$(…)`` and backticks).
    """
    parts: list[str] = []
    for seg in _CHAIN_OPS.split(command):
        seg = seg.strip()
        if not seg or seg in ("&&", "||", ";", "|"):
            continue
        parts.append(seg)
        for pattern in (_SUBST_DOLLAR, _SUBST_BACKTICK):
            for m in pattern.finditer(seg):
                parts.extend(_extract_command_parts(m.group(1)))
    return parts


def _format_output(stdout: str, stderr: str) -> str:
    """Merge *stdout* and *stderr* into a single result string."""
    sections: list[str] = []
    if stdout.strip():
        sections.append(stdout.strip())
    if stderr.strip():
        sections.append(stderr.strip())
    return "\n".join(sections) if sections else "Command completed successfully"


# ==============================================================================
# Tools
# ==============================================================================


@tool
async def run_command(command: str) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to execute.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
    except Exception as exc:
        raise ToolException(f"run command: {exc}") from exc

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        error_msg = stderr.strip() if stderr.strip() else f"Command failed with exit code {proc.returncode}"
        raise ToolException(error_msg)

    return _format_output(stdout, stderr)


# ==============================================================================
# Directory Listing
# ==============================================================================
#
# Instead of shelling out to ``find | head | sort``, we walk the filesystem in
# Python with an explicit ignore set (modelled after OpenCode's ``list`` tool).
# This avoids leaking noisy directories like ``.venv/`` or ``node_modules/``
# and produces a deterministic, tree-formatted output.

IGNORE_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".tox",
    ".nox",
    ".eggs",
    "dist",
    "build",
    "target",
    "vendor",
    ".next",
    ".nuxt",
    ".output",
    "coverage",
    ".coverage",
    ".idea",
    ".vscode",
}


def _collect_files(root: Path, max_files: int) -> tuple[list[str], bool]:
    """Walk *root* and return up to *max_files* relative paths.

    Directories whose name appears in ``IGNORE_DIRS`` are pruned at every
    level so they never contribute entries or slow the walk.

    Returns ``(paths, truncated)`` where *truncated* is ``True`` when the
    walk was stopped early because the limit was reached.
    """
    files: list[str] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories *in-place* so os.walk skips them.
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORE_DIRS
        )

        rel_dir = Path(dirpath).relative_to(root)

        for fname in sorted(filenames):
            if len(files) >= max_files:
                truncated = True
                break
            rel = str(rel_dir / fname) if str(rel_dir) != "." else fname
            files.append(rel)

        if truncated:
            break

    return files, truncated


def _render_tree(file_paths: list[str]) -> str:
    """Build a tree-style string from a sorted list of relative paths.

    Example output::

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
async def get_directory_structure(
    dir_path: str = ".",
    max_files: int = 200,
) -> str:
    """Return a tree-view listing of a directory structure.

    Automatically ignores common noisy directories (e.g. ``.venv``,
    ``node_modules``, ``__pycache__``).

    Args:
        dir_path: Path to the directory (default: current directory).
        max_files: Maximum number of files to include (default 200).
    """
    resolved = Path(dir_path).resolve()
    if not resolved.is_dir():
        raise ToolException(f"Not a directory: {resolved}")

    file_paths, truncated = _collect_files(resolved, max_files)

    if not file_paths:
        return f"(empty directory: {resolved})"

    output = _render_tree(file_paths)
    if truncated:
        output += f"\n\n... (truncated, showing first {max_files} files)"
    return output


TERMINAL_TOOLS = [run_command, get_directory_structure]
