"""Shared constants and utilities for relay tools."""

import os
from pathlib import Path

# ==============================================================================
# Ignore Patterns
# ==============================================================================
#
# Directories that are always pruned during filesystem walks.  Modelled after
# OpenCode's hardcoded ignore set so that noisy build/cache/venv directories
# never leak into tool output.

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


def walk_files(root: Path):
    """Yield relative file paths under *root*, pruning ``IGNORE_DIRS``."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
        rel_dir = os.path.relpath(dirpath, root)
        for fname in sorted(filenames):
            yield fname if rel_dir == "." else f"{rel_dir}/{fname}"
