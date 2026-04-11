"""Path resolution utilities shared across tools."""

from pathlib import Path


def resolve(file_path: str) -> Path:
    """Resolve *file_path* to an absolute ``Path``.

    Relative paths are resolved against ``Path.cwd()``.
    """
    return Path(file_path).resolve()
