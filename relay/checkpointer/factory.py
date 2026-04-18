"""Checkpointer factory for conversation persistence.

Two backends:

- **SQLite** (default): Persists conversation threads to a local
  ``.relay/checkpoints.db`` file via ``IndexedAsyncSqliteSaver``.
  Includes automatic message indexing for fast thread/history queries.
- **Memory**: In-memory ``MemoryCheckpointer`` — useful for testing
  or ephemeral sessions.

Both backends implement :class:`relay.checkpointer.base.BaseCheckpointer`
so the CLI can call ``get_threads()``, ``get_history()``, etc.

Usage::

    async with create_checkpointer() as cp:
        graph = build_graph(checkpointer=cp)
        ...
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from relay.checkpointer.base import BaseCheckpointer
from relay.checkpointer.impl.memory import MemoryCheckpointer
from relay.checkpointer.impl.sqlite import IndexedAsyncSqliteSaver

# Default storage directory, relative to the working directory.
_RELAY_DIR = ".relay"
_DB_FILENAME = "checkpoints.db"

# Keep serializer allowlist in one place so both memory/sqlite
# backends deserialize approval interrupts without warnings.
_ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    ("relay.middlewares.approval", "InterruptPayload"),
]


def _build_checkpoint_serializer() -> JsonPlusSerializer:
    """Create a JsonPlus serializer with relay-specific msgpack allowlist."""
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


def _ensure_db_path(working_dir: str | None = None) -> str:
    """Return the path to the SQLite database, creating the directory."""
    base = Path(working_dir) if working_dir else Path.cwd()
    relay_dir = base / _RELAY_DIR
    relay_dir.mkdir(parents=True, exist_ok=True)

    # Exclude from git if a repo root is found.
    _add_to_gitignore(base, _RELAY_DIR)

    return str(relay_dir / _DB_FILENAME)


def _add_to_gitignore(repo_root: Path, entry: str) -> None:
    """Append *entry* to ``.gitignore`` if it is not already listed."""
    gitignore = repo_root / ".gitignore"
    line = f"{entry}/\n"

    if gitignore.exists():
        content = gitignore.read_text()
        if entry in content:
            return
        # Ensure file ends with a newline before appending.
        if content and not content.endswith("\n"):
            line = "\n" + line
    try:
        with gitignore.open("a") as f:
            f.write(line)
    except OSError:
        # Not critical — skip silently if we can't write.
        pass


@asynccontextmanager
async def create_checkpointer(
    *,
    backend: str = "sqlite",
    working_dir: str | None = None,
) -> AsyncIterator[BaseCheckpointer]:
    """Create a checkpointer as an async context manager.

    Parameters
    ----------
    backend:
        ``"sqlite"`` (default) or ``"memory"``.
    working_dir:
        Base directory for the ``.relay/`` storage folder.
        Defaults to ``os.getcwd()``.
    """
    if backend == "memory":
        yield MemoryCheckpointer(serde=_build_checkpoint_serializer())
        return

    db_path = _ensure_db_path(working_dir)
    async with IndexedAsyncSqliteSaver.create(
        connection_string=db_path,
        serde=_build_checkpoint_serializer(),
    ) as saver:
        yield saver
