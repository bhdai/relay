"""Tests for the checkpointer factory."""

import pytest

from relay.checkpointer.base import BaseCheckpointer
from relay.checkpointer.factory import create_checkpointer
from relay.checkpointer.impl.memory import MemoryCheckpointer
from relay.checkpointer.impl.sqlite import IndexedAsyncSqliteSaver


class TestCreateCheckpointer:
    """Tests for create_checkpointer factory function."""

    @pytest.mark.asyncio
    async def test_memory_backend_returns_memory_checkpointer(self):
        """The 'memory' backend should produce a MemoryCheckpointer."""
        async with create_checkpointer(backend="memory") as cp:
            assert isinstance(cp, MemoryCheckpointer)
            assert isinstance(cp, BaseCheckpointer)

    @pytest.mark.asyncio
    async def test_sqlite_backend_returns_indexed_saver(self, tmp_path):
        """The 'sqlite' backend should produce an IndexedAsyncSqliteSaver."""
        async with create_checkpointer(
            backend="sqlite", working_dir=str(tmp_path)
        ) as cp:
            assert isinstance(cp, IndexedAsyncSqliteSaver)
            assert isinstance(cp, BaseCheckpointer)

    @pytest.mark.asyncio
    async def test_memory_checkpointer_get_threads_empty(self):
        """A fresh MemoryCheckpointer should have no threads."""
        async with create_checkpointer(backend="memory") as cp:
            assert isinstance(cp, BaseCheckpointer)
            threads = await cp.get_threads()
            assert threads == set()

    @pytest.mark.asyncio
    async def test_sqlite_checkpointer_get_threads_empty(self, tmp_path):
        """A fresh SQLite checkpointer should have no threads."""
        async with create_checkpointer(
            backend="sqlite", working_dir=str(tmp_path)
        ) as cp:
            assert isinstance(cp, BaseCheckpointer)
            threads = await cp.get_threads()
            assert threads == set()
