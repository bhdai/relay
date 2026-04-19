"""Tests for the checkpointer factory."""

import logging

import pytest

from relay.checkpointer.base import BaseCheckpointer
from relay.checkpointer.factory import create_checkpointer
from relay.checkpointer.impl.memory import MemoryCheckpointer
from relay.checkpointer.impl.sqlite import IndexedAsyncSqliteSaver
from relay.middlewares.permission import PermissionInterruptPayload


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

    @pytest.mark.asyncio
    async def test_serializer_allowlists_interrupt_payload(self, caplog):
        """Permission interrupt payloads should deserialize without unregistered-type warnings."""
        payload = PermissionInterruptPayload(
            request_id="req-1",
            question="Allow bash: git push --force?",
            permission="bash",
            patterns=["git push --force"],
            always_patterns=["git push *"],
        )

        with caplog.at_level(logging.WARNING):
            async with create_checkpointer(backend="memory") as cp:
                encoded = cp.serde.dumps_typed(payload)
                decoded = cp.serde.loads_typed(encoded)

        assert isinstance(decoded, PermissionInterruptPayload)
        assert decoded.question == "Allow bash: git push --force?"
        assert decoded.permission == "bash"


        assert isinstance(decoded, PermissionInterruptPayload)
        assert decoded.question == "Allow bash: git push --force?"
        assert decoded.permission == "bash"
        assert not any(
            "Deserializing unregistered type relay.middlewares.permission.PermissionInterruptPayload"
            in record.message
            for record in caplog.records
        )
