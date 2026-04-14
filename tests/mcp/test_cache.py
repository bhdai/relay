"""Tests for relay.mcp.cache — disk-based tool schema cache."""

import json
from pathlib import Path

import pytest

from relay.mcp.cache import MCP_CACHE_VERSION, MCPCache
from relay.tools.schema import ToolSchema


@pytest.fixture()
def sample_schemas() -> list[ToolSchema]:
    return [
        ToolSchema(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        ToolSchema(
            name="write_file",
            description="Write a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    ]


class TestMCPCache:
    async def test_save_and_load_round_trip(
        self, tmp_path: Path, sample_schemas: list[ToolSchema]
    ):
        cache = MCPCache(dir=tmp_path, hashes={"server_a": "hash123"})

        await cache.save("server_a", sample_schemas)
        loaded = await cache.load("server_a")

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].name == "read_file"
        assert loaded[1].name == "write_file"

    async def test_load_returns_none_when_missing(self, tmp_path: Path):
        cache = MCPCache(dir=tmp_path, hashes={})
        result = await cache.load("nonexistent_server")
        assert result is None

    async def test_load_invalidates_on_hash_mismatch(
        self, tmp_path: Path, sample_schemas: list[ToolSchema]
    ):
        # Save with one hash.
        cache = MCPCache(dir=tmp_path, hashes={"server_a": "hash_v1"})
        await cache.save("server_a", sample_schemas)

        # Load with different expected hash.
        cache2 = MCPCache(dir=tmp_path, hashes={"server_a": "hash_v2"})
        result = await cache2.load("server_a")
        assert result is None

    async def test_load_invalidates_on_version_mismatch(
        self, tmp_path: Path, sample_schemas: list[ToolSchema]
    ):
        # Write cache file with wrong version.
        cache_path = tmp_path / "server_a.json"
        data = {
            "version": "0.0.0",
            "hash": "hash123",
            "tools": [s.model_dump() for s in sample_schemas],
        }
        cache_path.write_text(json.dumps(data))

        cache = MCPCache(dir=tmp_path, hashes={"server_a": "hash123"})
        result = await cache.load("server_a")
        assert result is None

    async def test_no_dir_returns_none(self, sample_schemas: list[ToolSchema]):
        cache = MCPCache(dir=None, hashes={})
        # Should not raise.
        result = await cache.load("server_a")
        assert result is None

    async def test_no_dir_save_is_noop(self, sample_schemas: list[ToolSchema]):
        cache = MCPCache(dir=None, hashes={})
        # Should not raise.
        await cache.save("server_a", sample_schemas)

    async def test_load_handles_corrupt_json(self, tmp_path: Path):
        cache_path = tmp_path / "server_a.json"
        cache_path.write_text("not-json")

        cache = MCPCache(dir=tmp_path, hashes={})
        result = await cache.load("server_a")
        assert result is None
