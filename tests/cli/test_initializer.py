"""Tests for initializer startup ordering and failure behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from relay.cli.bootstrap.initializer import Initializer


class _EmptyMcpConfig:
    servers: dict = {}


class _FailingRegistry:
    def __init__(self, working_dir: Path) -> None:
        self.working_dir = working_dir
        self.config_dir = working_dir / ".relay"

    async def get_agent(self, name: str | None):
        raise ValueError(f"Agent '{name}' not found.  Available: []")

    async def load_mcp(self):
        return _EmptyMcpConfig()


@pytest.mark.asyncio
async def test_create_graph_validates_agent_before_opening_checkpointer(
    monkeypatch, tmp_path: Path
) -> None:
    def _unexpected_checkpointer(**kwargs):
        raise AssertionError("checkpointer should not be opened for invalid agent")

    monkeypatch.setattr(
        "relay.cli.bootstrap.initializer.create_checkpointer",
        _unexpected_checkpointer,
    )

    initializer = Initializer(registry=_FailingRegistry(tmp_path))

    with pytest.raises(ValueError, match="claude-style-coder"):
        await initializer.create_graph(agent_name="claude-style-coder")