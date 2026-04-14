"""Tests for relay.configs.registry — config loading and resolution."""

import shutil
from pathlib import Path

import pytest
import yaml

from relay.configs.agent import AgentConfig, SubAgentConfig
from relay.configs.registry import ConfigRegistry


def _write_subagent(config_dir: Path, name: str, **overrides) -> None:
    """Write a minimal subagent YAML to *config_dir*/subagents/."""
    subagents_dir = config_dir / "subagents"
    subagents_dir.mkdir(parents=True, exist_ok=True)
    data = {"name": name, "llm": "default", **overrides}
    (subagents_dir / f"{name}.yml").write_text(
        yaml.dump(data, default_flow_style=False)
    )


def _write_agent(config_dir: Path, name: str, **overrides) -> None:
    """Write a minimal agent YAML to *config_dir*/agents/."""
    agents_dir = config_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    data = {"name": name, "llm": "default", "default": True, **overrides}
    (agents_dir / f"{name}.yml").write_text(
        yaml.dump(data, default_flow_style=False)
    )


def _write_prompt(config_dir: Path, rel_path: str, content: str) -> None:
    """Write a prompt file at *config_dir*/*rel_path*."""
    prompt_file = config_dir / rel_path
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(content)


# ==============================================================================
# ensure_config_dir
# ==============================================================================


class TestEnsureConfigDir:
    async def test_copies_defaults_when_missing(self, tmp_path: Path):
        registry = ConfigRegistry(tmp_path)
        config_dir = tmp_path / ".relay"
        assert not config_dir.exists()

        await registry.ensure_config_dir()

        assert config_dir.exists()
        assert (config_dir / "agents").is_dir()
        assert (config_dir / "subagents").is_dir()

    async def test_idempotent_when_exists(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        config_dir.mkdir()
        (config_dir / "marker.txt").write_text("existing")

        registry = ConfigRegistry(tmp_path)
        await registry.ensure_config_dir()

        # Should not overwrite or error.
        assert (config_dir / "marker.txt").read_text() == "existing"


# ==============================================================================
# load_subagents
# ==============================================================================


class TestLoadSubagents:
    async def test_loads_from_yaml(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_subagent(config_dir, "explorer", description="Explores code")
        _write_subagent(config_dir, "worker", description="Does work")

        registry = ConfigRegistry(tmp_path)
        batch = await registry.load_subagents()

        assert len(batch.subagents) == 2
        assert batch.get_subagent("explorer") is not None
        assert batch.get_subagent("worker") is not None

    async def test_resolves_prompt_files(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_prompt(config_dir, "prompts/sub/explore.md", "Explore carefully.")
        _write_subagent(config_dir, "explorer", prompt="prompts/sub/explore.md")

        registry = ConfigRegistry(tmp_path)
        batch = await registry.load_subagents()

        explorer = batch.get_subagent("explorer")
        assert explorer is not None
        assert explorer.prompt == "Explore carefully."

    async def test_resolves_prompt_list(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_prompt(config_dir, "prompts/a.md", "Part A")
        _write_prompt(config_dir, "prompts/b.md", "Part B")
        _write_subagent(
            config_dir, "multi", prompt=["prompts/a.md", "prompts/b.md"]
        )

        registry = ConfigRegistry(tmp_path)
        batch = await registry.load_subagents()

        multi = batch.get_subagent("multi")
        assert multi is not None
        assert "Part A" in multi.prompt
        assert "Part B" in multi.prompt

    async def test_caches_results(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_subagent(config_dir, "explorer")

        registry = ConfigRegistry(tmp_path)
        first = await registry.load_subagents()
        second = await registry.load_subagents()

        assert first is second

    async def test_force_reload(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_subagent(config_dir, "explorer")

        registry = ConfigRegistry(tmp_path)
        first = await registry.load_subagents()
        second = await registry.load_subagents(force_reload=True)

        assert first is not second


# ==============================================================================
# load_agents
# ==============================================================================


class TestLoadAgents:
    async def test_loads_from_yaml(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_agent(config_dir, "general")

        registry = ConfigRegistry(tmp_path)
        batch = await registry.load_agents()

        assert len(batch.agents) == 1
        assert batch.agents[0].name == "general"

    async def test_resolves_prompt_files(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_prompt(config_dir, "prompts/main.md", "Be helpful.")
        _write_agent(config_dir, "general", prompt="prompts/main.md")

        registry = ConfigRegistry(tmp_path)
        batch = await registry.load_agents()

        agent = batch.get_agent("general")
        assert agent is not None
        assert agent.prompt == "Be helpful."

    async def test_validates_subagent_references(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_agent(config_dir, "general", subagents=["nonexistent"])

        registry = ConfigRegistry(tmp_path)
        with pytest.raises(ValueError, match="nonexistent"):
            await registry.load_agents()

    async def test_valid_subagent_references(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_subagent(config_dir, "explorer")
        _write_agent(config_dir, "general", subagents=["explorer"])

        registry = ConfigRegistry(tmp_path)
        batch = await registry.load_agents()

        agent = batch.get_agent("general")
        assert agent is not None
        assert agent.subagents == ["explorer"]


# ==============================================================================
# get_agent
# ==============================================================================


class TestGetAgent:
    async def test_returns_agent_by_name(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_agent(config_dir, "general")

        registry = ConfigRegistry(tmp_path)
        agent = await registry.get_agent("general")
        assert agent.name == "general"

    async def test_returns_default_when_none(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_agent(config_dir, "general", default=True)

        registry = ConfigRegistry(tmp_path)
        agent = await registry.get_agent(None)
        assert agent.name == "general"

    async def test_raises_for_unknown(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        _write_agent(config_dir, "general")

        registry = ConfigRegistry(tmp_path)
        with pytest.raises(ValueError, match="missing"):
            await registry.get_agent("missing")


# ==============================================================================
# Integration: loads from packaged defaults
# ==============================================================================


class TestPackagedDefaults:
    async def test_loads_default_configs(self, tmp_path: Path):
        """Verify the packaged defaults can be loaded successfully."""
        registry = ConfigRegistry(tmp_path)

        # This triggers ensure_config_dir and copies defaults.
        agents = await registry.load_agents()
        subagents = await registry.load_subagents()

        assert len(agents.agents) >= 1
        assert len(subagents.subagents) >= 2

        # Check the default agent has expected properties.
        default_agent = agents.get_default_agent()
        assert default_agent is not None
        assert default_agent.name == "general"
        assert default_agent.subagents is not None
        assert "explorer" in default_agent.subagents
        assert "general-purpose" in default_agent.subagents

        # Check subagent definitions exist.
        assert subagents.get_subagent("explorer") is not None
        assert subagents.get_subagent("general-purpose") is not None

    async def test_prompt_content_resolved(self, tmp_path: Path):
        """Verify that prompt file references are resolved into content."""
        registry = ConfigRegistry(tmp_path)
        agents = await registry.load_agents()

        default_agent = agents.get_default_agent()
        assert default_agent is not None
        # The prompt should be resolved text, not a file path list.
        assert isinstance(default_agent.prompt, str)
        assert len(default_agent.prompt) > 50  # should be substantial content


# ==============================================================================
# MCP config
# ==============================================================================


class TestMCPConfig:
    async def test_load_mcp_empty_when_no_file(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        config_dir.mkdir(parents=True)

        registry = ConfigRegistry(tmp_path)
        mcp = await registry.load_mcp()
        assert mcp.servers == {}

    async def test_load_mcp_parses_servers(self, tmp_path: Path):
        import json

        config_dir = tmp_path / ".relay"
        config_dir.mkdir(parents=True)
        mcp_json = config_dir / "mcp.json"
        mcp_json.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "my-server": {"command": "npx", "args": ["-y", "@my/srv"]}
                    }
                }
            )
        )

        registry = ConfigRegistry(tmp_path)
        mcp = await registry.load_mcp()
        assert "my-server" in mcp.servers

    async def test_load_mcp_cached(self, tmp_path: Path):
        config_dir = tmp_path / ".relay"
        config_dir.mkdir(parents=True)

        registry = ConfigRegistry(tmp_path)
        mcp1 = await registry.load_mcp()
        mcp2 = await registry.load_mcp()
        assert mcp1 is mcp2

    async def test_save_mcp(self, tmp_path: Path):
        import json

        from relay.mcp.config import MCPConfig as MCPConfigModel
        from relay.mcp.config import MCPServerConfig

        config_dir = tmp_path / ".relay"
        config_dir.mkdir(parents=True)

        registry = ConfigRegistry(tmp_path)
        config = MCPConfigModel(
            servers={"test-srv": MCPServerConfig(command="echo")}
        )
        await registry.save_mcp(config)

        # Verify file was written.
        mcp_json = config_dir / "mcp.json"
        assert mcp_json.exists()
        data = json.loads(mcp_json.read_text())
        assert "test-srv" in data["mcpServers"]

        # Verify cache was updated.
        loaded = await registry.load_mcp()
        assert loaded is config
