"""Central registry for loading, caching, and resolving configurations.

``ConfigRegistry`` is the single entry point for obtaining agent,
subagent, and MCP definitions.  It loads YAML files from either a
project-local config directory (``.relay/``) or the packaged defaults
that ship with relay.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from importlib.resources import files
from pathlib import Path

from relay.configs.agent import (
    AgentConfig,
    BatchAgentConfig,
    BatchSubAgentConfig,
    SubAgentConfig,
)
from relay.configs.llm import BatchLLMConfig, LLMConfig
from relay.configs.utils import load_prompt_content, load_yaml_dir
from relay.mcp.config import MCPConfig

logger = logging.getLogger(__name__)

# Directory name for project-local configs (e.g. ``$PROJECT/.relay/``).
CONFIG_DIR_NAME = ".relay"

# Subdirectory names inside the config dir.
CONFIG_AGENTS_DIR = Path(CONFIG_DIR_NAME) / "agents"
CONFIG_SUBAGENTS_DIR = Path(CONFIG_DIR_NAME) / "subagents"
CONFIG_LLMS_DIR = Path(CONFIG_DIR_NAME) / "llms"

# MCP config file (VS Code / Claude Code ``mcpServers`` format).
CONFIG_MCP_FILE = "mcp.json"

# MCP cache directory name.
CONFIG_MCP_CACHE_DIR = ".mcp-cache"

# Skills directory name.
CONFIG_SKILLS_DIR = "skills"

# Directory name for project-local configs (e.g. ``$PROJECT/.relay/``).
CONFIG_DIR_NAME = ".relay"

# Subdirectory names inside the config dir.
CONFIG_AGENTS_DIR = Path(CONFIG_DIR_NAME) / "agents"
CONFIG_SUBAGENTS_DIR = Path(CONFIG_DIR_NAME) / "subagents"


class ConfigRegistry:
    """Load, cache, and resolve agent and subagent configurations.

    Args:
        working_dir: Project root. The registry looks for a ``.relay/``
            directory here. If one does not exist, ``ensure_config_dir``
            copies the packaged defaults.
    """

    def __init__(self, working_dir: Path) -> None:
        self.working_dir = working_dir
        self.config_dir = working_dir / CONFIG_DIR_NAME

        # Lazy caches — populated by the first call to load_*.
        self._agents: BatchAgentConfig | None = None
        self._subagents: BatchSubAgentConfig | None = None
        self._llms: BatchLLMConfig | None = None
        self._mcp: MCPConfig | None = None

    # ==================================================================
    # Setup
    # ==================================================================

    async def ensure_config_dir(self) -> None:
        template_dir = Path(str(files("relay.resources") / "configs" / "default"))

        if not self.config_dir.exists():
            await asyncio.to_thread(shutil.copytree, template_dir, self.config_dir)
            logger.info("Created config directory from defaults: %s", self.config_dir)
            return

        copied_files = await asyncio.to_thread(
            self._copy_missing_default_files,
            template_dir,
            self.config_dir,
        )
        if copied_files:
            logger.info(
                "Backfilled %d missing config files into %s",
                copied_files,
                self.config_dir,
            )

    @staticmethod
    def _copy_missing_default_files(template_dir: Path, target_dir: Path) -> int:
        """Copy only missing top-level default entries into an existing config dir."""
        copied_files = 0

        for source in template_dir.iterdir():
            destination = target_dir / source.name

            if destination.exists():
                continue

            if source.is_dir():
                shutil.copytree(source, destination)
                copied_files += sum(1 for child in source.rglob("*") if child.is_file())
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied_files += 1

        return copied_files

    # ==================================================================
    # LLM configs
    # ==================================================================

    async def load_llms(
        self,
        *,
        force_reload: bool = False,
    ) -> BatchLLMConfig:
        """Load all LLM configs (cached)."""
        if self._llms is not None and not force_reload:
            return self._llms

        await self.ensure_config_dir()

        raw_items = await load_yaml_dir(self.config_dir / "llms")
        llms = [LLMConfig(**item) for item in raw_items]
        self._llms = BatchLLMConfig(llms=llms)
        return self._llms

    async def get_llm(self, alias: str) -> LLMConfig | None:
        llms = await self.load_llms()
        return llms.get_llm(alias)

    # ==================================================================
    # Subagent configs
    # ==================================================================

    async def load_subagents(
        self,
        *,
        force_reload: bool = False,
    ) -> BatchSubAgentConfig:
        """Load all subagent configs (cached)."""
        if self._subagents is not None and not force_reload:
            return self._subagents

        await self.ensure_config_dir()
        llms = await self.load_llms()

        raw_items = await load_yaml_dir(self.config_dir / "subagents")

        # Resolve prompt file references.
        resolved: list[SubAgentConfig] = []
        for item in raw_items:
            if prompt_ref := item.get("prompt"):
                item["prompt"] = await load_prompt_content(self.config_dir, prompt_ref)

            llm_alias = item.get("llm", "default")
            if llm_alias != "default" and llms.get_llm(llm_alias) is None:
                raise ValueError(
                    f"Subagent '{item.get('name', '?')}' references unknown "
                    f"llm '{llm_alias}'.  Available: {llms.llm_names}"
                )

            resolved.append(SubAgentConfig(**item))

        self._subagents = BatchSubAgentConfig(subagents=resolved)
        return self._subagents

    async def get_subagent(self, name: str) -> SubAgentConfig | None:
        subagents = await self.load_subagents()
        return subagents.get_subagent(name)

    # ==================================================================
    # Agent configs
    # ==================================================================

    async def load_agents(
        self,
        *,
        force_reload: bool = False,
    ) -> BatchAgentConfig:
        """Load all agent configs with resolved references (cached)."""
        if self._agents is not None and not force_reload:
            return self._agents

        await self.ensure_config_dir()
        llms = await self.load_llms()

        raw_items = await load_yaml_dir(self.config_dir / "agents")

        # Pre-load subagents so we can validate references.
        subagents = await self.load_subagents()

        resolved: list[AgentConfig] = []
        for item in raw_items:
            # Resolve prompt file paths into content.
            if prompt_ref := item.get("prompt"):
                item["prompt"] = await load_prompt_content(self.config_dir, prompt_ref)

            llm_alias = item.get("llm", "default")
            if llm_alias != "default" and llms.get_llm(llm_alias) is None:
                raise ValueError(
                    f"Agent '{item.get('name', '?')}' references unknown "
                    f"llm '{llm_alias}'.  Available: {llms.llm_names}"
                )

            # Validate subagent references.
            if subagent_names := item.get("subagents"):
                for sa_name in subagent_names:
                    if subagents.get_subagent(sa_name) is None:
                        raise ValueError(
                            f"Agent '{item.get('name', '?')}' references unknown "
                            f"subagent '{sa_name}'.  "
                            f"Available: {subagents.subagent_names}"
                        )

            resolved.append(AgentConfig(**item))

        self._agents = BatchAgentConfig(agents=resolved)
        return self._agents

    async def get_agent(self, name: str | None = None) -> AgentConfig:
        """Return agent by *name*, or the default agent.

        Raises:
            ValueError: If the requested agent cannot be found.
        """
        agents = await self.load_agents()
        agent = agents.get_agent(name)
        if agent is not None:
            return agent
        raise ValueError(
            f"Agent '{name}' not found.  Available: {agents.agent_names}"
        )

    # ==================================================================
    # MCP configs
    # ==================================================================

    async def load_mcp(
        self,
        *,
        force_reload: bool = False,
    ) -> MCPConfig:
        """Load MCP configuration from ``.relay/mcp.json`` (cached).

        Returns:
            The loaded MCP config. Returns an empty ``MCPConfig`` if the file
            does not exist.
        """
        if self._mcp is not None and not force_reload:
            return self._mcp

        mcp_path = self.config_dir / CONFIG_MCP_FILE
        self._mcp = await MCPConfig.from_json(mcp_path)
        return self._mcp

    async def save_mcp(self, config: MCPConfig) -> None:
        """Write MCP configuration back to ``.relay/mcp.json``."""
        import json

        mcp_path = self.config_dir / CONFIG_MCP_FILE
        await asyncio.to_thread(mcp_path.parent.mkdir, parents=True, exist_ok=True)

        mcp_servers = {}
        for name, server_config in config.servers.items():
            mcp_servers[name] = server_config.model_dump()

        data = {"mcpServers": mcp_servers}
        content = json.dumps(data, indent=2)
        await asyncio.to_thread(mcp_path.write_text, content)
        self._mcp = config
