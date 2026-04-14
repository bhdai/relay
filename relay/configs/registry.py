"""Central registry for loading, caching, and resolving configurations.

``ConfigRegistry`` is the single entry point for obtaining agent and
subagent definitions.  It loads YAML files from either a project-local
config directory (``.relay/``) or the packaged defaults that ship with
relay.

Compared to langrepl's registry this is intentionally minimal:

- No LLM config files — relay still resolves LLMs from env settings.
- No checkpointer config files — relay's checkpointer factory is
  separate.
- No sandbox, MCP, approval, skills, or server configs.

Those will arrive in later phases.
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
from relay.configs.utils import load_prompt_content, load_yaml_dir

logger = logging.getLogger(__name__)

# Directory name for project-local configs (e.g. ``$PROJECT/.relay/``).
CONFIG_DIR_NAME = ".relay"

# Subdirectory names inside the config dir.
CONFIG_AGENTS_DIR = Path(CONFIG_DIR_NAME) / "agents"
CONFIG_SUBAGENTS_DIR = Path(CONFIG_DIR_NAME) / "subagents"


class ConfigRegistry:
    """Load, cache, and resolve agent and subagent configurations.

    Parameters
    ----------
    working_dir:
        Project root.  The registry looks for a ``.relay/`` directory
        here.  If one does not exist, ``ensure_config_dir`` copies the
        packaged defaults.
    """

    def __init__(self, working_dir: Path) -> None:
        self.working_dir = working_dir
        self.config_dir = working_dir / CONFIG_DIR_NAME

        # Lazy caches — populated by the first call to load_*.
        self._agents: BatchAgentConfig | None = None
        self._subagents: BatchSubAgentConfig | None = None

    # ==================================================================
    # Setup
    # ==================================================================

    async def ensure_config_dir(self) -> None:
        """Create ``.relay/`` from packaged defaults if it does not exist."""
        if self.config_dir.exists():
            return

        template_dir = Path(str(files("relay.resources") / "configs" / "default"))
        await asyncio.to_thread(shutil.copytree, template_dir, self.config_dir)
        logger.info("Created config directory from defaults: %s", self.config_dir)

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

        raw_items = await load_yaml_dir(self.config_dir / "subagents")

        # Resolve prompt file references.
        resolved: list[SubAgentConfig] = []
        for item in raw_items:
            if prompt_ref := item.get("prompt"):
                item["prompt"] = await load_prompt_content(self.config_dir, prompt_ref)
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

        raw_items = await load_yaml_dir(self.config_dir / "agents")

        # Pre-load subagents so we can validate references.
        subagents = await self.load_subagents()

        resolved: list[AgentConfig] = []
        for item in raw_items:
            # Resolve prompt file paths into content.
            if prompt_ref := item.get("prompt"):
                item["prompt"] = await load_prompt_content(self.config_dir, prompt_ref)

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

        Raises ``ValueError`` if not found.
        """
        agents = await self.load_agents()
        agent = agents.get_agent(name)
        if agent is not None:
            return agent
        raise ValueError(
            f"Agent '{name}' not found.  Available: {agents.agent_names}"
        )
