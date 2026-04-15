"""Bootstrap service — graph and checkpointer lifecycle.

``Initializer`` is the single place that wires together the
checkpointer, agent factory, config registry, MCP client, skill
factory, and compiled graph.  The CLI session (and any future
entrypoint) asks the initializer for a ready-to-use graph instead
of constructing one itself.

When a ``ConfigRegistry`` is available the initializer uses the async
``create_from_config`` path so that agent definitions come from YAML
files.  Otherwise it falls back to the hardcoded factory path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from relay.agents.factory import AgentFactory
from relay.checkpointer import create_checkpointer
from relay.configs.registry import (
    CONFIG_MCP_CACHE_DIR,
    CONFIG_SKILLS_DIR,
    ConfigRegistry,
)
from relay.mcp.factory import MCPFactory
from relay.skills.factory import SkillFactory

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class Initializer:
    """Centralized service for initializing and managing agent resources.

    Owns:
    - Config registry (optional — enables data-driven agent assembly)
    - Agent factory
    - MCP factory (optional — loads MCP tools when config exists)
    - Skill factory (optional — loads skills when directory exists)
    - Checkpointer lifecycle
    - Graph creation with cleanup

    Usage::

        init = Initializer()
        async with init.get_graph(backend="sqlite") as graph:
            # stream, invoke, etc.
            ...
    """

    def __init__(
        self,
        *,
        factory: AgentFactory | None = None,
        registry: ConfigRegistry | None = None,
        working_dir: Path | None = None,
        model_name: str | None = None,
        mcp_factory: MCPFactory | None = None,
        skill_factory: SkillFactory | None = None,
    ) -> None:
        # If a working_dir is provided but no registry, create one.
        if registry is None and working_dir is not None:
            registry = ConfigRegistry(working_dir)

        self.registry = registry
        self.skill_factory = skill_factory or SkillFactory()
        self.mcp_factory = mcp_factory or MCPFactory()
        self.factory = factory or AgentFactory(
            registry=registry,
            model_name=model_name,
            skill_factory=self.skill_factory,
        )

    async def create_graph(
        self,
        *,
        backend: str = "sqlite",
        working_dir: str | None = None,
        agent_name: str | None = None,
    ) -> tuple[CompiledStateGraph, Callable[[], Awaitable[None]]]:
        """Create a graph with its checkpointer and MCP contexts.

        Returns ``(graph, cleanup_fn)`` where *cleanup_fn* is an async
        callable that the caller must await when done (closes MCP
        sessions and the checkpointer).
        """
        if self.registry is not None:
            await self.registry.get_agent(agent_name)

        checkpointer_ctx = create_checkpointer(
            backend=backend,
            working_dir=working_dir,
        )
        checkpointer = await checkpointer_ctx.__aenter__()

        # ----------------------------------------------------------
        # MCP (optional)
        # ----------------------------------------------------------
        mcp_client = None
        if self.registry is not None:
            mcp_config = await self.registry.load_mcp()
            if mcp_config.servers:
                wd = Path(working_dir) if working_dir else self.registry.working_dir
                mcp_client = await self.mcp_factory.create(
                    config=mcp_config,
                    cache_dir=wd / CONFIG_MCP_CACHE_DIR,
                )

        # ----------------------------------------------------------
        # Skills directory
        # ----------------------------------------------------------
        skills_dir: Path | None = None
        if self.registry is not None:
            candidate = self.registry.config_dir / CONFIG_SKILLS_DIR
            if candidate.exists():
                skills_dir = candidate

        # ----------------------------------------------------------
        # Build graph
        # ----------------------------------------------------------
        if self.registry is not None:
            graph = await self.factory.create_from_config(
                checkpointer=checkpointer,
                agent_name=agent_name,
                mcp_client=mcp_client,
                skills_dir=skills_dir,
            )
        else:
            graph = self.factory.create(checkpointer=checkpointer)

        async def cleanup() -> None:
            if mcp_client is not None:
                await mcp_client.close()
            await checkpointer_ctx.__aexit__(None, None, None)

        return graph, cleanup

    @asynccontextmanager
    async def get_graph(
        self,
        *,
        backend: str = "sqlite",
        working_dir: str | None = None,
        agent_name: str | None = None,
    ) -> AsyncIterator[CompiledStateGraph]:
        """Context manager that yields a compiled graph with checkpointer.

        The checkpointer and MCP sessions are opened on entry and
        closed on exit::

            async with initializer.get_graph() as graph:
                result = await graph.ainvoke(...)
        """
        graph, cleanup = await self.create_graph(
            backend=backend,
            working_dir=working_dir,
            agent_name=agent_name,
        )
        try:
            yield graph
        finally:
            await cleanup()
