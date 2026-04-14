"""Bootstrap service — graph and checkpointer lifecycle.

``Initializer`` is the single place that wires together the
checkpointer, agent factory, config registry, and compiled graph.
The CLI session (and any future entrypoint) asks the initializer for
a ready-to-use graph instead of constructing one itself.

When a ``ConfigRegistry`` is available the initializer uses the async
``create_from_config`` path so that agent definitions come from YAML
files.  Otherwise it falls back to the hardcoded factory path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from relay.agents.factory import AgentFactory
from relay.checkpointer import create_checkpointer
from relay.configs.registry import ConfigRegistry

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class Initializer:
    """Centralized service for initializing and managing agent resources.

    Owns:
    - Config registry (optional — enables data-driven agent assembly)
    - Agent factory
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
    ) -> None:
        # If a working_dir is provided but no registry, create one.
        if registry is None and working_dir is not None:
            registry = ConfigRegistry(working_dir)

        self.registry = registry
        self.factory = factory or AgentFactory(registry=registry)

    async def create_graph(
        self,
        *,
        backend: str = "sqlite",
        working_dir: str | None = None,
        agent_name: str | None = None,
    ) -> tuple[CompiledStateGraph, AsyncIterator]:
        """Create a graph with its checkpointer context.

        Returns ``(graph, checkpointer_ctx)`` where *checkpointer_ctx*
        is an async context manager that the caller must ``__aexit__``
        when done.  For a simpler API, use :meth:`get_graph` instead.
        """
        checkpointer_ctx = create_checkpointer(
            backend=backend,
            working_dir=working_dir,
        )
        checkpointer = await checkpointer_ctx.__aenter__()

        if self.registry is not None:
            graph = await self.factory.create_from_config(
                checkpointer=checkpointer,
                agent_name=agent_name,
            )
        else:
            graph = self.factory.create(checkpointer=checkpointer)

        return graph, checkpointer_ctx

    @asynccontextmanager
    async def get_graph(
        self,
        *,
        backend: str = "sqlite",
        working_dir: str | None = None,
        agent_name: str | None = None,
    ) -> AsyncIterator[CompiledStateGraph]:
        """Context manager that yields a compiled graph with checkpointer.

        The checkpointer is opened on entry and closed on exit::

            async with initializer.get_graph() as graph:
                result = await graph.ainvoke(...)
        """
        async with create_checkpointer(
            backend=backend,
            working_dir=working_dir,
        ) as checkpointer:
            if self.registry is not None:
                yield await self.factory.create_from_config(
                    checkpointer=checkpointer,
                    agent_name=agent_name,
                )
            else:
                yield self.factory.create(checkpointer=checkpointer)
