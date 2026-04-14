"""Bootstrap service — graph and checkpointer lifecycle.

``Initializer`` is the single place that wires together the
checkpointer, agent factory, and compiled graph.  The CLI session
(and any future entrypoint) asks the initializer for a ready-to-use
graph instead of constructing one itself.

This is a simplified version of langrepl's ``cli/bootstrap/initializer.py``.
Config registry, MCP, skills, and sandbox support will arrive in later
phases.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from relay.agents.factory import AgentFactory
from relay.checkpointer import create_checkpointer

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class Initializer:
    """Centralized service for initializing and managing agent resources.

    Owns:
    - Agent factory
    - Checkpointer lifecycle
    - Graph creation with cleanup

    Usage::

        init = Initializer()
        async with init.get_graph(backend="sqlite") as graph:
            # stream, invoke, etc.
            ...
    """

    def __init__(self, *, factory: AgentFactory | None = None) -> None:
        self.factory = factory or AgentFactory()

    async def create_graph(
        self,
        *,
        backend: str = "sqlite",
        working_dir: str | None = None,
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
        graph = self.factory.create(checkpointer=checkpointer)
        return graph, checkpointer_ctx

    @asynccontextmanager
    async def get_graph(
        self,
        *,
        backend: str = "sqlite",
        working_dir: str | None = None,
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
            yield self.factory.create(checkpointer=checkpointer)
