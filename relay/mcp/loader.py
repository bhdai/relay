"""MCP tool loading with repair command support.

The loader abstracts the difference between stateless servers (create
a fresh connection per load) and stateful servers (reuse a persistent
session).  If the initial load fails with an MCP-specific error and
a repair command is configured, the loader runs the repair command
and retries once.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.exceptions import McpError

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from mcp import ClientSession

    from relay.mcp.client import RepairConfig

logger = logging.getLogger(__name__)


def _is_mcp_error(exc: Exception) -> bool:
    """Check if exception is MCP-related (supports ExceptionGroups)."""
    if isinstance(exc, McpError):
        return True
    if isinstance(exc, ExceptionGroup):
        return any(_is_mcp_error(e) for e in exc.exceptions)
    return False


class MCPLoader:
    """Loads tools from MCP servers with repair on failure."""

    def __init__(
        self,
        get_tools: Callable[[str], Awaitable[list[BaseTool]]],
        get_session: Callable[[str], Awaitable[ClientSession]],
        close_session: Callable[[str], Awaitable[None]],
        repairs: dict[str, RepairConfig] | None = None,
    ) -> None:
        self._get_tools = get_tools
        self._get_session = get_session
        self._close_session = close_session
        self._repairs = repairs or {}

    async def stateless(self, server: str) -> list[BaseTool]:
        """Load tools from stateless server."""
        return await self._with_repair(server, lambda: self._get_tools(server))

    async def stateful(self, server: str) -> list[BaseTool]:
        """Load tools from stateful server via persistent session."""

        async def load() -> list[BaseTool]:
            session = await self._get_session(server)
            return list(await load_mcp_tools(session))

        return await self._with_repair(server, load)

    async def _with_repair(
        self,
        server: str,
        load: Callable[[], Awaitable[list[BaseTool]]],
    ) -> list[BaseTool]:
        """Execute load with optional repair on MCP error."""
        try:
            return list(await load())
        except Exception as e:
            repair = self._repairs.get(server)
            if _is_mcp_error(e) and repair:
                await self._close_session(server)
                await self._run_repair(repair)
                return list(await load())
            logger.error("Failed to load tools from %s: %s", server, e, exc_info=True)
            return []

    @staticmethod
    async def _run_repair(repair: RepairConfig) -> None:
        """Run a repair command as a subprocess."""
        proc = await asyncio.create_subprocess_exec(
            *repair.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=repair.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise
