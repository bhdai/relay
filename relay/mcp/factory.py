"""MCP factory — builds an ``MCPClient`` from configuration.

``MCPFactory`` reads ``MCPConfig`` and produces a fully-configured
``MCPClient`` with server connections, tool filters, metadata, and
cache settings.  Server config hashes are used for cache invalidation.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_mcp_adapters.sessions import Connection

from relay.mcp.client import MCPClient, RepairConfig, ServerMeta
from relay.mcp.config import MCPTransport

if TYPE_CHECKING:
    from relay.mcp.config import MCPConfig, MCPServerConfig

logger = logging.getLogger(__name__)


class MCPFactory:
    def __init__(
        self,
        enable_approval: bool = True,
    ):
        self.enable_approval = enable_approval
        self._client: MCPClient | None = None
        self._config_hash: int | None = None

    @staticmethod
    def _compute_server_hash(server: MCPServerConfig) -> str:
        """Compute a hash of server config for cache invalidation."""
        signature: dict[str, Any] = {
            "enabled": server.enabled,
            "transport": server.transport,
            "command": server.command,
            "args": tuple(server.args or []),
            "url": server.url,
            "headers": tuple(sorted((server.headers or {}).items())) if server.headers else (),
            "env": tuple(sorted(server.env.items())) if server.env else (),
            "include": tuple(server.include or []),
            "exclude": tuple(server.exclude or []),
            "repair_command": tuple(server.repair_command or []),
            "repair_timeout": server.repair_timeout,
            "stateful": server.stateful,
            "invoke_timeout": server.invoke_timeout,
        }
        return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()

    @classmethod
    def _get_config_hash(cls, config: MCPConfig, cache_dir: Path | None) -> int:
        server_hashes = tuple(
            sorted(
                (name, cls._compute_server_hash(server))
                for name, server in config.servers.items()
            )
        )
        return hash((server_hashes, str(cache_dir) if cache_dir else None))

    async def create(
        self,
        config: MCPConfig,
        cache_dir: Path | None = None,
    ) -> MCPClient:
        """Create or return a cached ``MCPClient`` from *config*.

        If the config has not changed since the last call the existing
        client is returned.
        """
        config_hash = self._get_config_hash(config, cache_dir)
        if self._client and self._config_hash == config_hash:
            return self._client

        server_config: dict[str, Connection] = {}
        tool_filters: dict[str, dict] = {}
        server_metadata: dict[str, ServerMeta] = {}

        for name, server in config.servers.items():
            if not server.enabled:
                continue

            env = dict(server.env) if server.env else {}

            server_dict: Connection | None = None

            if server.transport == MCPTransport.STDIO:
                if server.command:
                    server_dict = {
                        "transport": server.transport.value,
                        "command": server.command,
                        "args": server.args or [],
                        "env": env,
                    }
            elif server.transport.is_http:
                if server.url:
                    server_dict = {
                        "transport": server.transport.value,
                        "url": server.url,
                        "headers": server.headers,
                    }
                    if server.timeout is not None:
                        server_dict["timeout"] = server.timeout
                    if server.sse_read_timeout is not None:
                        server_dict["sse_read_timeout"] = server.sse_read_timeout

            if server_dict is None:
                continue

            if server.transport == MCPTransport.STDIO:
                logger.debug(
                    "MCP server '%s': transport=%s, command=%s %s",
                    name,
                    server.transport.value,
                    server.command,
                    " ".join(server.args or []),
                )
            else:
                logger.debug(
                    "MCP server '%s': transport=%s, url=%s",
                    name,
                    server.transport.value,
                    server.url,
                )
            server_config[name] = server_dict

            if server.include or server.exclude:
                tool_filters[name] = {
                    "include": server.include,
                    "exclude": server.exclude,
                }

            repair = None
            if server.repair_command:
                repair = RepairConfig(server.repair_command, server.repair_timeout)

            server_metadata[name] = ServerMeta(
                hash=self._compute_server_hash(server),
                stateful=server.stateful,
                invoke_timeout=server.invoke_timeout,
                repair=repair,
            )

        self._client = MCPClient(
            server_config,
            tool_filters,
            enable_approval=self.enable_approval,
            cache_dir=cache_dir,
            server_metadata=server_metadata,
        )
        self._config_hash = config_hash
        return self._client
