"""MCP (Model Context Protocol) server configuration models.

Defines the Pydantic models for MCP server definitions.  Configuration
is loaded from a JSON file (typically ``.relay/mcp.json``) that follows
the VS Code ``mcpServers`` format.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiofiles
from pydantic import BaseModel, Field, field_validator, model_validator


class MCPTransport(StrEnum):
    """MCP transport types."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"
    WEBSOCKET = "websocket"

    @property
    def is_http(self) -> bool:
        """HTTP-based transports that cannot be sandboxed."""
        return self in {MCPTransport.SSE, MCPTransport.HTTP, MCPTransport.WEBSOCKET}


class MCPServerConfig(BaseModel):
    command: str | None = Field(
        default=None, description="The command to execute the server"
    )
    url: str | None = Field(default=None, description="The URL of the server")
    headers: dict[str, str] | None = Field(
        default=None, description="Headers for the server connection"
    )
    args: list[str] = Field(
        default_factory=list, description="Arguments for the server command"
    )
    transport: MCPTransport = Field(
        default=MCPTransport.STDIO, description="Transport protocol"
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    include: list[str] = Field(default_factory=list, description="Tools to include")
    exclude: list[str] = Field(default_factory=list, description="Tools to exclude")
    enabled: bool = Field(default=True, description="Whether the server is enabled")
    repair_command: list[str] | None = Field(
        default=None,
        description="Command list to run if server initialization fails",
    )
    repair_timeout: int | None = Field(
        default=None, description="Repair command timeout in seconds"
    )
    stateful: bool = Field(
        default=False,
        description="Keep server connection alive between tool calls",
    )
    timeout: float | None = Field(
        default=None, description="Connection timeout in seconds"
    )
    sse_read_timeout: float | None = Field(
        default=None, description="SSE read timeout in seconds"
    )
    invoke_timeout: float | None = Field(
        default=None, description="Tool invocation timeout in seconds"
    )

    @field_validator("transport", mode="before")
    @classmethod
    def normalize_transport(cls, v: str) -> str:
        """Normalize http transport aliases to canonical 'http'."""
        if v in {"streamable_http", "streamable-http"}:
            return "http"
        return v

    @model_validator(mode="after")
    def validate_repair_timeout(self) -> MCPServerConfig:
        """Set default repair_timeout when repair_command is set."""
        if self.repair_command and self.repair_timeout is None:
            self.repair_timeout = 30
        return self


class MCPConfig(BaseModel):
    servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="MCP server configurations"
    )

    @classmethod
    async def from_json(
        cls, path: Path, context: dict[str, Any] | None = None
    ) -> MCPConfig:
        """Load MCP configuration from JSON file.

        Reads a ``mcpServers`` JSON file (VS Code / Claude Code format)
        and returns an ``MCPConfig`` with parsed server definitions.
        Returns an empty config if the file does not exist.
        """
        if not path.exists():
            return cls()

        async with aiofiles.open(path) as f:
            config_content = await f.read()

        config: dict[str, Any] = json.loads(config_content)
        mcp_servers = config.get("mcpServers", {})

        servers = {}
        for name, server_config in mcp_servers.items():
            servers[name] = MCPServerConfig(**server_config)

        return cls(servers=servers)
