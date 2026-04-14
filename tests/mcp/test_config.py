"""Tests for relay.mcp.config — MCP configuration models."""

import json
from pathlib import Path

import pytest

from relay.mcp.config import MCPConfig, MCPServerConfig, MCPTransport


# ==============================================================================
# MCPTransport
# ==============================================================================


class TestMCPTransport:
    def test_is_http_true_for_http_based(self):
        assert MCPTransport.SSE.is_http
        assert MCPTransport.HTTP.is_http
        assert MCPTransport.WEBSOCKET.is_http

    def test_is_http_false_for_stdio(self):
        assert not MCPTransport.STDIO.is_http


# ==============================================================================
# MCPServerConfig
# ==============================================================================


class TestMCPServerConfig:
    def test_minimal_stdio_server(self):
        cfg = MCPServerConfig(command="npx", args=["-y", "my-server"])
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "my-server"]
        assert cfg.transport == MCPTransport.STDIO
        assert cfg.enabled is True
        assert cfg.stateful is False

    def test_http_server_with_url(self):
        cfg = MCPServerConfig(
            url="http://localhost:3000",
            transport="http",
        )
        assert cfg.url == "http://localhost:3000"
        assert cfg.transport == MCPTransport.HTTP

    def test_normalize_streamable_http(self):
        cfg = MCPServerConfig(transport="streamable_http")
        assert cfg.transport == MCPTransport.HTTP

        cfg2 = MCPServerConfig(transport="streamable-http")
        assert cfg2.transport == MCPTransport.HTTP

    def test_repair_defaults_timeout_when_command_set(self):
        cfg = MCPServerConfig(repair_command=["npm", "install"])
        assert cfg.repair_timeout == 30

    def test_repair_timeout_not_defaulted_without_command(self):
        cfg = MCPServerConfig()
        assert cfg.repair_timeout is None

    def test_include_exclude_filters(self):
        cfg = MCPServerConfig(
            command="server",
            include=["tool_a", "tool_b"],
        )
        assert cfg.include == ["tool_a", "tool_b"]
        assert cfg.exclude == []


# ==============================================================================
# MCPConfig
# ==============================================================================


class TestMCPConfig:
    async def test_from_json_empty_when_missing(self, tmp_path: Path):
        cfg = await MCPConfig.from_json(tmp_path / "nonexistent.json")
        assert cfg.servers == {}

    async def test_from_json_parses_servers(self, tmp_path: Path):
        mcp_json = tmp_path / "mcp.json"
        data = {
            "mcpServers": {
                "my-server": {
                    "command": "npx",
                    "args": ["-y", "@my/server"],
                    "transport": "stdio",
                },
                "http-server": {
                    "url": "http://localhost:3000",
                    "transport": "http",
                },
            }
        }
        mcp_json.write_text(json.dumps(data))

        cfg = await MCPConfig.from_json(mcp_json)
        assert len(cfg.servers) == 2
        assert cfg.servers["my-server"].command == "npx"
        assert cfg.servers["http-server"].transport == MCPTransport.HTTP

    async def test_from_json_disabled_servers_still_loaded(self, tmp_path: Path):
        mcp_json = tmp_path / "mcp.json"
        data = {
            "mcpServers": {
                "disabled-server": {
                    "command": "srv",
                    "enabled": False,
                }
            }
        }
        mcp_json.write_text(json.dumps(data))

        cfg = await MCPConfig.from_json(mcp_json)
        assert "disabled-server" in cfg.servers
        assert cfg.servers["disabled-server"].enabled is False
