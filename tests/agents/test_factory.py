"""Tests for relay.agents.factory."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import tool as langchain_tool

from relay.agents.factory import AgentFactory
from relay.configs.registry import ConfigRegistry


# ==============================================================================
# Helpers
# ==============================================================================


@pytest.fixture()
def create_mock_tool():
    """Factory fixture that produces lightweight mock tools."""

    def _make(name: str):
        @langchain_tool
        def _tool() -> str:
            """Placeholder tool."""
            return name

        _tool.name = name
        return _tool

    return _make


# ==============================================================================
# Existing AgentFactory tests
# ==============================================================================


class TestAgentFactory:
    def test_create_returns_compiled_graph(self):
        """Factory.create() returns a graph with invoke/ainvoke."""
        llm = MagicMock()
        factory = AgentFactory(model=llm)
        graph = factory.create()
        assert callable(getattr(graph, "invoke", None))
        assert callable(getattr(graph, "ainvoke", None))

    def test_create_with_checkpointer(self):
        """Factory.create(checkpointer=...) wires the checkpointer."""
        from langgraph.checkpoint.memory import InMemorySaver

        llm = MagicMock()
        factory = AgentFactory(model=llm)
        checkpointer = InMemorySaver()
        graph = factory.create(checkpointer=checkpointer)
        assert callable(getattr(graph, "invoke", None))

    def test_default_model_uses_settings(self, monkeypatch):
        """When no model is passed, the factory creates one from settings."""
        # Patch get_settings to avoid needing real env vars.
        mock_settings = MagicMock()
        mock_settings.llm.provider = "openai"
        mock_settings.llm.model = "gpt-4.1-mini"
        mock_settings.llm.max_tokens = 10000
        mock_settings.llm.temperature = 0.1
        mock_settings.llm.streaming = True
        mock_settings.llm.openai_api_key.get_secret_value.return_value = "sk-test"
        mock_settings.llm.input_cost_per_mtok = 0.0
        mock_settings.llm.output_cost_per_mtok = 0.0
        mock_settings.rate_limit.requests_per_second = 5.0
        mock_settings.rate_limit.check_every_n_seconds = 0.1
        mock_settings.rate_limit.max_bucket_size = 10
        monkeypatch.setattr("relay.agents.factory.get_settings", lambda: mock_settings)

        factory = AgentFactory()
        graph = factory.create()
        assert callable(getattr(graph, "invoke", None))

    @pytest.mark.asyncio
    async def test_resolve_llm_metadata_uses_registry_alias(self, tmp_path: Path):
        registry = ConfigRegistry(tmp_path)
        await registry.ensure_config_dir()

        factory = AgentFactory(registry=registry, model_name="haiku-4.5-thinking")
        alias, input_cost, output_cost = await factory.resolve_llm_metadata()

        assert alias == "haiku-4.5-thinking"
        assert input_cost == 1.0
        assert output_cost == 5.0

    def test_model_from_config_uses_injected_model(self):
        model = MagicMock()
        factory = AgentFactory(model=model)

        resolved = factory._model_from_config(None)

        assert resolved is model

    def test_coordinator_tools_are_read_only(self):
        """Coordinator tool surface does not include mutating FS or terminal."""
        llm = MagicMock()
        factory = AgentFactory(model=llm)
        tools = factory._coordinator_tools()
        tool_names = {t.name for t in tools}

        # Should include read-only tools.
        assert "read_file" in tool_names
        assert "glob_files" in tool_names
        assert "grep_files" in tool_names
        assert "ls" in tool_names

        # Should NOT include mutating tools or think.
        assert "write_file" not in tool_names
        assert "edit_file" not in tool_names
        assert "run_command" not in tool_names
        assert "think" not in tool_names

    def test_subagent_configs_include_explorer_and_worker(self):
        """Factory defines explorer and general-purpose subagents."""
        configs = AgentFactory._subagent_configs()
        names = {c.name for c in configs}
        assert names == {"explorer", "general-purpose"}


# ==============================================================================
# _parse_tool_references
# ==============================================================================


class TestParseToolReferences:
    def test_parse_valid_references(self):
        tool_refs = [
            "impl:file_system:read_file",
            "internal:todo:write_todos",
        ]

        impl, mcp, internal = AgentFactory._parse_tool_references(tool_refs)

        assert impl == ["file_system:read_file"]
        assert mcp is None
        assert internal == ["todo:write_todos"]

    def test_parse_none_returns_none(self):
        impl, mcp, internal = AgentFactory._parse_tool_references(None)

        assert impl is None
        assert mcp is None
        assert internal is None

    def test_parse_empty_list_returns_none(self):
        impl, mcp, internal = AgentFactory._parse_tool_references([])

        assert impl is None
        assert mcp is None
        assert internal is None

    def test_parse_only_impl_tools(self):
        tool_refs = ["impl:file_system:read_file", "impl:web:fetch_url"]

        impl, mcp, internal = AgentFactory._parse_tool_references(tool_refs)

        assert impl == ["file_system:read_file", "web:fetch_url"]
        assert mcp is None
        assert internal is None

    def test_parse_invalid_format_skipped(self):
        tool_refs = [
            "impl:file_system:read_file",
            "invalid_format",
            "internal:todo:write_todos",
        ]

        impl, mcp, internal = AgentFactory._parse_tool_references(tool_refs)

        assert impl == ["file_system:read_file"]
        assert mcp is None
        assert internal == ["todo:write_todos"]

    def test_parse_unknown_tool_type_skipped(self):
        tool_refs = [
            "impl:file_system:read_file",
            "unknown:module:tool",
            "internal:todo:write_todos",
        ]

        impl, mcp, internal = AgentFactory._parse_tool_references(tool_refs)

        assert impl == ["file_system:read_file"]
        assert mcp is None
        assert internal == ["todo:write_todos"]

    def test_parse_wildcard_patterns(self):
        tool_refs = [
            "impl:*:*",
            "internal:todo:write_*",
        ]

        impl, mcp, internal = AgentFactory._parse_tool_references(tool_refs)

        assert impl == ["*:*"]
        assert mcp is None
        assert internal == ["todo:write_*"]

    def test_parse_negative_patterns(self):
        tool_refs = [
            "impl:*:*",
            "!impl:terminal:*",
            "internal:*:*",
            "!internal:todo:*",
        ]

        impl, mcp, internal = AgentFactory._parse_tool_references(tool_refs)

        assert impl == ["*:*", "!terminal:*"]
        assert mcp is None
        assert internal == ["*:*", "!todo:*"]

    def test_parse_mcp_references(self):
        tool_refs = [
            "impl:file_system:read_file",
            "mcp:my_server:my_tool",
            "mcp:other_server:*",
        ]

        impl, mcp, internal = AgentFactory._parse_tool_references(tool_refs)

        assert impl == ["file_system:read_file"]
        assert mcp == ["my_server:my_tool", "other_server:*"]
        assert internal is None


# ==============================================================================
# _filter_tools
# ==============================================================================


class TestFilterTools:
    def test_filter_by_exact_patterns(self, create_mock_tool):
        mock_tool1 = create_mock_tool("tool1")
        mock_tool2 = create_mock_tool("tool2")
        mock_tool3 = create_mock_tool("tool3")

        all_tools = [mock_tool1, mock_tool2, mock_tool3]
        tool_dict = AgentFactory._build_tool_dict(all_tools)
        module_map = {"tool1": "module1", "tool2": "module2", "tool3": "module3"}
        patterns = ["module1:tool1", "module3:tool3"]

        result = AgentFactory._filter_tools(tool_dict, patterns, module_map)

        assert len(result) == 2
        assert {t.name for t in result} == {"tool1", "tool3"}

    def test_filter_none_patterns_returns_empty(self, create_mock_tool):
        mock_tool = create_mock_tool("tool1")
        all_tools = [mock_tool]
        tool_dict = AgentFactory._build_tool_dict(all_tools)
        module_map = {"tool1": "module1"}

        result = AgentFactory._filter_tools(tool_dict, None, module_map)

        assert result == []

    def test_filter_empty_patterns_returns_empty(self, create_mock_tool):
        mock_tool = create_mock_tool("tool1")
        all_tools = [mock_tool]
        tool_dict = AgentFactory._build_tool_dict(all_tools)
        module_map = {"tool1": "module1"}

        result = AgentFactory._filter_tools(tool_dict, [], module_map)

        assert result == []

    def test_filter_no_matches(self, create_mock_tool):
        mock_tool1 = create_mock_tool("tool1")
        mock_tool2 = create_mock_tool("tool2")

        all_tools = [mock_tool1, mock_tool2]
        tool_dict = AgentFactory._build_tool_dict(all_tools)
        module_map = {"tool1": "module1", "tool2": "module2"}
        patterns = ["module3:tool3", "module4:tool4"]

        result = AgentFactory._filter_tools(tool_dict, patterns, module_map)

        assert result == []

    def test_filter_wildcard_all(self, create_mock_tool):
        mock_tool1 = create_mock_tool("tool1")
        mock_tool2 = create_mock_tool("tool2")

        all_tools = [mock_tool1, mock_tool2]
        tool_dict = AgentFactory._build_tool_dict(all_tools)
        module_map = {"tool1": "module1", "tool2": "module2"}
        patterns = ["*:*"]

        result = AgentFactory._filter_tools(tool_dict, patterns, module_map)

        assert len(result) == 2

    def test_filter_wildcard_module(self, create_mock_tool):
        mock_tool1 = create_mock_tool("read_file")
        mock_tool2 = create_mock_tool("write_file")
        mock_tool3 = create_mock_tool("fetch_url")

        all_tools = [mock_tool1, mock_tool2, mock_tool3]
        tool_dict = AgentFactory._build_tool_dict(all_tools)
        module_map = {
            "read_file": "file_system",
            "write_file": "file_system",
            "fetch_url": "web",
        }
        patterns = ["file_system:*"]

        result = AgentFactory._filter_tools(tool_dict, patterns, module_map)

        assert len(result) == 2
        assert {t.name for t in result} == {"read_file", "write_file"}

    def test_filter_wildcard_tool_name(self, create_mock_tool):
        mock_tool1 = create_mock_tool("read_file")
        mock_tool2 = create_mock_tool("read_dir")
        mock_tool3 = create_mock_tool("write_file")

        all_tools = [mock_tool1, mock_tool2, mock_tool3]
        tool_dict = AgentFactory._build_tool_dict(all_tools)
        module_map = {
            "read_file": "file_system",
            "read_dir": "file_system",
            "write_file": "file_system",
        }
        patterns = ["file_system:read_*"]

        result = AgentFactory._filter_tools(tool_dict, patterns, module_map)

        assert len(result) == 2
        assert {t.name for t in result} == {"read_file", "read_dir"}

    def test_filter_negative_pattern_excludes_tools(self, create_mock_tool):
        mock_tool1 = create_mock_tool("read_file")
        mock_tool2 = create_mock_tool("write_file")
        mock_tool3 = create_mock_tool("delete_file")

        all_tools = [mock_tool1, mock_tool2, mock_tool3]
        tool_dict = AgentFactory._build_tool_dict(all_tools)
        module_map = {
            "read_file": "file_system",
            "write_file": "file_system",
            "delete_file": "file_system",
        }
        patterns = ["file_system:*", "!file_system:delete_file"]

        result = AgentFactory._filter_tools(tool_dict, patterns, module_map)

        assert len(result) == 2
        assert {t.name for t in result} == {"read_file", "write_file"}

    def test_filter_negative_pattern_with_wildcard(self, create_mock_tool):
        mock_tool1 = create_mock_tool("read_file")
        mock_tool2 = create_mock_tool("write_file")
        mock_tool3 = create_mock_tool("run_command")

        all_tools = [mock_tool1, mock_tool2, mock_tool3]
        tool_dict = AgentFactory._build_tool_dict(all_tools)
        module_map = {
            "read_file": "file_system",
            "write_file": "file_system",
            "run_command": "terminal",
        }
        patterns = ["*:*", "!terminal:*"]

        result = AgentFactory._filter_tools(tool_dict, patterns, module_map)

        assert len(result) == 2
        assert {t.name for t in result} == {"read_file", "write_file"}


# ==============================================================================
# _filter_mcp_tools
# ==============================================================================


class TestFilterMcpTools:
    def test_filter_mcp_tools_by_server(self, create_mock_tool):
        """MCP tools use ``server__name`` format in both tool_dict and module_map."""
        tool_a = create_mock_tool("srv__tool_a")
        tool_b = create_mock_tool("srv__tool_b")
        tool_c = create_mock_tool("other__tool_c")

        mcp_dict = AgentFactory._build_tool_dict([tool_a, tool_b, tool_c])
        mcp_module_map = {
            "srv__tool_a": "srv",
            "srv__tool_b": "srv",
            "other__tool_c": "other",
        }

        result = AgentFactory._filter_mcp_tools(
            mcp_dict, ["srv:*"], mcp_module_map
        )
        assert {t.name for t in result} == {"srv__tool_a", "srv__tool_b"}

    def test_filter_mcp_tools_none_patterns(self, create_mock_tool):
        tool_a = create_mock_tool("srv__tool_a")
        mcp_dict = AgentFactory._build_tool_dict([tool_a])
        mcp_module_map = {"srv__tool_a": "srv"}

        result = AgentFactory._filter_mcp_tools(mcp_dict, None, mcp_module_map)
        assert result == []

    def test_filter_mcp_tools_none_dict(self):
        result = AgentFactory._filter_mcp_tools(None, ["srv:*"], None)
        assert result == []


class TestRuntimeModelResolution:
    @pytest.mark.asyncio
    async def test_cli_model_override_takes_precedence(self, monkeypatch):
        mock_settings = MagicMock()
        mock_settings.llm.provider = "openai"
        mock_settings.llm.model = "gpt-4.1-mini"
        mock_settings.llm.max_tokens = 10000
        mock_settings.llm.temperature = 0.1
        mock_settings.llm.streaming = True
        mock_settings.llm.input_cost_per_mtok = 0.0
        mock_settings.llm.output_cost_per_mtok = 0.0
        mock_settings.rate_limit.requests_per_second = 5.0
        mock_settings.rate_limit.check_every_n_seconds = 0.1
        mock_settings.rate_limit.max_bucket_size = 10

        monkeypatch.setattr("relay.agents.factory.get_settings", lambda: mock_settings)

        factory = AgentFactory(model_name="gpt-5.1-codex-mini")
        llm_config = await factory._resolve_llm_config(configured_llm_name="default")

        assert llm_config.model == "gpt-5.1-codex-mini"

    @pytest.mark.asyncio
    async def test_agent_llm_is_used_when_not_default(self, monkeypatch):
        mock_settings = MagicMock()
        mock_settings.llm.provider = "openai"
        mock_settings.llm.model = "gpt-4.1-mini"
        mock_settings.llm.max_tokens = 10000
        mock_settings.llm.temperature = 0.1
        mock_settings.llm.streaming = True
        mock_settings.llm.input_cost_per_mtok = 0.0
        mock_settings.llm.output_cost_per_mtok = 0.0
        mock_settings.rate_limit.requests_per_second = 5.0
        mock_settings.rate_limit.check_every_n_seconds = 0.1
        mock_settings.rate_limit.max_bucket_size = 10

        monkeypatch.setattr("relay.agents.factory.get_settings", lambda: mock_settings)

        factory = AgentFactory()
        llm_config = await factory._resolve_llm_config(
            configured_llm_name="gpt-5.1-codex-mini"
        )

        assert llm_config.model == "gpt-5.1-codex-mini"


class TestConfiguredToolResolution:
    def test_resolve_tools_matches_logical_filesystem_module(self):
        factory = AgentFactory(model=MagicMock())

        tools = factory._resolve_tools_from_patterns(
            [
                "impl:file_system:read_file",
                "impl:file_system:grep_files",
                "internal:todo:write_todos",
            ]
        )

        assert {tool.name for tool in tools} == {
            "read_file",
            "grep_files",
            "write_todos",
        }
