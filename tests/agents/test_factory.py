"""Tests for relay.agents.factory."""

from unittest.mock import MagicMock

from relay.agents.factory import AgentFactory


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
        mock_settings.llm.model = "gpt-4.1-mini"
        mock_settings.llm.openai_api_key.get_secret_value.return_value = "sk-test"
        monkeypatch.setattr("relay.agents.factory.get_settings", lambda: mock_settings)

        factory = AgentFactory()
        graph = factory.create()
        assert callable(getattr(graph, "invoke", None))

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

        # Should NOT include mutating tools.
        assert "write_file" not in tool_names
        assert "edit_file" not in tool_names
        assert "run_command" not in tool_names

    def test_subagent_configs_include_explorer_and_worker(self):
        """Factory defines explorer and general-purpose subagents."""
        configs = AgentFactory._subagent_configs()
        names = {c.name for c in configs}
        assert names == {"explorer", "general-purpose"}
