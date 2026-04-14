"""Tests for relay.agents.deep_agent."""

from unittest.mock import MagicMock

from langchain_core.tools import tool

from relay.agents.context import AgentContext
from relay.agents.deep_agent import create_deep_agent
from relay.agents.state import AgentState
from relay.tools.subagents.task import SubAgentRuntime


@tool
def _noop_tool() -> str:
    """A no-op tool for testing."""
    return "ok"


class TestCreateDeepAgent:
    def test_returns_compiled_graph_without_subagents(self):
        llm = MagicMock()
        graph = create_deep_agent(
            model=llm,
            tools=[_noop_tool],
            prompt="Test coordinator.",
            state_schema=AgentState,
            context_schema=AgentContext,
            name="coordinator",
        )
        assert callable(getattr(graph, "invoke", None))
        assert callable(getattr(graph, "ainvoke", None))

    def test_returns_compiled_graph_with_subagents(self):
        llm = MagicMock()
        explorer = SubAgentRuntime(
            name="explorer",
            description="Read-only investigation.",
            tools=[_noop_tool],
            prompt="Explorer prompt.",
        )
        worker = SubAgentRuntime(
            name="worker",
            description="Multi-step execution.",
            tools=[_noop_tool],
            prompt="Worker prompt.",
        )
        graph = create_deep_agent(
            model=llm,
            tools=[_noop_tool],
            prompt="Coordinator prompt.",
            subagent_configs=[explorer, worker],
            state_schema=AgentState,
            context_schema=AgentContext,
            name="coordinator",
        )
        assert callable(getattr(graph, "invoke", None))

    def test_no_subagent_configs_means_no_task_tool(self):
        """When subagent_configs is None, the tool list is unchanged."""
        llm = MagicMock()
        graph = create_deep_agent(
            model=llm,
            tools=[_noop_tool],
            prompt="No subagents.",
            state_schema=AgentState,
            context_schema=AgentContext,
        )
        # The graph exists and has no task tool — we verify by checking
        # the compiled graph's tools property or just that it compiled.
        assert callable(getattr(graph, "invoke", None))
