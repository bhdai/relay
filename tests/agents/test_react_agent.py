"""Tests for relay.agents.react_agent."""

from unittest.mock import MagicMock

from langchain_core.tools import tool

from relay.agents.context import AgentContext
from relay.agents.react_agent import create_react_agent
from relay.agents.state import AgentState


@tool
def _noop_tool() -> str:
    """A no-op tool for testing."""
    return "ok"


class TestCreateReactAgent:
    def test_returns_compiled_graph(self):
        llm = MagicMock()
        graph = create_react_agent(
            llm,
            tools=[_noop_tool],
            prompt="You are a test agent.\n{working_dir}",
            state_schema=AgentState,
            context_schema=AgentContext,
            name="test-agent",
        )
        # A compiled graph exposes an invoke method.
        assert callable(getattr(graph, "invoke", None))
        assert callable(getattr(graph, "ainvoke", None))

    def test_accepts_no_state_schema(self):
        llm = MagicMock()
        graph = create_react_agent(
            llm,
            tools=[_noop_tool],
            prompt="Minimal agent.",
        )
        assert callable(getattr(graph, "invoke", None))

    def test_accepts_checkpointer(self):
        from langgraph.checkpoint.memory import InMemorySaver

        llm = MagicMock()
        checkpointer = InMemorySaver()
        graph = create_react_agent(
            llm,
            tools=[_noop_tool],
            prompt="Agent with checkpointer.",
            checkpointer=checkpointer,
        )
        assert callable(getattr(graph, "invoke", None))
