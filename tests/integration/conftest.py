import pytest
from langchain.agents import AgentState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``integration`` marker to every test in this directory."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def create_test_graph():
    """Factory fixture: builds a minimal START → tools → END graph.

    Accepts an optional ``state_schema`` to use a custom state class
    (e.g. the extended ``relay.state.AgentState``).
    """

    def _create(tools: list, state_schema=AgentState):
        graph = StateGraph(state_schema)
        tool_node = ToolNode(tools, handle_tool_errors=True)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("tools")
        graph.set_finish_point("tools")
        return graph.compile(checkpointer=MemorySaver())

    return _create
