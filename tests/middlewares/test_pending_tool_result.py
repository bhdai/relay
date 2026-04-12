"""Tests for PendingToolResultMiddleware."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.middlewares.pending_tool_result import PendingToolResultMiddleware


@pytest.fixture
def middleware():
    return PendingToolResultMiddleware()


@pytest.fixture
def mock_runtime():
    runtime = AsyncMock()
    runtime.context = AgentContext()
    return runtime


def _state(**overrides) -> AgentState:
    """Build a minimal AgentState dict with sensible defaults."""
    defaults: AgentState = {
        "messages": [],
        "todos": None,
        "files": None,
        "current_input_tokens": None,
        "current_output_tokens": None,
        "total_cost": None,
    }
    defaults.update(overrides)
    return defaults


@pytest.mark.asyncio
async def test_no_messages(middleware, mock_runtime):
    result = await middleware.abefore_agent(_state(), mock_runtime)
    assert result is None


@pytest.mark.asyncio
async def test_no_ai_message(middleware, mock_runtime):
    state = _state(messages=[HumanMessage(content="hello")])
    result = await middleware.abefore_agent(state, mock_runtime)
    assert result is None


@pytest.mark.asyncio
async def test_ai_message_without_tool_calls(middleware, mock_runtime):
    state = _state(
        messages=[
            HumanMessage(content="hello"),
            AIMessage(content="response"),
        ]
    )
    result = await middleware.abefore_agent(state, mock_runtime)
    assert result is None


@pytest.mark.asyncio
async def test_inject_interrupted_for_missing_tool_result(middleware, mock_runtime):
    """Missing ToolMessage should be replaced with an 'Interrupted.' error."""
    state = _state(
        messages=[
            HumanMessage(content="hello"),
            AIMessage(
                content="calling tool",
                tool_calls=[{"id": "call_1", "name": "test_tool", "args": {}}],
            ),
        ]
    )
    result = await middleware.abefore_agent(state, mock_runtime)

    assert result is not None
    messages = result["messages"]
    # [RemoveMessage, HumanMessage, AIMessage, ToolMessage(Interrupted)]
    assert isinstance(messages[0], RemoveMessage)
    assert len(messages) == 4
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[3], ToolMessage)
    assert messages[3].tool_call_id == "call_1"
    assert messages[3].content == "Interrupted."


@pytest.mark.asyncio
async def test_move_tool_result_after_human_message(middleware, mock_runtime):
    """ToolMessage separated from AIMessage by a HumanMessage should be moved."""
    state = _state(
        messages=[
            AIMessage(
                content="calling tool",
                tool_calls=[{"id": "call_1", "name": "test_tool", "args": {}}],
            ),
            HumanMessage(content="interrupt"),
            ToolMessage(content="result", tool_call_id="call_1"),
        ]
    )
    result = await middleware.abefore_agent(state, mock_runtime)

    assert result is not None
    messages = result["messages"]
    assert isinstance(messages[0], RemoveMessage)
    assert len(messages) == 4
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call_1"
    assert isinstance(messages[3], HumanMessage)


@pytest.mark.asyncio
async def test_multiple_tool_calls_mixed_results(middleware, mock_runtime):
    """One present and one missing ToolMessage across two tool calls."""
    state = _state(
        messages=[
            AIMessage(
                content="calling tools",
                tool_calls=[
                    {"id": "call_1", "name": "tool_1", "args": {}},
                    {"id": "call_2", "name": "tool_2", "args": {}},
                ],
            ),
            HumanMessage(content="interrupt"),
            ToolMessage(content="result_1", tool_call_id="call_1"),
        ]
    )
    result = await middleware.abefore_agent(state, mock_runtime)

    assert result is not None
    messages = result["messages"]
    assert isinstance(messages[0], RemoveMessage)
    assert len(messages) == 5
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call_1"
    assert isinstance(messages[3], ToolMessage)
    assert messages[3].tool_call_id == "call_2"
    assert messages[3].content == "Interrupted."
    assert isinstance(messages[4], HumanMessage)


@pytest.mark.asyncio
async def test_tool_results_in_correct_position(middleware, mock_runtime):
    """Properly ordered results need no repair."""
    state = _state(
        messages=[
            AIMessage(
                content="calling tool",
                tool_calls=[{"id": "call_1", "name": "test_tool", "args": {}}],
            ),
            ToolMessage(content="result", tool_call_id="call_1"),
            HumanMessage(content="next message"),
        ]
    )
    result = await middleware.abefore_agent(state, mock_runtime)
    assert result is None


@pytest.mark.asyncio
async def test_multiple_tool_results_correct_order(middleware, mock_runtime):
    """Out-of-order ToolMessages should be re-sorted to match tool_calls."""
    state = _state(
        messages=[
            AIMessage(
                content="calling tools",
                tool_calls=[
                    {"id": "call_1", "name": "tool_1", "args": {}},
                    {"id": "call_2", "name": "tool_2", "args": {}},
                ],
            ),
            HumanMessage(content="interrupt"),
            ToolMessage(content="result_2", tool_call_id="call_2"),
            ToolMessage(content="result_1", tool_call_id="call_1"),
        ]
    )
    result = await middleware.abefore_agent(state, mock_runtime)

    assert result is not None
    messages = result["messages"]
    assert isinstance(messages[0], RemoveMessage)
    assert len(messages) == 5
    assert isinstance(messages[1], AIMessage)
    # Reordered to match tool_calls declaration order.
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call_1"
    assert messages[2].content == "result_1"
    assert isinstance(messages[3], ToolMessage)
    assert messages[3].tool_call_id == "call_2"
    assert messages[3].content == "result_2"
    assert isinstance(messages[4], HumanMessage)
