"""Tests for ReturnDirectMiddleware."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.middlewares.return_direct import ReturnDirectMiddleware


class TestReturnDirectMiddleware:
    """Tests for ReturnDirectMiddleware class."""

    @pytest.mark.asyncio
    async def test_jumps_to_end_with_return_direct(self):
        """ToolMessage with return_direct=True should jump to end."""
        middleware = ReturnDirectMiddleware()

        tool_msg = ToolMessage(
            name="test_tool",
            content="result",
            tool_call_id="call_1",
            return_direct=True,
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="test", id="msg_1"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_1", "name": "test_tool", "args": {}}],
                    id="msg_2",
                ),
                tool_msg,
            ],
            "todos": None,
            "files": {},
            "current_input_tokens": None,
            "current_output_tokens": None,
            "total_cost": None,
        }

        runtime = Mock()
        runtime.context = AgentContext()

        result = await middleware.abefore_model(state, runtime)

        assert result is not None
        assert result["jump_to"] == "end"

    @pytest.mark.asyncio
    async def test_does_not_jump_without_return_direct(self):
        """Normal ToolMessage should not trigger a jump."""
        middleware = ReturnDirectMiddleware()

        state: AgentState = {
            "messages": [
                HumanMessage(content="test", id="msg_1"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_1", "name": "test_tool", "args": {}}],
                    id="msg_2",
                ),
                ToolMessage(
                    name="test_tool",
                    content="result",
                    tool_call_id="call_1",
                    id="msg_3",
                ),
            ],
            "todos": None,
            "files": {},
            "current_input_tokens": None,
            "current_output_tokens": None,
            "total_cost": None,
        }

        runtime = Mock()
        runtime.context = AgentContext()

        result = await middleware.abefore_model(state, runtime)
        assert result is None

    @pytest.mark.asyncio
    async def test_checks_only_recent_tool_messages(self):
        """Old return_direct messages should not affect a newer batch."""
        middleware = ReturnDirectMiddleware()

        state: AgentState = {
            "messages": [
                HumanMessage(content="old test", id="msg_1"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_old", "name": "old_tool", "args": {}}],
                    id="msg_2",
                ),
                ToolMessage(
                    name="old_tool",
                    content="old result",
                    tool_call_id="call_old",
                    id="msg_3",
                    return_direct=True,
                ),
                # The HumanMessage breaks the backwards scan, so old
                # return_direct is not seen.
                HumanMessage(content="new test", id="msg_4"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_new", "name": "new_tool", "args": {}}],
                    id="msg_5",
                ),
                ToolMessage(
                    name="new_tool",
                    content="new result",
                    tool_call_id="call_new",
                    id="msg_6",
                ),
            ],
            "todos": None,
            "files": {},
            "current_input_tokens": None,
            "current_output_tokens": None,
            "total_cost": None,
        }

        runtime = Mock()
        runtime.context = AgentContext()

        result = await middleware.abefore_model(state, runtime)
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_empty_messages(self):
        """Empty message list should not error."""
        middleware = ReturnDirectMiddleware()

        state: AgentState = {
            "messages": [],
            "todos": None,
            "files": {},
            "current_input_tokens": None,
            "current_output_tokens": None,
            "total_cost": None,
        }

        runtime = Mock()
        runtime.context = AgentContext()

        result = await middleware.abefore_model(state, runtime)
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_multiple_recent_tool_messages(self):
        """Any return_direct in the recent batch should trigger the jump."""
        middleware = ReturnDirectMiddleware()

        state: AgentState = {
            "messages": [
                HumanMessage(content="test", id="msg_1"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "call_1", "name": "tool1", "args": {}},
                        {"id": "call_2", "name": "tool2", "args": {}},
                    ],
                    id="msg_2",
                ),
                ToolMessage(
                    name="tool1",
                    content="result1",
                    tool_call_id="call_1",
                    id="msg_3",
                ),
                ToolMessage(
                    name="tool2",
                    content="result2",
                    tool_call_id="call_2",
                    id="msg_4",
                    return_direct=True,
                ),
            ],
            "todos": None,
            "files": {},
            "current_input_tokens": None,
            "current_output_tokens": None,
            "total_cost": None,
        }

        runtime = Mock()
        runtime.context = AgentContext()

        result = await middleware.abefore_model(state, runtime)

        assert result is not None
        assert result["jump_to"] == "end"
