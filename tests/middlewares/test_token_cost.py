"""Tests for TokenCostMiddleware."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.middlewares.token_cost import TokenCostMiddleware


class TestTokenCostMiddleware:
    """Tests for TokenCostMiddleware class."""

    @pytest.mark.asyncio
    async def test_extracts_token_usage_from_ai_message(self, tmp_path):
        """Token counts and cost should be extracted from usage_metadata."""
        middleware = TokenCostMiddleware()

        ai_message = AIMessage(
            content="test response",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )

        state: AgentState = {
            "messages": [ai_message],
            "todos": None,
            "files": {},
            "current_input_tokens": None,
            "current_output_tokens": None,
            "total_cost": None,
        }

        runtime = Mock()
        runtime.context = AgentContext(
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=2.0,
        )

        result = await middleware.aafter_model(state, runtime)

        assert result is not None
        assert result["current_input_tokens"] == 100
        assert result["current_output_tokens"] == 50
        # cost = 100/1M * 1.0 + 50/1M * 2.0 = 0.0001 + 0.0001 = 0.0002
        assert result["total_cost"] == pytest.approx(0.0002)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_messages(self):
        """Empty message list should produce no state update."""
        middleware = TokenCostMiddleware()

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

        result = await middleware.aafter_model(state, runtime)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_usage_metadata(self):
        """AIMessage without usage_metadata should produce no update."""
        middleware = TokenCostMiddleware()

        ai_message = Mock(spec=AIMessage)
        ai_message.usage_metadata = None

        state: AgentState = {
            "messages": [ai_message],
            "todos": None,
            "files": {},
            "current_input_tokens": None,
            "current_output_tokens": None,
            "total_cost": None,
        }

        runtime = Mock()
        runtime.context = AgentContext()

        result = await middleware.aafter_model(state, runtime)
        assert result is None

    @pytest.mark.asyncio
    async def test_zero_cost_when_pricing_is_zero(self):
        """When pricing rates are 0.0 (default), cost should be zero."""
        middleware = TokenCostMiddleware()

        ai_message = AIMessage(
            content="test",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )

        state: AgentState = {
            "messages": [ai_message],
            "todos": None,
            "files": {},
            "current_input_tokens": None,
            "current_output_tokens": None,
            "total_cost": None,
        }

        runtime = Mock()
        # Default AgentContext has input/output cost of 0.0
        runtime.context = AgentContext()

        result = await middleware.aafter_model(state, runtime)

        assert result is not None
        assert result["current_input_tokens"] == 100
        assert result["current_output_tokens"] == 50
        assert result["total_cost"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_cost_with_only_input_pricing(self):
        """When only one rate is non-zero, cost reflects that rate only."""
        middleware = TokenCostMiddleware()

        ai_message = AIMessage(
            content="test",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )

        state: AgentState = {
            "messages": [ai_message],
            "todos": None,
            "files": {},
            "current_input_tokens": None,
            "current_output_tokens": None,
            "total_cost": None,
        }

        runtime = Mock()
        runtime.context = AgentContext(
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=0.0,
        )

        result = await middleware.aafter_model(state, runtime)

        assert result is not None
        assert result["current_input_tokens"] == 100
        # cost = 100/1M * 1.0 + 50/1M * 0.0 = 0.0001
        assert result["total_cost"] == pytest.approx(0.0001)
