"""Tests for relay.configs.agent — schema validation and lookups."""

import pytest

from relay.configs.agent import (
    AgentConfig,
    BaseAgentConfig,
    BatchAgentConfig,
    BatchSubAgentConfig,
    SubAgentConfig,
    ToolsConfig,
)


# ==============================================================================
# ToolsConfig
# ==============================================================================


class TestToolsConfig:
    def test_defaults(self):
        tc = ToolsConfig()
        assert tc.patterns == []
        assert tc.output_max_tokens is None

    def test_with_patterns(self):
        tc = ToolsConfig(patterns=["impl:web:*", "internal:*:*"], output_max_tokens=5000)
        assert len(tc.patterns) == 2
        assert tc.output_max_tokens == 5000


# ==============================================================================
# BaseAgentConfig / SubAgentConfig
# ==============================================================================


class TestSubAgentConfig:
    def test_minimal(self):
        sa = SubAgentConfig(name="explorer")
        assert sa.name == "explorer"
        assert sa.llm == "default"
        assert sa.recursion_limit == 100
        assert sa.tools is None

    def test_full(self):
        sa = SubAgentConfig(
            name="worker",
            description="Does work",
            prompt="You are a worker.",
            llm="gpt-4.1",
            tools=ToolsConfig(patterns=["impl:*:*"]),
            recursion_limit=50,
        )
        assert sa.name == "worker"
        assert sa.description == "Does work"
        assert sa.llm == "gpt-4.1"
        assert sa.tools is not None
        assert len(sa.tools.patterns) == 1


# ==============================================================================
# AgentConfig
# ==============================================================================


class TestAgentConfig:
    def test_minimal(self):
        a = AgentConfig(name="general")
        assert a.name == "general"
        assert a.default is False
        assert a.subagents is None
        assert a.checkpointer is None

    def test_with_subagents_and_checkpointer(self):
        a = AgentConfig(
            name="main",
            default=True,
            subagents=["explorer", "worker"],
            checkpointer="sqlite",
        )
        assert a.default is True
        assert a.subagents == ["explorer", "worker"]
        assert a.checkpointer == "sqlite"


# ==============================================================================
# BatchAgentConfig
# ==============================================================================


class TestBatchAgentConfig:
    def test_get_agent_by_name(self):
        batch = BatchAgentConfig(agents=[
            AgentConfig(name="alpha"),
            AgentConfig(name="beta"),
        ])
        assert batch.get_agent("alpha") is not None
        assert batch.get_agent("alpha").name == "alpha"
        assert batch.get_agent("missing") is None

    def test_get_default_agent_explicit(self):
        batch = BatchAgentConfig(agents=[
            AgentConfig(name="alpha"),
            AgentConfig(name="beta", default=True),
        ])
        default = batch.get_default_agent()
        assert default is not None
        assert default.name == "beta"

    def test_get_default_agent_fallback_to_first(self):
        batch = BatchAgentConfig(agents=[
            AgentConfig(name="alpha"),
            AgentConfig(name="beta"),
        ])
        default = batch.get_default_agent()
        assert default is not None
        assert default.name == "alpha"

    def test_empty_batch(self):
        batch = BatchAgentConfig()
        assert batch.get_default_agent() is None
        assert batch.agent_names == []

    def test_get_agent_none_returns_default(self):
        batch = BatchAgentConfig(agents=[
            AgentConfig(name="only", default=True),
        ])
        assert batch.get_agent(None).name == "only"


# ==============================================================================
# BatchSubAgentConfig
# ==============================================================================


class TestBatchSubAgentConfig:
    def test_get_subagent(self):
        batch = BatchSubAgentConfig(subagents=[
            SubAgentConfig(name="explorer"),
            SubAgentConfig(name="worker"),
        ])
        assert batch.get_subagent("explorer") is not None
        assert batch.get_subagent("missing") is None
        assert batch.subagent_names == ["explorer", "worker"]
