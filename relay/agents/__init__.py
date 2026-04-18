from __future__ import annotations

from typing import Any

from relay.agents.context import AgentContext
from relay.agents.state import AgentState


def __getattr__(name: str) -> Any:
    """Lazily expose heavy exports to avoid package init cycles."""
    if name == "AgentFactory":
        from relay.agents.factory import AgentFactory

        return AgentFactory
    raise AttributeError(name)


def create_deep_agent(*args, **kwargs):
    """Lazily import deep-agent builder to avoid package init cycles."""
    from relay.agents.deep_agent import create_deep_agent as _create_deep_agent

    return _create_deep_agent(*args, **kwargs)


def create_react_agent(*args, **kwargs):
    """Lazily import ReAct builder to avoid package init cycles."""
    from relay.agents.react_agent import create_react_agent as _create_react_agent

    return _create_react_agent(*args, **kwargs)


__all__ = [
    "AgentContext",
    "AgentFactory",
    "AgentState",
    "create_deep_agent",
    "create_react_agent",
]
