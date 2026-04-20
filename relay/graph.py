"""Build the relay agent graph.

**Compatibility shim.**  The real construction logic now lives in
``relay.agents.factory.AgentFactory``.  This module remains as the
public entry point for ``langgraph.json`` (LangGraph Studio) and any
external callers still importing from ``relay.graph``.

Prefer using ``AgentFactory`` or ``relay.cli.bootstrap.Initializer``
directly in new code.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver

from relay.agents.factory import AgentFactory

# Module-level factory instance — reused across calls so that tool
# preparation side-effects (handle_tool_error) run only once.
_factory = AgentFactory()


def build_graph():
    """Construct Relay's coordinator + subagent graph.

    This zero-argument factory is the entry point for LangGraph Studio
    (``langgraph dev``) and the ``langgraph.json`` configuration.

    For callers that need a checkpointer (e.g. the CLI), use
    ``build_graph_with_checkpointer`` instead.
    """
    return _factory.create_from_config()


def build_graph_with_checkpointer(
    checkpointer: BaseCheckpointSaver,
):
    """Construct Relay's graph with an explicit checkpointer.

    Args:
        checkpointer: Checkpoint saver to attach to the graph.

    Deprecated:
        Use ``AgentFactory.create(checkpointer=...)`` or
        ``Initializer.get_graph(...)`` instead.
    """
    return _factory.create(checkpointer=checkpointer)
