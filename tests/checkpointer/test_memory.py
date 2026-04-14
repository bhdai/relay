"""Tests for MemoryCheckpointer extended methods."""

import pytest
from langchain_core.messages import HumanMessage

from relay.checkpointer.base import BaseCheckpointer
from relay.checkpointer.impl.memory import MemoryCheckpointer


class TestMemoryCheckpointer:
    """Tests for MemoryCheckpointer class."""

    def test_is_base_checkpointer(self):
        """MemoryCheckpointer should implement BaseCheckpointer."""
        cp = MemoryCheckpointer()
        assert isinstance(cp, BaseCheckpointer)

    @pytest.mark.asyncio
    async def test_get_threads_empty(self):
        """Fresh checkpointer should have no threads."""
        cp = MemoryCheckpointer()
        assert await cp.get_threads() == set()

    @pytest.mark.asyncio
    async def test_get_threads_after_put(self):
        """After saving a checkpoint, the thread should be discoverable."""
        cp = MemoryCheckpointer()

        # Use the checkpointer through a simple graph to create a thread.
        from langgraph.graph import StateGraph

        class State(dict):
            pass

        builder = StateGraph(State)
        builder.add_node("noop", lambda s: s)
        builder.set_entry_point("noop")
        builder.set_finish_point("noop")
        graph = builder.compile(checkpointer=cp)

        await graph.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "t1"}},
        )

        threads = await cp.get_threads()
        assert "t1" in threads

    @pytest.mark.asyncio
    async def test_get_history(self):
        """get_history should return checkpoints oldest-first."""
        cp = MemoryCheckpointer()

        from langgraph.graph import StateGraph

        class State(dict):
            pass

        builder = StateGraph(State)
        builder.add_node("noop", lambda s: s)
        builder.set_entry_point("noop")
        builder.set_finish_point("noop")
        graph = builder.compile(checkpointer=cp)

        # Create two checkpoints on the same thread.
        await graph.ainvoke(
            {"messages": [HumanMessage(content="first")]},
            config={"configurable": {"thread_id": "t1"}},
        )
        await graph.ainvoke(
            {"messages": [HumanMessage(content="second")]},
            config={"configurable": {"thread_id": "t1"}},
        )

        from langchain_core.runnables import RunnableConfig

        latest = await cp.aget_tuple(
            RunnableConfig(configurable={"thread_id": "t1"})
        )
        assert latest is not None

        history = await cp.get_history(latest)
        # Should have at least 2 checkpoints, oldest first.
        assert len(history) >= 2
        # The last entry should be the same as latest.
        assert history[-1].checkpoint.get("id") == latest.checkpoint.get("id")

    @pytest.mark.asyncio
    async def test_delete_after_removes_checkpoints(self):
        """delete_after should remove checkpoints after the given ID."""
        cp = MemoryCheckpointer()

        from langgraph.graph import StateGraph

        class State(dict):
            pass

        builder = StateGraph(State)
        builder.add_node("noop", lambda s: s)
        builder.set_entry_point("noop")
        builder.set_finish_point("noop")
        graph = builder.compile(checkpointer=cp)

        await graph.ainvoke(
            {"messages": [HumanMessage(content="first")]},
            config={"configurable": {"thread_id": "t1"}},
        )

        from langchain_core.runnables import RunnableConfig

        first_latest = await cp.aget_tuple(
            RunnableConfig(configurable={"thread_id": "t1"})
        )
        first_id = first_latest.checkpoint.get("id")

        await graph.ainvoke(
            {"messages": [HumanMessage(content="second")]},
            config={"configurable": {"thread_id": "t1"}},
        )

        deleted = await cp.delete_after("t1", first_id)
        assert deleted >= 1
