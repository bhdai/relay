"""Tests for IndexedAsyncSqliteSaver."""

from typing import Annotated

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import add_messages
from typing_extensions import TypedDict

from relay.checkpointer.base import BaseCheckpointer, ThreadSummary
from relay.checkpointer.impl.sqlite import IndexedAsyncSqliteSaver


class _MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


class TestIndexedAsyncSqliteSaver:
    """Tests for the SQLite checkpointer implementation."""

    @pytest.mark.asyncio
    async def test_is_base_checkpointer(self):
        """IndexedAsyncSqliteSaver should implement BaseCheckpointer."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
            assert isinstance(cp, BaseCheckpointer)

    @pytest.mark.asyncio
    async def test_get_threads_empty(self):
        """Fresh SQLite checkpointer should have no threads."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
            threads = await cp.get_threads()
            assert threads == set()

    @pytest.mark.asyncio
    async def test_get_threads_after_put(self):
        """After saving via a graph, the thread should be discoverable."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
            from langgraph.graph import StateGraph

            class State(dict):
                pass

            builder = StateGraph(State)
            builder.add_node("noop", lambda s: s)
            builder.set_entry_point("noop")
            builder.set_finish_point("noop")
            graph = builder.compile(checkpointer=cp)

            await graph.ainvoke(
                {"messages": [HumanMessage(content="test")]},
                config={"configurable": {"thread_id": "t1"}},
            )

            threads = await cp.get_threads()
            assert "t1" in threads

    @pytest.mark.asyncio
    async def test_get_history(self):
        """get_history should return checkpoints in chronological order."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
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
            await graph.ainvoke(
                {"messages": [HumanMessage(content="second")]},
                config={"configurable": {"thread_id": "t1"}},
            )

            latest = await cp.aget_tuple(
                RunnableConfig(configurable={"thread_id": "t1"})
            )
            assert latest is not None

            history = await cp.get_history(latest)
            assert len(history) >= 2
            assert history[-1].checkpoint.get("id") == latest.checkpoint.get("id")

    @pytest.mark.asyncio
    async def test_delete_after(self):
        """delete_after should remove later checkpoints."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
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

    @pytest.mark.asyncio
    async def test_message_index_created(self):
        """The setup should create checkpoint_messages table."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
            async with cp.lock:
                cursor = await cp.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='checkpoint_messages'"
                )
                row = await cursor.fetchone()
            assert row is not None

    @pytest.mark.asyncio
    async def test_get_thread_summaries_empty(self):
        """Fresh SQLite checkpointer should return no summaries."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
            summaries = await cp.get_thread_summaries()
            assert summaries == []

    @pytest.mark.asyncio
    async def test_get_thread_summaries_returns_previews(self):
        """Summaries should include the last human message as preview."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
            from langgraph.graph import StateGraph

            builder = StateGraph(_MessagesState)
            builder.add_node("noop", lambda s: s)
            builder.set_entry_point("noop")
            builder.set_finish_point("noop")
            graph = builder.compile(checkpointer=cp)

            await graph.ainvoke(
                {"messages": [HumanMessage(content="hello from thread one")]},
                config={"configurable": {"thread_id": "t1"}},
            )
            await graph.ainvoke(
                {"messages": [HumanMessage(content="hello from thread two")]},
                config={"configurable": {"thread_id": "t2"}},
            )

            summaries = await cp.get_thread_summaries()
            assert len(summaries) == 2

            ids = {s.thread_id for s in summaries}
            assert ids == {"t1", "t2"}

            for s in summaries:
                assert isinstance(s, ThreadSummary)
                assert "hello from thread" in s.last_message

    @pytest.mark.asyncio
    async def test_get_thread_summaries_sorted_newest_first(self):
        """Summaries should be sorted by timestamp, newest first."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
            from langgraph.graph import StateGraph

            builder = StateGraph(_MessagesState)
            builder.add_node("noop", lambda s: s)
            builder.set_entry_point("noop")
            builder.set_finish_point("noop")
            graph = builder.compile(checkpointer=cp)

            await graph.ainvoke(
                {"messages": [HumanMessage(content="first")]},
                config={"configurable": {"thread_id": "t1"}},
            )
            await graph.ainvoke(
                {"messages": [HumanMessage(content="second")]},
                config={"configurable": {"thread_id": "t2"}},
            )

            summaries = await cp.get_thread_summaries()
            assert summaries[0].thread_id == "t2"
            assert summaries[1].thread_id == "t1"

    @pytest.mark.asyncio
    async def test_get_thread_summaries_fallback_without_index(self):
        """Summaries should still work via fallback when index is empty."""
        async with IndexedAsyncSqliteSaver.create(connection_string=":memory:") as cp:
            from langgraph.graph import StateGraph

            builder = StateGraph(_MessagesState)
            builder.add_node("noop", lambda s: s)
            builder.set_entry_point("noop")
            builder.set_finish_point("noop")
            graph = builder.compile(checkpointer=cp)

            await graph.ainvoke(
                {"messages": [HumanMessage(content="test message")]},
                config={"configurable": {"thread_id": "t1"}},
            )

            # Clear the index table to force the fallback path.
            async with cp.lock:
                await cp.conn.execute("DELETE FROM checkpoint_messages")
                await cp.conn.commit()

            summaries = await cp.get_thread_summaries()
            assert len(summaries) == 1
            assert summaries[0].thread_id == "t1"
            assert "test message" in summaries[0].last_message
