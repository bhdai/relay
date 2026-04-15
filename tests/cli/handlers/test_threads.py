"""Tests for ThreadManager checkpointer-backed thread listing."""

from __future__ import annotations

from typing import Annotated

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict

from relay.checkpointer.base import ThreadSummary
from relay.checkpointer.impl.memory import MemoryCheckpointer
from relay.cli.handlers.threads import ThreadManager


class _MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


class TestThreadManagerBuildList:
    """Tests for ThreadManager._build_thread_list merge logic."""

    @pytest.mark.asyncio
    async def test_empty_without_checkpointer(self):
        """With no checkpointer and no recorded threads, list is empty."""
        tm = ThreadManager()
        entries = await tm._build_thread_list(None, None)
        assert entries == []

    @pytest.mark.asyncio
    async def test_session_local_threads_without_checkpointer(self):
        """Session-local threads appear when no checkpointer is given."""
        tm = ThreadManager()
        tm.record("local-1", "first message")
        tm.record("local-2", "second message")

        entries = await tm._build_thread_list(None, None)
        ids = {e.thread_id for e in entries}
        assert ids == {"local-1", "local-2"}

    @pytest.mark.asyncio
    async def test_checkpointer_threads_appear(self):
        """Persisted threads appear from the checkpointer."""
        cp = MemoryCheckpointer()

        from langgraph.graph import StateGraph

        builder = StateGraph(_MessagesState)
        builder.add_node("noop", lambda s: s)
        builder.set_entry_point("noop")
        builder.set_finish_point("noop")
        graph = builder.compile(checkpointer=cp)

        await graph.ainvoke(
            {"messages": [HumanMessage(content="persisted hello")]},
            config={"configurable": {"thread_id": "persisted-1"}},
        )

        tm = ThreadManager()
        entries = await tm._build_thread_list(cp, None)
        assert len(entries) == 1
        assert entries[0].thread_id == "persisted-1"
        assert "persisted hello" in entries[0].last_message

    @pytest.mark.asyncio
    async def test_merge_session_and_checkpointer(self):
        """Session-local threads merge with checkpointer threads."""
        cp = MemoryCheckpointer()

        from langgraph.graph import StateGraph

        builder = StateGraph(_MessagesState)
        builder.add_node("noop", lambda s: s)
        builder.set_entry_point("noop")
        builder.set_finish_point("noop")
        graph = builder.compile(checkpointer=cp)

        await graph.ainvoke(
            {"messages": [HumanMessage(content="persisted")]},
            config={"configurable": {"thread_id": "persisted-1"}},
        )

        tm = ThreadManager()
        tm.record("local-1", "local only")

        entries = await tm._build_thread_list(cp, None)
        ids = {e.thread_id for e in entries}
        assert ids == {"persisted-1", "local-1"}

    @pytest.mark.asyncio
    async def test_no_duplicate_when_session_matches_persisted(self):
        """A thread in both session and checkpointer should not duplicate."""
        cp = MemoryCheckpointer()

        from langgraph.graph import StateGraph

        builder = StateGraph(_MessagesState)
        builder.add_node("noop", lambda s: s)
        builder.set_entry_point("noop")
        builder.set_finish_point("noop")
        graph = builder.compile(checkpointer=cp)

        await graph.ainvoke(
            {"messages": [HumanMessage(content="hello")]},
            config={"configurable": {"thread_id": "shared-1"}},
        )

        tm = ThreadManager()
        tm.record("shared-1", "local copy")

        entries = await tm._build_thread_list(cp, None)
        ids = [e.thread_id for e in entries]
        # Should appear exactly once (checkpointer wins).
        assert ids.count("shared-1") == 1

    @pytest.mark.asyncio
    async def test_current_thread_filtered_out(self):
        """The current thread should not appear in the list."""
        tm = ThreadManager()
        tm.record("current", "my thread")
        tm.record("other", "other thread")

        entries = await tm._build_thread_list(None, current_thread_id="current")
        ids = {e.thread_id for e in entries}
        assert "current" not in ids
        assert "other" in ids
