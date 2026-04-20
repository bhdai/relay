"""In-memory checkpointer with extended query methods.

Wraps LangGraph's ``MemorySaver`` with the additional thread discovery,
history, and replay methods defined by
``relay.checkpointer.base.BaseCheckpointer``.

Useful for testing and ephemeral sessions where persistence is not required.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from relay.checkpointer.base import BaseCheckpointer, HumanMessageEntry, ThreadSummary

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langgraph.checkpoint.base import CheckpointTuple

logger = logging.getLogger(__name__)


class MemoryCheckpointer(MemorySaver, BaseCheckpointer):
    """In-memory checkpointer that extends LangGraph's MemorySaver.

    Does not persist across application restarts.
    """

    # ------------------------------------------------------------------
    # Thread discovery
    # ------------------------------------------------------------------

    async def get_threads(self) -> set[str]:
        """Return all thread IDs stored in memory."""
        return set(self.storage.keys())

    async def get_thread_summaries(self) -> list[ThreadSummary]:
        """Return summaries for all in-memory threads, sorted newest-first."""
        summaries: list[ThreadSummary] = []

        for tid in self.storage:
            config = RunnableConfig(configurable={"thread_id": tid})
            checkpoint_tuple = await self.aget_tuple(config)
            if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
                continue

            messages = checkpoint_tuple.checkpoint.get(
                "channel_values", {}
            ).get("messages", [])

            # Find the last human message for the preview.
            preview = "(no messages)"
            for msg in reversed(messages):
                if msg.type == "human":
                    text = getattr(msg, "text", None) or str(
                        getattr(msg, "content", "")
                    )
                    preview = text[:120]
                    break

            ts = checkpoint_tuple.checkpoint.get("ts", "")
            summaries.append(
                ThreadSummary(
                    thread_id=tid,
                    last_message=preview,
                    timestamp=ts or "",
                )
            )

        summaries.sort(key=lambda s: s.timestamp, reverse=True)
        return summaries

    # ------------------------------------------------------------------
    # History traversal
    # ------------------------------------------------------------------

    async def get_history(self, latest: CheckpointTuple) -> list[CheckpointTuple]:
        """Walk the parent chain from *latest* back to the root.

        Returns checkpoints in chronological order (oldest first).
        """
        history: list[CheckpointTuple] = []
        current: CheckpointTuple | None = latest

        while current is not None:
            history.append(current)
            current = (
                await self.aget_tuple(current.parent_config)
                if current.parent_config
                else None
            )

        history.reverse()
        return history

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def _delete_checkpoints(
        self, thread_id: str, checkpoint_ns: str, checkpoint_ids: list[str]
    ) -> int:
        """Delete specific checkpoints from in-memory storage."""
        deleted = 0

        # MemorySaver layout: storage[thread_id][checkpoint_ns][checkpoint_id]
        if thread_id in self.storage and checkpoint_ns in self.storage[thread_id]:
            for cp_id in checkpoint_ids:
                if cp_id in self.storage[thread_id][checkpoint_ns]:
                    del self.storage[thread_id][checkpoint_ns][cp_id]
                    deleted += 1

        # Also delete associated writes: writes[(thread_id, ns, cp_id)]
        for cp_id in checkpoint_ids:
            key = (thread_id, checkpoint_ns, cp_id)
            if key in self.writes:
                del self.writes[key]

        return deleted

    async def delete_after(self, thread_id: str, checkpoint_id: str | None) -> int:
        """Delete all checkpoints after *checkpoint_id*."""
        config = RunnableConfig(configurable={"thread_id": thread_id})
        latest = await self.aget_tuple(config)
        if not latest:
            return 0

        history = await self.get_history(latest)

        if checkpoint_id is None:
            # Delete all checkpoints.
            idx = -1
        else:
            idx_or_none = next(
                (
                    i
                    for i, cp in enumerate(history)
                    if cp.checkpoint.get("id") == checkpoint_id
                ),
                None,
            )
            if idx_or_none is None:
                return 0
            idx = idx_or_none

        to_delete = history[idx + 1 :]
        if not to_delete:
            return 0

        ids_to_delete = {
            cp_id for cp in to_delete if (cp_id := cp.checkpoint.get("id"))
        }

        total = 0
        if thread_id in self.storage:
            for ns in list(self.storage[thread_id].keys()):
                ns_ids = [
                    cp_id
                    for cp_id in self.storage[thread_id][ns]
                    if cp_id in ids_to_delete
                ]
                total += self._delete_checkpoints(thread_id, ns, ns_ids)

        return total

    # ------------------------------------------------------------------
    # Human message extraction
    # ------------------------------------------------------------------

    async def get_human_messages(
        self,
        thread_id: str,
        latest: CheckpointTuple,
        on_indexing: Callable[[], None] | None = None,
    ) -> tuple[list[HumanMessageEntry], list[BaseMessage]]:
        """Extract human messages with replay metadata."""
        if not latest or not latest.checkpoint:
            return [], []

        all_messages = latest.checkpoint.get("channel_values", {}).get("messages", [])
        channel_values = latest.checkpoint.get("channel_values", {})

        # Build a mapping: message_count → checkpoint_id so we can link
        # each human message to the checkpoint that preceded it.
        history = await self.get_history(latest)
        checkpoint_by_msg_count: dict[int, str] = {}
        for cp_tuple in history:
            cp = cp_tuple.checkpoint
            if cp and "channel_values" in cp:
                msg_count = len(cp["channel_values"].get("messages", []))
                checkpoint_by_msg_count[msg_count] = cp.get("id")

        human_messages: list[HumanMessageEntry] = []
        for i, msg in enumerate(all_messages):
            if msg.type == "human":
                human_messages.append(
                    HumanMessageEntry(
                        text=msg.text,
                        messages_before_count=i,
                        checkpoint_id=checkpoint_by_msg_count.get(i),
                        input_tokens=channel_values.get("current_input_tokens"),
                        output_tokens=channel_values.get("current_output_tokens"),
                        total_cost=channel_values.get("total_cost"),
                    )
                )

        return human_messages, all_messages
