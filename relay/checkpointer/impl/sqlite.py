"""Async SQLite checkpointer with message indexing.

Wraps ``AsyncSqliteSaver`` with the extended query methods from
``relay.checkpointer.base.BaseCheckpointer`` and adds an automatic message
index table for fast thread and history queries.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from relay.checkpointer.base import BaseCheckpointer, HumanMessageEntry, ThreadSummary

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.messages import BaseMessage
    from langgraph.checkpoint.base import (
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
        SerializerProtocol,
    )

logger = logging.getLogger(__name__)


class IndexedAsyncSqliteSaver(AsyncSqliteSaver, BaseCheckpointer):
    """``AsyncSqliteSaver`` with automatic message indexing for performance.

    On every ``aput`` the index table is updated so that thread listing
    and human-message extraction can use SQL queries instead of
    deserialising every checkpoint.
    """

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        *,
        connection_string: str,
        serde: SerializerProtocol | None = None,
    ) -> AsyncIterator[IndexedAsyncSqliteSaver]:
        """Create and set up an ``IndexedAsyncSqliteSaver``.

        Args:
            connection_string: SQLite database path.
            serde: Optional checkpoint serializer.

        Yields:
            A ready-to-use SQLite saver.

        Example:
            async with IndexedAsyncSqliteSaver.create(connection_string=path) as cp:
                ...
        """
        async with aiosqlite.connect(connection_string) as conn:
            instance = cls(conn, serde=serde)
            await instance.setup()
            yield instance

    # ------------------------------------------------------------------
    # Setup — creates the message-index table
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Initialise tables including the message index."""
        await super().setup()

        async with self.lock:
            await self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoint_messages (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                message_idx INTEGER NOT NULL,
                message_type TEXT NOT NULL,
                message_preview TEXT,
                checkpoint_ts TEXT,
                PRIMARY KEY (thread_id, checkpoint_id, message_idx)
            );
            CREATE INDEX IF NOT EXISTS idx_thread_ns_messages
                ON checkpoint_messages(thread_id, checkpoint_ns, message_type);
            CREATE INDEX IF NOT EXISTS idx_thread_lookup
                ON checkpoints(thread_id, checkpoint_ns);
            CREATE INDEX IF NOT EXISTS idx_checkpoint_id
                ON checkpoints(checkpoint_id);
            """)

            # Migrate existing databases that lack the checkpoint_ts column.
            cursor = await self.conn.execute(
                "PRAGMA table_info(checkpoint_messages)"
            )
            columns = {row[1] for row in await cursor.fetchall()}
            if "checkpoint_ts" not in columns:
                await self.conn.execute(
                    "ALTER TABLE checkpoint_messages ADD COLUMN checkpoint_ts TEXT"
                )

            await self.conn.commit()

        logger.debug("Message index tables created")

    # ------------------------------------------------------------------
    # Checkpoint write — update index on every save
    # ------------------------------------------------------------------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, int | float | str],
    ) -> RunnableConfig:
        """Save checkpoint and update the message index."""
        result = await super().aput(config, checkpoint, metadata, new_versions)
        await self._index_messages(config, checkpoint)
        return result

    # ------------------------------------------------------------------
    # Message indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _get_message_preview(msg) -> str:
        """Return a short preview string for a message."""
        text = getattr(msg, "text", None) or str(getattr(msg, "content", ""))
        return text[:120] if text else ""

    async def _index_messages(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> None:
        """Extract and index messages from a checkpoint."""
        try:
            thread_id = config["configurable"].get("thread_id")
            checkpoint_id = checkpoint.get("id")
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            checkpoint_ts = checkpoint.get("ts", "")

            if not thread_id or not checkpoint_id:
                return

            messages = checkpoint.get("channel_values", {}).get("messages", [])
            if not messages:
                return

            async with self.lock:
                await self.conn.execute(
                    "DELETE FROM checkpoint_messages "
                    "WHERE thread_id = ? AND checkpoint_id = ?",
                    (thread_id, checkpoint_id),
                )

                rows = [
                    (
                        thread_id,
                        checkpoint_id,
                        checkpoint_ns,
                        idx,
                        msg.type,
                        self._get_message_preview(msg),
                        checkpoint_ts,
                    )
                    for idx, msg in enumerate(messages)
                ]

                await self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO checkpoint_messages
                    (thread_id, checkpoint_id, checkpoint_ns,
                     message_idx, message_type, message_preview,
                     checkpoint_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                await self.conn.commit()

        except Exception as e:
            logger.warning("Failed to index messages: %s", e)

    # ------------------------------------------------------------------
    # Thread discovery
    # ------------------------------------------------------------------

    async def get_threads(self) -> set[str]:
        """Return all thread IDs from the database."""
        try:
            async with self.lock:
                cursor = await self.conn.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints "
                    "WHERE checkpoint_ns = ''"
                )
                rows = await cursor.fetchall()
            return {row[0] for row in rows if row[0]}
        except Exception as e:
            logger.error("Failed to get thread IDs: %s", e)
            return set()

    async def get_thread_summaries(self) -> list[ThreadSummary]:
        """Return summaries for all threads, sorted newest-first.

        Uses the ``checkpoint_messages`` index table when available to
        avoid deserialising every checkpoint.  Falls back to reading
        the latest checkpoint per thread when the index is empty.
        """
        try:
            # ----------------------------------------------------------
            # Fast path: pull the latest human message per thread from
            # the index table.
            # ----------------------------------------------------------
            async with self.lock:
                cursor = await self.conn.execute("""
                    SELECT
                        cm.thread_id,
                        cm.message_preview,
                        cm.checkpoint_ts
                    FROM checkpoint_messages cm
                    JOIN (
                        SELECT thread_id, MAX(checkpoint_id) AS latest_cp
                        FROM checkpoints
                        WHERE checkpoint_ns = ''
                        GROUP BY thread_id
                    ) lc ON cm.thread_id = lc.thread_id
                           AND cm.checkpoint_id = lc.latest_cp
                    WHERE cm.message_type = 'human'
                    ORDER BY cm.message_idx DESC
                """)
                rows = await cursor.fetchall()

            # De-duplicate: keep only the last human message per thread.
            seen: dict[str, ThreadSummary] = {}
            for thread_id, preview, ts in rows:
                if thread_id not in seen:
                    seen[thread_id] = ThreadSummary(
                        thread_id=thread_id,
                        last_message=preview or "(no content)",
                        timestamp=ts or "",
                    )

            if seen:
                return sorted(
                    seen.values(), key=lambda s: s.timestamp, reverse=True
                )

            # ----------------------------------------------------------
            # Slow path: index is empty — fall back to deserialising
            # the latest checkpoint per thread.
            # ----------------------------------------------------------
            thread_ids = await self.get_threads()
            summaries: list[ThreadSummary] = []

            for tid in thread_ids:
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

        except Exception as e:
            logger.error("Failed to get thread summaries: %s", e)
            return []

    # ------------------------------------------------------------------
    # History traversal
    # ------------------------------------------------------------------

    async def get_history(self, latest: CheckpointTuple) -> list[CheckpointTuple]:
        """Walk the parent chain back to the root (oldest first)."""
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

    async def delete_after(self, thread_id: str, checkpoint_id: str | None) -> int:
        """Delete checkpoints after *checkpoint_id*.  Returns count deleted."""
        config = RunnableConfig(configurable={"thread_id": thread_id})
        latest = await self.aget_tuple(config)
        if not latest:
            return 0

        history = await self.get_history(latest)

        if checkpoint_id is None:
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

        ids_to_delete = [
            cp_id for cp in to_delete if (cp_id := cp.checkpoint.get("id"))
        ]
        if not ids_to_delete:
            return 0

        placeholders = ",".join("?" * len(ids_to_delete))
        try:
            async with self.lock:
                cursor = await self.conn.execute(
                    f"DELETE FROM checkpoints WHERE thread_id = ? "
                    f"AND checkpoint_id IN ({placeholders})",
                    [thread_id, *ids_to_delete],
                )
                deleted = cursor.rowcount

                await self.conn.execute(
                    f"DELETE FROM checkpoint_messages WHERE thread_id = ? "
                    f"AND checkpoint_id IN ({placeholders})",
                    [thread_id, *ids_to_delete],
                )
                await self.conn.commit()

            return deleted
        except Exception as e:
            logger.error("Failed to delete checkpoints: %s", e)
            return 0

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
        checkpoint_ns = latest.config.get("configurable", {}).get("checkpoint_ns", "")

        # Check whether the index is populated for this thread.
        async with self.lock:
            cursor = await self.conn.execute(
                "SELECT COUNT(*) FROM checkpoint_messages "
                "WHERE thread_id = ? AND checkpoint_ns = ?",
                (thread_id, checkpoint_ns),
            )
            row = await cursor.fetchone()
            has_index = row and row[0] > 0

        # Lazily populate the index if this is the first query.
        if not has_index:
            if on_indexing:
                on_indexing()

            history = await self.get_history(latest)
            for cp_tuple in history:
                cp = cp_tuple.checkpoint
                if cp:
                    cfg = RunnableConfig(
                        configurable={
                            "thread_id": cp_tuple.config.get(
                                "configurable", {}
                            ).get("thread_id"),
                            "checkpoint_ns": cp_tuple.config.get(
                                "configurable", {}
                            ).get("checkpoint_ns", ""),
                        }
                    )
                    await self._index_messages(cfg, cp)

        # Build mapping: message_count → checkpoint_id from the index.
        checkpoint_by_msg_count: dict[int, str] = {}
        async with self.lock:
            cursor = await self.conn.execute(
                """
                SELECT checkpoint_id, MAX(message_idx) + 1 AS msg_count
                FROM checkpoint_messages
                WHERE thread_id = ? AND checkpoint_ns = ?
                GROUP BY checkpoint_id
                ORDER BY msg_count
                """,
                (thread_id, checkpoint_ns),
            )
            rows = await cursor.fetchall()
        for cp_id, msg_count in rows:
            checkpoint_by_msg_count[msg_count] = cp_id

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
