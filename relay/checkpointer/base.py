"""Base checkpointer with extended query methods.

Relay's ``BaseCheckpointer`` extends LangGraph's
``BaseCheckpointSaver`` with methods for thread discovery, history
traversal, and replay metadata — operations that the CLI needs for
``/resume`` and thread listing but that the stock savers do not
provide.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver as _BaseCheckpointSaver

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langgraph.checkpoint.base import CheckpointTuple


@dataclass
class ThreadSummary:
    """Summary of a persisted thread for the ``/resume`` thread picker.

    Populated by :meth:`BaseCheckpointer.get_thread_summaries` from
    checkpoint data so the CLI can display meaningful previews.
    """

    thread_id: str
    last_message: str
    timestamp: str = ""


@dataclass
class HumanMessageEntry:
    """A human message with replay metadata.

    Used by the CLI to display thread history and to support replaying
    from a specific checkpoint.
    """

    text: str
    messages_before_count: int
    checkpoint_id: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_cost: float | None = None


class BaseCheckpointer(_BaseCheckpointSaver):
    """Base checkpointer with additional query methods.

    Subclasses must implement the four ``async`` methods below in
    addition to the standard ``aget``/``aput`` from LangGraph.
    """

    async def get_threads(self) -> set[str]:
        """Return all known thread IDs."""
        raise NotImplementedError

    async def get_thread_summaries(self) -> list[ThreadSummary]:
        """Return summaries of all threads, sorted newest-first.

        Each summary includes the thread ID, a preview of the last
        human message, and the checkpoint timestamp.  The CLI uses
        this to display a meaningful thread picker in ``/resume``.
        """
        raise NotImplementedError

    async def get_history(self, latest: CheckpointTuple) -> list[CheckpointTuple]:
        """Return checkpoint history in chronological order (oldest first)."""
        raise NotImplementedError

    async def delete_after(self, thread_id: str, checkpoint_id: str | None) -> int:
        """Delete checkpoints after *checkpoint_id*.  Returns count deleted."""
        raise NotImplementedError

    async def get_human_messages(
        self,
        thread_id: str,
        latest: CheckpointTuple,
        on_indexing: Callable[[], None] | None = None,
    ) -> tuple[list[HumanMessageEntry], list[BaseMessage]]:
        """Return human messages with replay metadata.

        Returns
        -------
        tuple
            ``(human_messages, all_messages)``
        """
        raise NotImplementedError
