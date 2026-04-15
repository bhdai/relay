"""Tests for CLI message dispatch error handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from relay.cli.core.context import Context
from relay.cli.dispatchers.messages import MessageDispatcher


def _make_session() -> SimpleNamespace:
    return SimpleNamespace(
        context=Context(thread_id="thread-1"),
        graph=object(),
    )


@pytest.mark.asyncio
async def test_dispatch_renders_error_instead_of_raising() -> None:
    """Provider errors should be shown to the user without crashing the REPL."""
    dispatcher = MessageDispatcher(_make_session())
    dispatcher._run_stream = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("relay.cli.dispatchers.messages.render_error") as mock_render_error:
        await dispatcher.dispatch("hello")

    mock_render_error.assert_called_once_with("boom")


@pytest.mark.asyncio
async def test_resume_from_interrupt_renders_error_instead_of_raising() -> None:
    """Resume failures should be handled like normal dispatch failures."""
    dispatcher = MessageDispatcher(_make_session())
    dispatcher.interrupt_handler.handle = AsyncMock(return_value={"intr-1": "yes"})
    dispatcher._run_stream = AsyncMock(side_effect=RuntimeError("resume failed"))

    with patch("relay.cli.dispatchers.messages.render_error") as mock_render_error:
        await dispatcher.resume_from_interrupt([object()])

    mock_render_error.assert_called_once_with("resume failed")


def test_format_stream_error_summarizes_rate_limits() -> None:
    """OpenAI TPM errors should be converted into a compact retry hint."""
    exc = RuntimeError(
        "Rate limit reached for gpt-5.1-codex-mini in organization org-test "
        "on tokens per min (TPM): Limit 200000, Used 189447, Requested 33160. "
        "Please try again in 6.782s."
    )

    message = MessageDispatcher._format_stream_error(exc)

    assert message == (
        "Rate limit reached from the model provider; retry in 6.782s; "
        "request wanted 33160 tokens."
    )