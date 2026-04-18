"""Tests for CLI slash command dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from relay.cli.core.context import Context
from relay.cli.dispatchers.commands import CommandDispatcher
from relay.configs.approval import ApprovalMode


def _make_session() -> SimpleNamespace:
    return SimpleNamespace(
        context=Context(thread_id="thread-1", approval_mode=ApprovalMode.SEMI_ACTIVE),
        prompt=SimpleNamespace(refresh_style=Mock()),
        threads=SimpleNamespace(record=Mock()),
    )


@pytest.mark.asyncio
async def test_approval_command_cycles_mode() -> None:
    session = _make_session()
    dispatcher = CommandDispatcher(session)

    with patch("relay.cli.dispatchers.commands.render_info") as mock_render_info:
        should_exit = await dispatcher.dispatch("/approval")

    assert should_exit is False
    assert session.context.approval_mode == ApprovalMode.ACTIVE
    session.prompt.refresh_style.assert_called_once()
    mock_render_info.assert_called_once_with("Approval mode: active")


@pytest.mark.asyncio
async def test_approval_command_sets_specific_mode() -> None:
    session = _make_session()
    dispatcher = CommandDispatcher(session)

    with patch("relay.cli.dispatchers.commands.render_info") as mock_render_info:
        should_exit = await dispatcher.dispatch("/approval aggressive")

    assert should_exit is False
    assert session.context.approval_mode == ApprovalMode.AGGRESSIVE
    session.prompt.refresh_style.assert_called_once()
    mock_render_info.assert_called_once_with("Approval mode: aggressive")


@pytest.mark.asyncio
async def test_approval_command_rejects_invalid_mode() -> None:
    session = _make_session()
    dispatcher = CommandDispatcher(session)

    with patch("relay.cli.dispatchers.commands.render_error") as mock_render_error:
        should_exit = await dispatcher.dispatch("/approval nope")

    assert should_exit is False
    assert session.context.approval_mode == ApprovalMode.SEMI_ACTIVE
    mock_render_error.assert_called_once_with(
        "Invalid approval mode. Use: semi-active, active, aggressive"
    )
