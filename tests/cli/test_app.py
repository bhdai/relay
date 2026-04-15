"""Tests for the relay CLI bootstrap entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from relay.cli.bootstrap.app import create_parser, main


def test_create_parser_supports_agent_model_and_working_dir() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "-w",
            "/tmp/project",
            "-a",
            "claude-style-coder",
            "-m",
            "gpt-5.1-codex-mini",
        ]
    )

    assert args.working_dir == "/tmp/project"
    assert args.agent == "claude-style-coder"
    assert args.model == "gpt-5.1-codex-mini"


@pytest.mark.asyncio
async def test_main_starts_session_with_cli_selection(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr("relay.cli.bootstrap.app.Session", FakeSession)

    exit_code = await main(
        [
            "-w",
            str(tmp_path),
            "-a",
            "code-reviewer",
            "-m",
            "gpt-5.1-codex-mini",
        ]
    )

    assert exit_code == 0
    assert captured["started"] is True
    assert captured["working_dir"] == tmp_path.resolve()
    assert captured["agent_name"] == "code-reviewer"
    assert captured["model_name"] == "gpt-5.1-codex-mini"