"""Tests for CLI session defaults."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from relay.cli.core.session import Session


def test_session_defaults_pricing_from_settings() -> None:
    """The default CLI context should inherit pricing from settings."""
    fake_settings = SimpleNamespace(
        llm=SimpleNamespace(
            input_cost_per_mtok=0.4,
            output_cost_per_mtok=1.6,
        )
    )

    with patch("relay.cli.core.session.get_settings", return_value=fake_settings):
        session = Session()

    assert session.context.input_cost_per_mtok == 0.4
    assert session.context.output_cost_per_mtok == 1.6