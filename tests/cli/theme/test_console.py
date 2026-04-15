"""Tests for themed CLI style resolution."""

from __future__ import annotations

import pytest

from relay.cli.theme import console
from relay.cli.ui.renderer import render_error, render_tool_call, render_tool_error


@pytest.mark.parametrize("name", ["error", "warning", "indicator"])
def test_get_style_allows_bold_semantic_styles(name: str) -> None:
    """Semantic theme styles should compose with bold without parse errors."""
    style = console.get_style(name, bold=True)

    assert style.bold is True
    assert style.color is not None


def test_error_renderers_do_not_raise_for_semantic_styles() -> None:
    """Error renderers should use resolved theme styles instead of raw strings."""
    render_tool_error("broken_tool", "boom")
    render_error("boom")


def test_render_tool_call_header_uses_resolved_indicator_style() -> None:
    """Tool call headers should render without Rich style parsing failures."""
    render_tool_call("search", {"query": "relay"})