"""CLI theme — one hardcoded Tokyo Night palette for v1.

Exports:
    ``theme`` — the active ``BaseTheme`` instance.
    ``console`` — a ``ThemedConsole`` pre-wired with that theme.

Relay does not need multi-theme support, auto-detection, or a theme
registry yet.  When it does, this module is the place to add them.

Langrepl equivalent:
    ``langrepl.cli.theme.__init__``
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.style import Style
from rich.theme import Theme

from relay.cli.theme.base import BaseTheme
from relay.cli.theme.console import ThemedConsole


# ==============================================================================
# Tokyo Night Colour Palette
# ==============================================================================
#
# Ported from langrepl's ``tokyo_night.py``.  Only the subset of
# colours that relay actually references today is included; add more
# as the UI grows.


@dataclass
class _TokyoNightColors:
    """Raw hex colours for the Tokyo Night palette."""

    deep_blue: str = "#1a1b26"
    dark_blue: str = "#24283b"
    light_blue_white: str = "#c0caf5"
    muted_blue: str = "#9aa5ce"
    blue_gray: str = "#565f89"
    dark_gray: str = "#414868"

    bright_blue: str = "#7aa2f7"
    cyan: str = "#7dcfff"
    purple: str = "#bb9af7"
    teal: str = "#8be4e1"
    yellow: str = "#e4e38b"
    pink: str = "#e48be4"
    orange: str = "#ff9e64"
    sky_blue: str = "#89ddff"


class TokyoNightTheme:
    """Tokyo Night theme — satisfies ``BaseTheme``."""

    def __init__(self) -> None:
        self._c = _TokyoNightColors()
        self.rich_theme = self._build_rich_theme()

    # -- Rich theme --------------------------------------------------------

    def _build_rich_theme(self) -> Theme:
        c = self._c
        return Theme(
            {
                # Basic text
                "default": Style(color=c.light_blue_white),
                "primary": Style(color=c.light_blue_white),
                "muted": Style(color=c.blue_gray),
                # Semantic
                "success": Style(color=c.teal),
                "warning": Style(color=c.yellow),
                "error": Style(color=c.pink),
                "info": Style(color=c.bright_blue),
                # UI elements
                "prompt": Style(color=c.bright_blue, bold=True),
                "accent": Style(color=c.bright_blue, bold=True),
                "command": Style(color=c.purple),
                "indicator": Style(color=c.teal),
                # Code
                "code": Style(color=c.teal),
            }
        )

    # -- BaseTheme semantic colour accessors --------------------------------

    @property
    def primary_text(self) -> str:
        return self._c.light_blue_white

    @property
    def muted_text(self) -> str:
        return self._c.blue_gray

    @property
    def background(self) -> str:
        return self._c.deep_blue

    @property
    def background_light(self) -> str:
        return self._c.dark_blue

    @property
    def success_color(self) -> str:
        return self._c.teal

    @property
    def error_color(self) -> str:
        return self._c.pink

    @property
    def warning_color(self) -> str:
        return self._c.yellow

    @property
    def info_color(self) -> str:
        return self._c.bright_blue

    @property
    def prompt_color(self) -> str:
        return self._c.bright_blue

    @property
    def accent_color(self) -> str:
        return self._c.cyan

    @property
    def indicator_color(self) -> str:
        return self._c.teal

    @property
    def command_color(self) -> str:
        return self._c.purple


# ==============================================================================
# Module-level singletons
# ==============================================================================

theme: BaseTheme = TokyoNightTheme()
console: ThemedConsole = ThemedConsole(theme)
