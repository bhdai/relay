"""Base theme protocol for the CLI theme system.

Defines the semantic colour interface that every theme must satisfy.
The relay UI layer reads colours through these properties so that
prompt-toolkit styles and Rich styles stay consistent.

Langrepl equivalent:
    ``langrepl.cli.theme.base.BaseTheme``
"""

from __future__ import annotations

from typing import Protocol

from rich.theme import Theme


class BaseTheme(Protocol):
    """Protocol defining the interface for all CLI themes.

    Each property returns a colour string (e.g. ``"#7aa2f7"``).
    ``rich_theme`` is a Rich ``Theme`` instance pre-built from the
    same palette so that ``Console(theme=...)`` picks up semantic
    style names like ``"success"`` or ``"error"``.
    """

    rich_theme: Theme

    # -- Text colours -----------------------------------------------------

    @property
    def primary_text(self) -> str: ...

    @property
    def muted_text(self) -> str: ...

    # -- Background colours -----------------------------------------------

    @property
    def background(self) -> str: ...

    @property
    def background_light(self) -> str: ...

    # -- Semantic colours --------------------------------------------------

    @property
    def success_color(self) -> str: ...

    @property
    def error_color(self) -> str: ...

    @property
    def warning_color(self) -> str: ...

    @property
    def info_color(self) -> str: ...

    # -- UI element colours ------------------------------------------------

    @property
    def prompt_color(self) -> str: ...

    @property
    def accent_color(self) -> str: ...

    @property
    def indicator_color(self) -> str: ...

    @property
    def command_color(self) -> str: ...
