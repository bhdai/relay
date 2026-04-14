"""Theme-aware Rich console wrapper.

``ThemedConsole`` wraps a Rich ``Console`` pre-configured with the
active theme's ``rich_theme``.  It adds semantic shortcut methods
(``print_error``, ``print_success``, …) so callers never build
style strings by hand.

Langrepl equivalent:
    ``langrepl.cli.theme.console.ThemedConsole``
"""

from __future__ import annotations

from rich.console import Console

from relay.cli.theme.base import BaseTheme


class ThemedConsole:
    """Console wrapper with configurable theme."""

    def __init__(self, theme: BaseTheme) -> None:
        self.console = Console(
            theme=theme.rich_theme,
            force_terminal=True,
        )

    # -- Delegated print ---------------------------------------------------

    def print(self, *args, **kwargs) -> None:  # noqa: A003 – shadows builtin
        """Proxy to ``Console.print``."""
        self.console.print(*args, **kwargs)

    # -- Semantic helpers --------------------------------------------------

    def print_error(self, content: str) -> None:
        """Print an error message with ``✗`` prefix."""
        self.console.print(f"[error]✗[/error] {content}")

    def print_warning(self, content: str) -> None:
        """Print a warning message with ``⚠`` prefix."""
        self.console.print(f"[warning]⚠[/warning] {content}")

    def print_success(self, content: str) -> None:
        """Print a success message with ``✓`` prefix."""
        self.console.print(f"[success]✓[/success] {content}")

    def print_info(self, content: str) -> None:
        """Print an informational message in muted style."""
        self.console.print(content, style="muted")

    # -- Utilities ---------------------------------------------------------

    @property
    def width(self) -> int:
        """Terminal width in columns."""
        return self.console.width
