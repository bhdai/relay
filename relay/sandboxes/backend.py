"""Abstract base for sandbox backends.

OS-specific implementations (bubblewrap on Linux, seatbelt on macOS)
subclass ``SandboxBackend`` and implement :meth:`execute`.  The
``SandboxMiddleware`` calls :meth:`execute` to run tool code inside a
restricted subprocess.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SandboxBackend(ABC):
    """Protocol every sandbox backend must satisfy."""

    @abstractmethod
    async def execute(
        self,
        *,
        module_path: str,
        tool_name: str,
        args: dict[str, Any],
        tool_runtime: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a tool function inside the sandbox.

        Parameters
        ----------
        module_path:
            Fully-qualified Python module containing the function.
        tool_name:
            Name of the function to invoke.
        args:
            Keyword arguments to pass to the function.
        tool_runtime:
            Serialised ``ToolRuntime`` context (state, config, context).

        Returns
        -------
        dict
            Must include ``"success"`` (bool).  On success, ``"content"``
            holds the tool output.  On failure, ``"error"`` and
            optionally ``"traceback"`` / ``"stderr"`` describe what went
            wrong.
        """
        ...
