"""Abstract base for sandbox backends.

OS-specific implementations (bubblewrap on Linux) subclass
``SandboxBackend`` and implement :meth:`build_command`.  The base class
owns the ``execute`` method that spawns a sandboxed worker subprocess
via stdin/stdout JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from relay.sandboxes.constants import (
    MAX_STDERR,
    MAX_STDOUT,
    SANDBOX_ENV_BASE,
    WORKER_MODULE,
)

if TYPE_CHECKING:
    from relay.configs.sandbox import SandboxConfig, SandboxType

logger = logging.getLogger(__name__)


# ==============================================================================
# Sandbox Binding
# ==============================================================================
#
# Represents a resolved pattern → backend association.  The
# ``SandboxFactory`` produces a list of these; the ``SandboxMiddleware``
# walks them in order to find the first matching backend for each tool.


class SandboxBinding(BaseModel):
    """Binds tool patterns to a sandbox backend instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    patterns: list[str]
    backend: SandboxBackend | None


# ==============================================================================
# SandboxBackend ABC
# ==============================================================================


class SandboxBackend(ABC):
    """Protocol every sandbox backend must satisfy.

    Concrete backends (``BubblewrapBackend``) only need to implement
    ``build_command`` and ``validate_environment``.  The ``execute``
    method is inherited from here and handles the worker subprocess
    lifecycle.
    """

    def __init__(
        self,
        config: SandboxConfig,
        working_dir: Path,
    ) -> None:
        self.config = config
        self.working_dir = self._resolve_working_dir(working_dir)

    @staticmethod
    def _resolve_working_dir(working_dir: Path) -> Path:
        resolved = working_dir.resolve()
        if not resolved.exists():
            raise ValueError(f"Working directory does not exist: {working_dir}")
        if not resolved.is_dir():
            raise ValueError(f"Working directory is not a directory: {working_dir}")
        return resolved

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def type(self) -> SandboxType:
        return self.config.type

    @property
    def includes_working_dir(self) -> bool:
        """True when "." appears in read or write path lists."""
        return "." in self.config.filesystem.read or "." in self.config.filesystem.write

    def _allows_network(self) -> bool:
        return "*" in self.config.network.remote or bool(self.config.network.local)

    def get_sandbox_env(self) -> dict[str, str]:
        """Environment variables passed into the sandboxed process."""
        home = os.environ.get("HOME", str(Path.home()))
        user_paths = f"{home}/.local/bin"
        base_path = SANDBOX_ENV_BASE.get("PATH", "")
        return {
            "HOME": home,
            **SANDBOX_ENV_BASE,
            "PATH": f"{user_paths}:{base_path}",
        }

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_command(
        self,
        command: list[str],
        extra_env: dict[str, str] | None = None,
    ) -> list[str]:
        """Wrap *command* in the sandbox invocation (e.g. bwrap …)."""

    @abstractmethod
    def validate_environment(self) -> None:
        """Raise if the sandbox binary is unavailable on this OS."""

    # ------------------------------------------------------------------
    # Output collection helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _collect_output(
        stream: asyncio.StreamReader | None,
        max_size: int,
    ) -> tuple[bytes, bool]:
        """Read *stream* until EOF, truncating after *max_size* bytes."""
        if not stream:
            return b"", False
        chunks: list[bytes] = []
        size = 0
        truncated = False
        while chunk := await stream.read(65536):
            if size < max_size:
                chunks.append(chunk)
                size += len(chunk)
            else:
                truncated = True
        return b"".join(chunks), truncated

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        *,
        module_path: str,
        tool_name: str,
        args: dict[str, Any],
        tool_runtime: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Run a tool function inside the sandbox via the worker module.

        The worker receives a JSON request on stdin and writes a JSON
        response to stdout.  The sandbox command wraps the Python
        interpreter invocation with OS-level isolation.
        """

        # ----- Serialize the request -----
        try:
            request = json.dumps(
                {
                    "module": module_path,
                    "tool_name": tool_name,
                    "args": args,
                    "tool_runtime": tool_runtime,
                },
                default=str,
            )
        except (TypeError, ValueError) as e:
            return {"success": False, "error": f"Cannot serialize tool args: {e}"}

        # ----- Build the sandboxed command -----
        sandbox_cmd = self.build_command([sys.executable, "-m", WORKER_MODULE])
        logger.debug(
            "Executing in %s [%s]: %s", self.name, self.type, " ".join(sandbox_cmd)
        )

        cwd = (
            str(self.working_dir)
            if self.includes_working_dir
            else str(Path(sys.executable).parent)
        )

        # ----- Spawn and communicate -----
        try:
            process = await asyncio.create_subprocess_exec(
                *sandbox_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=True,
            )

            if process.stdin:
                process.stdin.write(request.encode())
                await process.stdin.drain()
                process.stdin.close()

            stdout_task = asyncio.create_task(
                self._collect_output(process.stdout, MAX_STDOUT)
            )
            stderr_task = asyncio.create_task(
                self._collect_output(process.stderr, MAX_STDERR)
            )

            try:
                (stdout, stdout_truncated), (stderr, _) = await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task),
                    timeout=timeout,
                )
                await process.wait()
            except TimeoutError:
                # Kill the entire process group so child processes don't linger.
                try:
                    if process.pid:
                        os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                process.kill()
                await process.wait()
                for task in (stdout_task, stderr_task):
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                return {
                    "success": False,
                    "error": f"Sandbox execution timed out after {timeout}s",
                }

            if stdout_truncated:
                return {
                    "success": False,
                    "error": (
                        f"Output exceeded {MAX_STDOUT // (1024 * 1024)}MB limit"
                    ),
                }

            if process.returncode != 0:
                return {
                    "success": False,
                    "error": (
                        f"{self.name} failed with exit code {process.returncode}"
                    ),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                }

            try:
                return json.loads(stdout.decode("utf-8"))
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse worker output",
                    "stdout": stdout.decode("utf-8", errors="replace"),
                }

        except Exception as e:
            return {"success": False, "error": str(e)}
