"""Linux bubblewrap (bwrap) sandbox backend.

Wraps tool execution inside a bubblewrap namespace with:
- Filesystem isolation (tmpfs root, explicit read/write mounts)
- Hidden paths mapped to /dev/null or empty tmpfs
- PID, IPC, UTS namespace isolation
- Optional network isolation (``--unshare-net``)
- Optional seccomp BPF filter blocking dangerous syscalls
"""

from __future__ import annotations

import glob as globmod
import logging
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from relay.sandboxes.backend import SandboxBackend
from relay.sandboxes.constants import (
    BWRAP_AF_UNIX,
    BWRAP_BLOCKED_SYSCALLS,
    BWRAP_PTRACE_TRACEME,
)

if TYPE_CHECKING:
    from relay.configs.sandbox import SandboxConfig

logger = logging.getLogger(__name__)

# Seccomp is optional — pyseccomp may not be installed.
SECCOMP_AVAILABLE = False
seccomp: Any = None

try:
    import seccomp  # type: ignore[import-not-found,no-redef]

    SECCOMP_AVAILABLE = True
except ImportError:
    pass


# ==============================================================================
# Path Helpers
# ==============================================================================


def _expand_pattern(pattern: str, working_dir: Path) -> list[Path]:
    """Expand a path pattern (possibly with globs or ``~``) to real paths."""
    expanded = Path(pattern).expanduser()
    base_path = expanded if expanded.is_absolute() else working_dir / expanded

    pattern_str = str(base_path)
    if globmod.has_magic(pattern_str):
        # Find the deepest non-glob prefix and glob from there.
        parts = base_path.parts
        glob_base = Path(parts[0])
        for i, part in enumerate(parts[1:], start=1):
            if globmod.has_magic(part):
                if glob_base.exists():
                    remainder = str(Path(*parts[i:]))
                    return list(glob_base.glob(remainder))
                return []
            glob_base = glob_base / part

    if base_path.exists():
        return [base_path]
    return []


def _matches_hidden(path: Path, hidden_patterns: list[str], working_dir: Path) -> bool:
    """Return True if *path* should be hidden according to *hidden_patterns*."""
    resolved = path.resolve() if path.exists() else path

    for pattern in hidden_patterns:
        expanded = Path(pattern).expanduser()

        if "*" in pattern:
            check = resolved
            while check != check.parent:
                if check.match(pattern):
                    return True
                check = check.parent
        else:
            if not expanded.is_absolute():
                expanded = working_dir / expanded
            pattern_path = expanded.resolve() if expanded.exists() else expanded
            try:
                resolved.relative_to(pattern_path)
                return True
            except ValueError:
                if resolved == pattern_path:
                    return True

    return False


# ==============================================================================
# BubblewrapBackend
# ==============================================================================


class BubblewrapBackend(SandboxBackend):
    """Linux bubblewrap sandbox implementation.

    Requires ``bwrap`` on ``PATH``.  Optionally uses ``pyseccomp``
    for BPF system-call filtering.
    """

    def __init__(
        self,
        config: SandboxConfig,
        working_dir: Path,
    ) -> None:
        super().__init__(config, working_dir)
        self._filter_path: Path | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_environment(self) -> None:
        if not shutil.which("bwrap"):
            raise RuntimeError(
                "bwrap not found in PATH. Install bubblewrap: "
                "apt install bubblewrap"
            )

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def build_command(
        self,
        command: list[str],
        extra_env: dict[str, str] | None = None,
    ) -> list[str]:
        """Build the complete ``bwrap …`` invocation."""
        args = ["bwrap", "--clearenv"]
        args.extend(self._build_env_args(extra_env))

        # Start with an empty root filesystem, then mount only what's allowed.
        args.extend(["--tmpfs", "/", "--tmpfs", "/dev"])
        args.extend(self._build_read_args())
        args.extend(self._build_write_args())
        args.extend(self._build_hidden_args())
        args.extend(self._build_namespace_args())

        base_cmd = args + command

        # If seccomp is available, wrap bwrap inside bash to pass the
        # BPF filter via an fd (bwrap's --seccomp expects an fd number).
        filter_path = self._get_filter_path()
        if not filter_path:
            return base_cmd

        return [
            "bash",
            "-c",
            f'exec 3<{shlex.quote(str(filter_path))} && bwrap "$@"',
            "--",
            "--seccomp",
            "3",
        ] + base_cmd[1:]

    # ------------------------------------------------------------------
    # Argument builders
    # ------------------------------------------------------------------

    def _build_env_args(self, extra_env: dict[str, str] | None = None) -> list[str]:
        args: list[str] = []
        for key, value in self.get_sandbox_env().items():
            args.extend(["--setenv", key, value])
        if extra_env:
            for key, value in extra_env.items():
                args.extend(["--setenv", key, value])
        return args

    def _build_read_args(self) -> list[str]:
        """Build ``--ro-bind`` arguments for read-only paths."""
        args: list[str] = []
        for pattern in self.config.filesystem.read:
            if pattern == ".":
                args.extend(
                    ["--ro-bind", str(self.working_dir), str(self.working_dir)]
                )
                continue
            resolved_paths = _expand_pattern(pattern, self.working_dir)
            if not resolved_paths and "*" in pattern:
                logger.warning("Sandbox read pattern '%s' matched no files", pattern)
            for resolved in resolved_paths:
                if not _matches_hidden(
                    resolved, self.config.filesystem.hidden, self.working_dir
                ):
                    args.extend(["--ro-bind", str(resolved), str(resolved)])
        return args

    def _build_write_args(self) -> list[str]:
        """Build ``--bind`` arguments for writable paths."""
        args: list[str] = []
        for pattern in self.config.filesystem.write:
            if pattern == ".":
                args.extend(["--bind", str(self.working_dir), str(self.working_dir)])
                continue
            resolved_paths = _expand_pattern(pattern, self.working_dir)
            if not resolved_paths and "*" in pattern:
                logger.warning("Sandbox write pattern '%s' matched no files", pattern)
            for resolved in resolved_paths:
                if not _matches_hidden(
                    resolved, self.config.filesystem.hidden, self.working_dir
                ):
                    args.extend(["--bind", str(resolved), str(resolved)])
        return args

    def _build_hidden_args(self) -> list[str]:
        """Build mount overrides that hide sensitive paths."""
        args: list[str] = []
        for pattern in self.config.filesystem.hidden:
            for path in _expand_pattern(pattern, self.working_dir):
                if path.is_file():
                    args.extend(["--ro-bind", "/dev/null", str(path)])
                elif path.is_dir():
                    args.extend(["--tmpfs", str(path)])
        return args

    def _build_namespace_args(self) -> list[str]:
        """Build namespace and process isolation arguments."""
        args = [
            "--unshare-user",
            "--uid",
            str(os.getuid()),
            "--gid",
            str(os.getgid()),
        ]
        args.extend(["--unshare-pid", "--unshare-ipc", "--unshare-uts"])
        args.extend(["--die-with-parent", "--new-session", "--proc", "/proc"])

        chdir_path = self.working_dir if self.includes_working_dir else Path("/")
        args.extend(["--chdir", str(chdir_path)])

        if self._allows_network():
            args.append("--share-net")
        else:
            args.append("--unshare-net")

        return args

    # ------------------------------------------------------------------
    # Seccomp BPF
    # ------------------------------------------------------------------

    def _build_seccomp_filter(self) -> bytes | None:
        """Build a seccomp BPF filter that blocks dangerous syscalls."""
        if not SECCOMP_AVAILABLE:
            return None
        try:
            f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)

            # Block ptrace TRACEME (sandbox escape vector).
            f.add_rule(
                seccomp.ERRNO(1),
                "ptrace",
                seccomp.Arg(0, seccomp.EQ, BWRAP_PTRACE_TRACEME),
            )

            # Block AF_UNIX sockets unless local sockets are configured.
            if not self.config.network.local:
                f.add_rule(
                    seccomp.ERRNO(1),
                    "socket",
                    seccomp.Arg(0, seccomp.EQ, BWRAP_AF_UNIX),
                )

            for syscall_name in BWRAP_BLOCKED_SYSCALLS:
                try:
                    f.add_rule(seccomp.ERRNO(1), syscall_name)
                except Exception:
                    pass

            return f.export_bpf()
        except Exception as e:
            logger.warning("Failed to generate seccomp filter: %s", e)
            return None

    def _get_filter_path(self) -> Path | None:
        """Get or create a cached seccomp BPF filter file."""
        if self._filter_path and self._filter_path.exists():
            return self._filter_path

        bpf_data = self._build_seccomp_filter()
        if not bpf_data:
            return None

        # Write to a temporary file that persists for the lifetime of this backend.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bpf") as f:
            f.write(bpf_data)
            self._filter_path = Path(f.name)

        return self._filter_path
