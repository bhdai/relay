"""Terminal tools for executing shell commands.

Provides the agent with controlled access to the system shell.
"""

import asyncio
import re

from langchain_core.tools import ToolException, tool


# ==============================================================================
# Command Parsing Helpers
# ==============================================================================
#
# These helpers break compound shell commands into individual parts so that
# each segment can be inspected independently (e.g. for logging or policy
# checks).  They handle ``&&``, ``||``, ``;``, ``|``, and nested ``$(…)``
# / backtick substitutions.

_CHAIN_OPS = re.compile(r"\s*(&&|\|\||;|\|)\s*")
_SUBST_DOLLAR = re.compile(r"\$\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
_SUBST_BACKTICK = re.compile(r"`([^`]+)`")


def _extract_command_parts(command: str) -> list[str]:
    """Return every atomic command segment in *command*.

    Handles chained operators (``&&``, ``||``, ``;``, ``|``) and nested
    command substitutions (``$(…)`` and backticks).
    """
    parts: list[str] = []
    for seg in _CHAIN_OPS.split(command):
        seg = seg.strip()
        if not seg or seg in ("&&", "||", ";", "|"):
            continue
        parts.append(seg)
        for pattern in (_SUBST_DOLLAR, _SUBST_BACKTICK):
            for m in pattern.finditer(seg):
                parts.extend(_extract_command_parts(m.group(1)))
    return parts


def _format_output(stdout: str, stderr: str) -> str:
    """Merge *stdout* and *stderr* into a single result string."""
    sections: list[str] = []
    if stdout.strip():
        sections.append(stdout.strip())
    if stderr.strip():
        sections.append(stderr.strip())
    return "\n".join(sections) if sections else "Command completed successfully"


# ==============================================================================
# Tools
# ==============================================================================


@tool
async def run_command(command: str) -> str:
    """Execute a shell command and return its output.

    Use for tasks the filesystem tools cannot perform: running build
    systems, package managers (uv, npm, cargo), test runners, git, etc.
    For reading, searching, or editing files, prefer the dedicated
    filesystem tools (``read_file``, ``glob_files``, ``grep_files``,
    ``edit_file``) — they are safer, auto-ignore noisy directories, and
    produce cleaner output.

    Commands run in the working directory shown in ``<env>``.
    Non-zero exit codes raise a ``ToolException`` with the stderr output.
    Interactive programs that prompt for input will hang; avoid them.

    Args:
        command: The shell command to execute.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
    except Exception as exc:
        raise ToolException(f"run command: {exc}") from exc

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        error_msg = stderr.strip() if stderr.strip() else f"Command failed with exit code {proc.returncode}"
        raise ToolException(error_msg)

    return _format_output(stdout, stderr)


TERMINAL_TOOLS = [run_command]
