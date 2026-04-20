"""Terminal tools for executing shell commands.

Provides the agent with controlled access to the system shell.
"""

import asyncio
import re

from langchain_core.tools import ToolException, tool


# ==============================================================================
# Command prefix extraction
# ==============================================================================


def _command_prefix(command: str) -> str:
    """Extract a meaningful command prefix for 'always' permission patterns.

    Uses a heuristic rather than a full bash AST parser: split on the first
    shell operator to isolate the primary command segment, then take the
    command binary and — if the immediately following token does not start
    with ``-`` — its first subcommand.

    This covers the common cases (``git push``, ``npm install``,
    ``cargo build``) without adding a tree-sitter dependency.

    Examples:
        "git push --force origin main" -> "git push"
        "npm install foo" -> "npm install"
        "ls -la /tmp" -> "ls"
        "rm -rf /" -> "rm"
        "" -> "*"

    Args:
        command: Raw shell command string.

    Returns:
        A prefix string suitable for use in wildcard patterns, e.g.
        ``"git push"`` (the caller appends ``" *"``).  Returns ``"*"``
        when the command is empty.
    """
    # Isolate the primary segment before any ``&&``, ``||``, ``;``, or ``|``.
    primary = re.split(r"\s*(&&|\|\||;|\|)\s*", command.strip(), maxsplit=1)[0]
    parts = primary.split()

    if not parts:
        return "*"

    cmd = parts[0]
    # Include the subcommand only when it looks like a subcommand (no leading "-").
    if len(parts) >= 2 and not parts[1].startswith("-"):
        return f"{cmd} {parts[1]}"
    return cmd


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

# ==============================================================================
# Tool permission configuration
# ==============================================================================
#
# run_command requests the "bash" permission, evaluated against the full
# command string.  "always" patterns use a coarser prefix (e.g. "git push *")
# so that approving one git-push command covers future push variants without
# requiring a separate approval for every flag combination.

run_command.metadata = {
    "permission_config": {
        "permission": "bash",
        # The full command string is the concrete pattern being evaluated.
        "patterns_fn": lambda args: [args.get("command", "*")],
        # The "always" pattern uses the command prefix so that approving
        # "git push" covers "git push --force", "git push origin main", etc.
        "always_fn": lambda args: [
            _command_prefix(args.get("command", "")) + " *"
        ],
        "metadata_fn": lambda args: {"command": args.get("command", "")},
    }
}
