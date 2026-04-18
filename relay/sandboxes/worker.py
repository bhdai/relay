"""Sandbox worker — runs inside the sandboxed environment.

This script is the entry point executed by `python -m relay.sandboxes.worker`
inside a bubblewrap namespace.  It reads a JSON request from stdin,
imports the specified tool module, invokes the tool, and writes a JSON
response to stdout.

The protocol is intentionally simple (single JSON object in, single
JSON object out) to minimize the attack surface of the sandbox boundary.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import signal
import sys
import traceback
from typing import Any

from relay.sandboxes.constants import ALLOWED_MODULE_PREFIX


def _serialize_result(result: Any) -> dict:
    """Serialize a tool result to a JSON-compatible dict."""

    # LangChain ToolMessage.
    if hasattr(result, "content") and hasattr(result, "name"):
        return {
            "success": True,
            "content": result.content,
            "name": getattr(result, "name", None),
            "status": getattr(result, "status", None),
            "is_error": getattr(result, "is_error", False),
        }

    # LangGraph Command.
    if hasattr(result, "update") and hasattr(result, "goto"):
        from dataclasses import asdict

        return {"success": True, "is_command": True, **asdict(result)}

    return {"success": True, "content": str(result)}


async def _run(
    module_path: str,
    tool_name: str,
    args: dict[str, Any],
    tool_runtime: dict[str, Any] | None = None,
) -> dict:
    """Import and invoke a tool inside the sandbox.

    Only modules under ``ALLOWED_MODULE_PREFIX`` can be loaded.  This
    prevents the sandboxed process from importing arbitrary code.
    """

    # ----- Module allowlist check -----
    if not module_path.startswith(ALLOWED_MODULE_PREFIX):
        return {
            "success": False,
            "error": f"Module '{module_path}' not in allowed prefix '{ALLOWED_MODULE_PREFIX}'",
        }

    try:
        module = importlib.import_module(module_path)
        tool = getattr(module, tool_name)

        if not hasattr(tool, "ainvoke"):
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is not a LangChain tool (no ainvoke)",
            }

        # TODO: Inject deserialized ToolRuntime when the worker runtime
        # protocol is fully fleshed out.  For now tools that need the
        # runtime (e.g. to read AgentContext.working_dir) will receive
        # an empty runtime or None.

        return _serialize_result(await tool.ainvoke(args))

    except Exception as e:
        tb = traceback.format_exc()
        sys.stderr.write(f"Worker error: {e}\n{tb}\n")
        sys.stderr.flush()
        return {"success": False, "error": str(e), "traceback": tb}


def main() -> None:
    """Entry point: read JSON request from stdin, write JSON response to stdout."""

    # Graceful shutdown on SIGTERM (bwrap sends this on --die-with-parent).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON input: {e}"}))
        sys.exit(1)

    module_path = request.get("module")
    tool_name = request.get("tool_name")

    if not module_path or not tool_name:
        print(json.dumps({"success": False, "error": "Missing 'module' or 'tool_name'"}))
        sys.exit(1)

    result = asyncio.run(
        _run(
            module_path,
            tool_name,
            request.get("args", {}),
            request.get("tool_runtime"),
        )
    )

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
