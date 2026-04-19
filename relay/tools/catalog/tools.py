"""Tool catalog — dynamic tool discovery and invocation.

These meta-tools let the LLM discover, inspect, and invoke tools from
its ``AgentContext.tool_catalog`` at runtime.  This is useful when the
full tool surface is too large to attach directly, and a
``use_catalog: true`` config flag routes tools through these three
intermediaries instead.
"""

from __future__ import annotations

import json
import re

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.types import Command

from relay.agents.context import AgentContext
from relay.tools.schema import ToolSchema


@tool
async def fetch_tools(
    runtime: ToolRuntime[AgentContext], pattern: str | None = None
) -> str:
    """Discover and search for available tools in the catalog.

    WITHOUT pattern: Returns ALL available tools.
    WITH pattern: Returns ONLY matching tools (case-insensitive regex on names and descriptions).

    Args:
        pattern: Optional regex pattern to filter tools.

    Returns:
        Newline-separated list of matching tool names, sorted alphabetically.
    """
    tools = runtime.context.tool_catalog

    if pattern is None:
        return "\n".join(sorted(t.name for t in tools))

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ToolException(f"Invalid regex pattern: {e}") from e

    matches = []
    for t in tools:
        if regex.search(t.name):
            matches.append(t.name)
        elif t.description and regex.search(t.description):
            matches.append(t.name)

    if not matches:
        return "No tools found matching pattern"

    return "\n".join(sorted(matches))


fetch_tools.metadata = {"permission_config": {"permission": "fetch_tools"}}


@tool
async def get_tool(tool_name: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Get a tool's documentation and parameters.

    Args:
        tool_name: Name of the tool (from fetch_tools output).

    Returns:
        JSON with: name, description, parameters.
    """
    tools = runtime.context.tool_catalog
    found = next((t for t in tools if t.name == tool_name), None)

    if not found:
        raise ToolException(f"Tool '{tool_name}' not found")

    schema = ToolSchema.from_tool(found).model_dump()
    return json.dumps(schema, indent=2)


get_tool.metadata = {"permission_config": {"permission": "get_tool"}}


@tool
async def run_tool(
    tool_name: str, tool_args: dict, runtime: ToolRuntime[AgentContext]
) -> str | ToolMessage | Command:
    """Execute a tool from the catalog with the specified arguments.

    Args:
        tool_name: Name of the tool to run.
        tool_args: Dictionary of arguments matching the tool's input schema.

    Returns:
        The result of the tool execution.
    """
    tools = runtime.context.tool_catalog
    underlying_tool = next((t for t in tools if t.name == tool_name), None)

    if not underlying_tool:
        raise ToolException(f"Tool '{tool_name}' not found")

    tool_expects_runtime = False
    if underlying_tool.args_schema is not None and hasattr(
        underlying_tool.args_schema, "model_fields"
    ):
        tool_expects_runtime = "runtime" in underlying_tool.args_schema.model_fields

    invoke_args = {**tool_args}
    if tool_expects_runtime:
        invoke_args["runtime"] = runtime

    result = await underlying_tool.ainvoke(invoke_args)
    return result


run_tool.metadata = {
    "permission_config": {
        # The proxy looks through to the underlying tool's permission_config.
        # The PermissionMiddleware uses this flag to resolve the real tool
        # from AgentContext.tool_catalog and evaluate its permission semantics.
        "is_catalog_proxy": True,
    }
}


CATALOG_TOOLS = [fetch_tools, get_tool, run_tool]
