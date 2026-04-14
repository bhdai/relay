"""Tool factory — centralised tool registry with module maps.

``ToolFactory`` replaces the flat ``_TOOL_GROUPS`` dict that previously
lived in ``relay.agents.factory``.  It registers every tool alongside
its source module name so that the agent factory can resolve declarative
patterns like ``file_system:read_file`` or ``terminal:*`` without
hardcoding a mapping table.

The factory distinguishes two categories:

- **impl** tools: callable by the LLM (filesystem, terminal, web).
- **internal** tools: manage agent state but are not directly callable
  by the LLM (memory, todo).

This mirrors langrepl's ``tools/factory.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


class ToolFactory:
    """Central registry for impl and internal tools with module maps.

    The module map records which source module (e.g. ``"file_system"``,
    ``"terminal"``) each tool came from.  This is the key that makes
    two-part pattern matching (``module:tool_name``) work.

    Tool group imports are deferred to ``__init__`` to avoid circular
    imports (tool modules depend on ``relay.agents.state`` which
    re-exports through ``relay.agents.__init__``).
    """

    def __init__(self) -> None:
        # Deferred imports to break the circular dependency chain:
        # relay.tools.factory → relay.tools.internal.memory →
        # relay.agents.state → relay.agents → relay.agents.factory →
        # relay.tools.factory
        from relay.tools.catalog import CATALOG_TOOLS, SKILL_CATALOG_TOOLS
        from relay.tools.impl.filesystem import FILE_SYSTEM_TOOLS
        from relay.tools.impl.terminal import TERMINAL_TOOLS
        from relay.tools.impl.web import WEB_TOOLS
        from relay.tools.internal.memory import MEMORY_TOOLS
        from relay.tools.internal.todo import TODO_TOOLS

        self.impl_tools: list[BaseTool] = []
        self.internal_tools: list[BaseTool] = []
        self.catalog_tools: list[BaseTool] = list(CATALOG_TOOLS)
        self.skill_catalog_tools: list[BaseTool] = list(SKILL_CATALOG_TOOLS)
        self._impl_module_map: dict[str, str] = {}
        self._internal_module_map: dict[str, str] = {}

        # Register impl tool groups.  The module name is derived from
        # each tool's defining module (the last component of its
        # ``__module__``).
        for tool_group in [FILE_SYSTEM_TOOLS, TERMINAL_TOOLS, WEB_TOOLS]:
            for t in tool_group:
                func = getattr(t, "func", None) or getattr(t, "coroutine", None)
                if func:
                    module_name = func.__module__.split(".")[-1]
                    self._impl_module_map[t.name] = module_name
            self.impl_tools.extend(tool_group)

        # Register internal tool groups.
        for tool_group in [MEMORY_TOOLS, TODO_TOOLS]:
            for t in tool_group:
                func = getattr(t, "func", None) or getattr(t, "coroutine", None)
                if func:
                    module_name = func.__module__.split(".")[-1]
                    self._internal_module_map[t.name] = module_name
            self.internal_tools.extend(tool_group)

    def get_impl_tools(self) -> list[BaseTool]:
        """Return all implementation tools."""
        return self.impl_tools

    def get_internal_tools(self) -> list[BaseTool]:
        """Return all internal (state-management) tools."""
        return self.internal_tools

    def get_impl_module_map(self) -> dict[str, str]:
        """Return tool-name → module-name mapping for impl tools."""
        return self._impl_module_map

    def get_internal_module_map(self) -> dict[str, str]:
        """Return tool-name → module-name mapping for internal tools."""
        return self._internal_module_map

    def get_catalog_tools(self) -> list[BaseTool]:
        """Return the tool-catalog meta-tools (fetch_tools, get_tool, run_tool)."""
        return self.catalog_tools

    def get_skill_catalog_tools(self) -> list[BaseTool]:
        """Return the skill-catalog meta-tools (fetch_skills, get_skill)."""
        return self.skill_catalog_tools
