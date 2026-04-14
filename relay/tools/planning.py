"""Planning tools for deliberate reflection during long tasks.

The ``think`` tool now lives in ``relay.tools.subagents.task`` alongside
the ``task`` tool.  This module re-exports it for backward compatibility.
"""

from relay.tools.subagents.task import think

PLANNING_TOOLS = [think]