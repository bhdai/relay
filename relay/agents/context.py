"""Out-of-band runtime context passed to the agent graph.

``AgentContext`` carries request-scoped values (working directory,
platform info, cost rates) that tools and prompt templates can read
but that should *not* live inside the conversation message list.

Pass it via ``context=AgentContext(…)`` when invoking the graph, and
declare it with ``context_schema=AgentContext`` in
``create_react_agent``.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    working_dir: str = Field(default_factory=lambda: str(Path.cwd()))
    platform: str = Field(default_factory=lambda: sys.platform)
    shell: str = Field(
        default_factory=lambda: os.environ.get("SHELL", ""),
    )
    current_date_time_zoned: str = Field(
        default_factory=lambda: datetime.now().astimezone().isoformat()
    )
    # The accumulated per-session always-approved permission rules, stored in
    # serialisable dict form so LangGraph can checkpoint them alongside
    # conversation state.  Normalisation into PermissionRule objects happens
    # at middleware construction time, not here.
    #
    # Each entry has the shape:
    #   {"permission": str, "pattern": str, "action": "allow" | "deny" | "ask"}
    permission_ruleset: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Session-accumulated always-approved permission rules in "
            "serialisable dict form.  Set by the factory from agent config "
            "and grown at runtime by PermissionMiddleware 'always' replies."
        ),
    )
    # NOTE: PermissionService is intentionally NOT stored on AgentContext.
    # It holds asyncio state and is not serialisable.  The middleware
    # constructs it once per session from permission_ruleset at startup.
    user_memory: str = ""
    tool_output_max_tokens: int = 10_000
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    tool_catalog: list[Any] = Field(default_factory=list, exclude=True)
    skill_catalog: list[Any] = Field(default_factory=list, exclude=True)

    @property
    def template_vars(self) -> dict[str, Any]:
        """Variables available for system-prompt templating."""
        return {
            "working_dir": self.working_dir,
            "platform": self.platform,
            "shell": self.shell,
            "current_date_time_zoned": self.current_date_time_zoned,
            "user_memory": self.user_memory,
        }
