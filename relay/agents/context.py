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

from relay.configs.approval import ApprovalMode


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
    approval_mode: ApprovalMode = ApprovalMode.SEMI_ACTIVE
    user_memory: str = ""
    tool_output_max_tokens: int = 10_000
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0

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
