"""Tool approval configuration classes.

Defines the approval mode, per-tool rules, and the persistent
``ToolApprovalConfig`` that lives in ``.relay/config.approval.json``.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ApprovalMode(str, Enum):
    """Tool approval mode for interactive sessions.

    Controls how aggressively the agent auto-approves tool invocations:

    - ``SEMI_ACTIVE`` — prompt the user for every call not explicitly
      allowed/denied (default, safest).
    - ``ACTIVE`` — auto-approve unless the tool is in ``always_deny``
      or ``always_ask``.
    - ``AGGRESSIVE`` — auto-approve unless the tool is in
      ``always_deny``.
    """

    SEMI_ACTIVE = "semi-active"
    ACTIVE = "active"
    AGGRESSIVE = "aggressive"


class ToolApprovalRule(BaseModel):
    """A rule matching a specific tool call by name and optional argument patterns."""

    name: str
    args: dict[str, Any] | None = None

    def matches_call(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Return ``True`` if *tool_name* / *tool_args* satisfy this rule."""
        if self.name != tool_name:
            return False

        if not self.args:
            return True

        for key, expected_value in self.args.items():
            if key not in tool_args:
                return False

            actual_value = str(tool_args[key])
            expected_str = str(expected_value)

            if actual_value == expected_str:
                continue

            # Fall back to regex matching.
            try:
                if re.compile(expected_str).search(actual_value):
                    continue
            except re.error:
                pass

            return False

        return True


def _default_always_ask_rules() -> list[ToolApprovalRule]:
    """Default rules for critical commands that always require approval."""
    return [
        ToolApprovalRule(name="run_command", args={"command": r"rm\s+-rf.*"}),
        ToolApprovalRule(name="run_command", args={"command": r"git\s+push.*"}),
        ToolApprovalRule(
            name="run_command", args={"command": r"git\s+reset\s+--hard.*"}
        ),
        ToolApprovalRule(name="run_command", args={"command": r"sudo\s+.*"}),
    ]


class ToolApprovalConfig(BaseModel):
    """Persistent approval configuration stored in JSON.

    Loaded from and saved to ``.relay/config.approval.json``.
    """

    always_allow: list[ToolApprovalRule] = Field(default_factory=list)
    always_deny: list[ToolApprovalRule] = Field(default_factory=list)
    always_ask: list[ToolApprovalRule] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, file_path: Path) -> ToolApprovalConfig:
        """Load from *file_path*, creating the file with defaults if missing."""
        if not file_path.exists():
            config = cls(always_ask=_default_always_ask_rules())
            config.save_to_json_file(file_path)
            return config

        try:
            import json

            with open(file_path) as f:
                raw = json.load(f)

            # Ensure defaults for always_ask if not present.
            if "always_ask" not in raw:
                raw["always_ask"] = [
                    r.model_dump() for r in _default_always_ask_rules()
                ]
                with open(file_path, "w") as f:
                    json.dump(raw, f, indent=2)

            return cls.model_validate(raw)
        except Exception:
            return cls()

    def save_to_json_file(self, file_path: Path) -> None:
        """Persist the current config to *file_path* as JSON."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
