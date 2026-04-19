"""Permission config normalization — YAML dict → Ruleset.

This module converts the human-readable YAML permission configuration
(stored in agent YAML files or code defaults) into the flat ``Ruleset``
format that ``evaluate()`` and ``PermissionService`` consume.

Config format (YAML)::

    permission:
      # Shorthand: a single action for all patterns under this key.
      bash: allow

      # Pattern map: different actions for specific patterns.
      read:
        "*.env": ask
        "*.env.*": ask
        "*.env.example": allow
        "*": allow

Rules preserve insertion order, which matters: ``evaluate()`` uses
last-match-wins semantics, so rules listed later override earlier ones.
The pattern map form therefore lets you write coarse rules first and
fine-grained overrides last.

The ``migrate_from_approval_config`` helper translates the legacy
``ToolApprovalConfig`` format (``.relay/config.approval.json``) into a
``Ruleset``.  It is used on first startup when no ``.relay/permission.json``
exists yet.
"""

from __future__ import annotations

import os
from typing import Any

from relay.permission.schema import PermissionAction, PermissionRule, Ruleset

# ==============================================================================
# Default permission rulesets
# ==============================================================================

# Default permissions modeled on Opencode's agent.ts defaults.
# The dict is ordered; last-match-wins means more specific entries placed later
# in the same pattern map override earlier broad ones.
DEFAULT_PERMISSION: dict[str, Any] = {
    # Base: allow all permissions unless overridden below.
    "*": "allow",
    # Ask before triggering repeated-loop detection (future cross-cutting
    # enforcement; define the key now so agents can configure it).
    "doom_loop": "ask",
    # Ask before accessing any path outside the working directory.
    "external_directory": {
        "*": "ask",
    },
    # Shell commands and filesystem mutations are the primary side effects
    # exposed by the built-in toolset.  In the default semi-active flow they
    # should prompt unless the session mode or agent config explicitly
    # re-allows them.
    "bash": "ask",
    "edit": "ask",
    # Fine-grained read controls: allow most files but ask for sensitive
    # environment-variable files that commonly hold secrets.  Note that
    # "*.env.example" re-allows files that match "*.env.*", so it must
    # come last in the map (last-match-wins).
    "read": {
        "*": "allow",
        "*.env": "ask",
        "*.env.*": "ask",
        "*.env.example": "allow",  # example files are safe to read freely
    },
}

# Read-only subagent permission preset (e.g. the "explorer" subagent).
# Denies write/edit/bash/task by default while allowing all read-oriented
# and web permissions.  Agents that use this preset cannot modify the
# filesystem or run shell commands without an explicit override.
READONLY_PERMISSION: dict[str, Any] = {
    "*": "deny",
    "grep": "allow",
    "glob": "allow",
    "list": "allow",
    "read": "allow",
    "bash": "allow",
    "web": "allow",
}


# ==============================================================================
# Internal helpers
# ==============================================================================


def _expand_path(pattern: str) -> str:
    """Expand ``~`` and ``$HOME`` / ``${HOME}`` in path patterns.

    Only leading ``~`` (user home) and environment variable references are
    expanded.  The result always uses forward slashes so that patterns
    written for Linux also match on other platforms.
    """
    expanded = os.path.expandvars(os.path.expanduser(pattern))
    return expanded.replace("\\", "/")


def _validate_action(value: Any, *, context: str = "") -> PermissionAction:
    """Return *value* as a ``PermissionAction`` or raise ``ValueError``.

    Args:
        value: The action string to validate.
        context: Human-readable location hint included in the error message.

    Raises:
        ValueError: If *value* is not ``"allow"``, ``"deny"``, or ``"ask"``.
    """
    valid: frozenset[str] = frozenset({"allow", "deny", "ask"})
    if value not in valid:
        loc = f" (at {context})" if context else ""
        raise ValueError(
            f"Invalid permission action {value!r}{loc}. "
            f"Must be one of: {sorted(valid)}"
        )
    return value  # type: ignore[return-value]


# ==============================================================================
# Public API
# ==============================================================================


def from_config(permission: dict[str, Any]) -> Ruleset:
    """Convert a YAML permission config dict into a flat ``Ruleset``.

    Accepts two forms for each permission key:

    **Shorthand** — a single action string that applies to all patterns
    for that key::

        "bash": "allow"   # → Rule(permission="bash", pattern="*", action="allow")
        "*": "allow"      # → Rule(permission="*",    pattern="*", action="allow")

    **Pattern map** — a dict mapping concrete patterns to actions.
    Insertion order is preserved (Python 3.7+) and is significant because
    ``evaluate()`` uses last-match-wins::

        "read": {
            "*.env": "ask",
            "*":     "allow",
        }
        # → [Rule(permission="read", pattern="*.env", action="ask"),
        #    Rule(permission="read", pattern="*",     action="allow")]

    Path patterns in the pattern-map form are expanded: ``~`` is replaced
    with the user's home directory and ``$HOME`` / ``${HOME}`` are resolved
    via ``os.path.expandvars``.

    Unknown permission keys are accepted — this allows forward compatibility
    with permission keys added in later phases (e.g. ``"doom_loop"``).

    Args:
        permission: A dict mapping permission keys to actions or pattern maps.

    Returns:
        A flat ``Ruleset`` (``list[PermissionRule]``) in insertion order.

    Raises:
        ValueError: If any action string is not ``"allow"``, ``"deny"``,
            or ``"ask"``.
        TypeError: If a permission value is neither a string nor a dict.
    """
    rules: Ruleset = []

    for perm_key, value in permission.items():
        if isinstance(value, str):
            # Shorthand: the same action applies to every pattern ("*").
            action = _validate_action(value, context=f"permission key {perm_key!r}")
            rules.append(PermissionRule(permission=perm_key, pattern="*", action=action))

        elif isinstance(value, dict):
            # Pattern map: each entry maps one concrete pattern to one action.
            # We do not expand the permission key itself — only value patterns.
            for raw_pattern, action_str in value.items():
                expanded = _expand_path(str(raw_pattern))
                action = _validate_action(
                    action_str,
                    context=f"permission key {perm_key!r}, pattern {raw_pattern!r}",
                )
                rules.append(
                    PermissionRule(
                        permission=perm_key,
                        pattern=expanded,
                        action=action,
                    )
                )

        else:
            raise TypeError(
                f"Permission value for key {perm_key!r} must be a string or dict, "
                f"got {type(value).__name__!r}."
            )

    return rules


def merge(*rulesets: Ruleset) -> Ruleset:
    """Concatenate rulesets into a single flat list.

    Later rulesets take precedence over earlier ones via last-match-wins
    semantics in ``evaluate()``.  Rules in ``rulesets[1]`` can therefore
    override those in ``rulesets[0]``, and so on.

    An "always" reply from the user appends to ``PermissionService._approved``
    which is then merged here at evaluation time, allowing session-level
    overrides on top of agent config.

    Args:
        *rulesets: Zero or more ``Ruleset`` objects to merge.

    Returns:
        A single ``Ruleset`` containing all rules in order.
    """
    return [rule for rs in rulesets for rule in rs]


# ==============================================================================
# Migration helpers
# ==============================================================================

# Maps legacy Relay tool names to the new permission vocabulary.
# Tools not listed here fall back to their tool name as the permission key.
_TOOL_NAME_TO_PERMISSION: dict[str, str] = {
    "run_command": "bash",
    "read_file": "read",
    "write_file": "edit",
    "edit_file": "edit",
    "create_dir": "edit",
    "move_file": "edit",
    "delete_file": "edit",
    "glob_files": "glob",
    "grep_files": "grep",
    "ls": "list",
    "web_search": "web",
    "web_fetch": "web",
    "read_memory": "memory",
    "write_memory": "memory",
}

# Maps legacy arg keys to the pattern value for the new permission model.
# When a ToolApprovalRule specifies args, we use the first recognised arg
# key as the pattern string.  The canonical cases are run_command → command.
_TOOL_ARG_TO_PATTERN_KEY: dict[str, str] = {
    "run_command": "command",
    "read_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "glob_files": "pattern",
    "grep_files": "pattern",
}


def migrate_from_approval_config(approval_config: Any) -> Ruleset:
    """Convert a legacy ``ToolApprovalConfig`` into a ``Ruleset``.

    Translates:
    - ``always_allow`` rules → ``PermissionRule(action="allow")``
    - ``always_deny``  rules → ``PermissionRule(action="deny")``
    - ``always_ask``   rules → ``PermissionRule(action="ask")``

    For ``run_command`` rules with an ``args["command"]`` value, that
    command string becomes the pattern (matching Opencode's behaviour).
    For all other tools, the pattern defaults to ``"*"`` (all invocations).

    The legacy regex values stored in ``ToolApprovalRule.args`` are
    converted to plain wildcard patterns on a best-effort basis — complex
    regexes are preserved as-is, which may not match the new wildcard
    engine.  Manual review after migration is recommended.

    Args:
        approval_config: A ``ToolApprovalConfig`` instance from
            ``relay.configs.approval``.

    Returns:
        A flat ``Ruleset`` compatible with ``evaluate()`` and
        ``PermissionService``.
    """
    rules: Ruleset = []

    def _convert_rule(tool_rule: Any, action: PermissionAction) -> PermissionRule:
        """Convert one ``ToolApprovalRule`` to a ``PermissionRule``."""
        tool_name: str = tool_rule.name
        perm_key = _TOOL_NAME_TO_PERMISSION.get(tool_name, tool_name)

        pattern = "*"
        if tool_rule.args:
            # Extract the first recognised argument as the pattern.
            arg_key = _TOOL_ARG_TO_PATTERN_KEY.get(tool_name)
            if arg_key and arg_key in tool_rule.args:
                pattern = str(tool_rule.args[arg_key])

        return PermissionRule(permission=perm_key, pattern=pattern, action=action)

    for rule in approval_config.always_allow:
        rules.append(_convert_rule(rule, "allow"))

    for rule in approval_config.always_deny:
        rules.append(_convert_rule(rule, "deny"))

    for rule in approval_config.always_ask:
        rules.append(_convert_rule(rule, "ask"))

    return rules
