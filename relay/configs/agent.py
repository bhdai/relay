"""Declarative agent and subagent configuration schemas.

These Pydantic models represent the YAML-serialisable definitions that
live under ``relay/resources/configs/default/`` (or a project-local
``.relay/`` directory).  The schemas mirror the structure of langrepl's
``configs/agent.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ==============================================================================
# Tool Configuration
# ==============================================================================


class ToolsConfig(BaseModel):
    """Declarative tool surface definition.

    ``patterns`` uses a colon-separated namespace syntax::

        impl:web:*          — all web tools
        impl:file_system:read_file  — single tool
        internal:*:*        — all internal tools

    Pattern matching is handled at build time by the tool factory
    (Phase 5).  For now the factory interprets patterns as hints for
    the hardcoded tool sets.
    """

    patterns: list[str] = Field(
        default_factory=list,
        description="Tool reference patterns (namespace:category:name)",
    )
    output_max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens per tool output before truncation",
    )


# ==============================================================================
# Base Agent Config
# ==============================================================================


class BaseAgentConfig(BaseModel):
    """Fields shared by top-level agents and subagents."""

    name: str = Field(description="Unique identifier for this agent definition")
    description: str = Field(
        default="",
        description="Human-readable description shown in delegation UI",
    )
    prompt: str | list[str] = Field(
        default="",
        description=(
            "System prompt — a single file path, list of file paths "
            "(concatenated with double newlines), or literal text"
        ),
    )
    llm: str = Field(
        default="default",
        description=(
            "LLM alias.  'default' resolves to the env-configured model. "
            "Named aliases will resolve via LLM config files in a future phase."
        ),
    )
    tools: ToolsConfig | None = Field(
        default=None,
        description="Tool surface configuration",
    )
    permission: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Permission ruleset for this agent's tool execution. "
            "Accepts shorthand (\"bash\": \"allow\") or pattern maps "
            "(\"read\": {\"*.env\": \"ask\", \"*\": \"allow\"}). "
            "Merged with DEFAULT_PERMISSION at build time; later entries win."
        ),
    )
    recursion_limit: int = Field(
        default=100,
        description="Maximum number of execution steps for this agent",
    )


# ==============================================================================
# Agent Config (top-level coordinator)
# ==============================================================================


class AgentConfig(BaseAgentConfig):
    """Configuration for a top-level (coordinator) agent.

    Adds fields that only make sense at the top level: default flag,
    subagent references, and checkpointer selection.
    """

    default: bool = Field(
        default=False,
        description="Whether this is the default agent when none is specified",
    )
    subagents: list[str] | None = Field(
        default=None,
        description=(
            "Names of subagent configs to attach.  Resolved by the "
            "registry into full SubAgentConfig objects."
        ),
    )
    checkpointer: str | None = Field(
        default=None,
        description="Checkpointer backend name (e.g. 'sqlite')",
    )


# ==============================================================================
# SubAgent Config
# ==============================================================================


class SubAgentConfig(BaseAgentConfig):
    """Configuration for a delegated subagent.

    Subagents do not have checkpointers or subagents of their own.
    """


# ==============================================================================
# Batch Containers
# ==============================================================================


class BatchAgentConfig(BaseModel):
    """Collection of top-level agent definitions."""

    agents: list[AgentConfig] = Field(
        default_factory=list,
        description="All loaded agent configs",
    )

    @property
    def agent_names(self) -> list[str]:
        return [a.name for a in self.agents]

    def get_agent(self, name: str | None = None) -> AgentConfig | None:
        """Return agent by *name*, or the default agent if *name* is ``None``."""
        if name is None:
            return self.get_default_agent()
        return next((a for a in self.agents if a.name == name), None)

    def get_default_agent(self) -> AgentConfig | None:
        if not self.agents:
            return None
        default = next((a for a in self.agents if a.default), None)
        return default or self.agents[0]


class BatchSubAgentConfig(BaseModel):
    """Collection of subagent definitions."""

    subagents: list[SubAgentConfig] = Field(
        default_factory=list,
        description="All loaded subagent configs",
    )

    @property
    def subagent_names(self) -> list[str]:
        return [s.name for s in self.subagents]

    def get_subagent(self, name: str) -> SubAgentConfig | None:
        return next((s for s in self.subagents if s.name == name), None)
