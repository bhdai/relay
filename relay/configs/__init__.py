"""Declarative agent and runtime configuration.

Schemas, a config registry, and utility functions for loading YAML-based
agent and subagent definitions.  See ``relay/resources/configs/default/``
for the packaged defaults.
"""

from relay.configs.agent import (
    AgentConfig,
    BaseAgentConfig,
    BatchAgentConfig,
    BatchSubAgentConfig,
    SubAgentConfig,
    ToolsConfig,
)
from relay.configs.approval import ApprovalMode, ToolApprovalConfig, ToolApprovalRule
from relay.configs.registry import ConfigRegistry
from relay.configs.utils import load_prompt_content

__all__ = [
    "AgentConfig",
    "ApprovalMode",
    "BaseAgentConfig",
    "BatchAgentConfig",
    "BatchSubAgentConfig",
    "ConfigRegistry",
    "SubAgentConfig",
    "ToolApprovalConfig",
    "ToolApprovalRule",
    "ToolsConfig",
    "load_prompt_content",
]
