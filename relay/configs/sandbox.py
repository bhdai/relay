"""Sandbox configuration classes.

Defines filesystem, network, and profile-binding models that control
how each tool is sandboxed.  Configuration is loaded from YAML sandbox
profile files and then referenced by agent configs.
"""

from __future__ import annotations

import sys
from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ==============================================================================
# Enumerations
# ==============================================================================


class SandboxType(str, Enum):
    """Sandbox backend type."""

    BUBBLEWRAP = "bubblewrap"


class SandboxOS(str, Enum):
    """Target operating system for sandbox."""

    LINUX = "linux"


# ==============================================================================
# Filesystem & Network Rules
# ==============================================================================


class FilesystemConfig(BaseModel):
    """Filesystem access rules for sandbox.

    Use ``"."`` in *read* or *write* to refer to the agent's
    ``working_dir``.  All other entries are absolute paths or glob
    patterns expanded at runtime.
    """

    read: list[str] = Field(
        default_factory=list,
        description="Paths allowed for reading ('.' = working_dir)",
    )
    write: list[str] = Field(
        default_factory=list,
        description="Paths allowed for writing ('.' = working_dir)",
    )
    hidden: list[str] = Field(
        default_factory=list,
        description="Paths/patterns hidden from sandbox (supports glob: ~/.ssh, *.pem)",
    )


class NetworkConfig(BaseModel):
    """Network access rules for sandbox.

    Network filtering is currently binary: if *remote* is empty, all
    outbound TCP is blocked (``--unshare-net``).  If *remote* contains
    any value (including ``"*"``), the network namespace is shared.
    """

    remote: list[str] = Field(
        default_factory=list,
        description="Allowed remote hosts ('*' = allow all, empty = deny all)",
    )
    local: list[str] = Field(
        default_factory=list,
        description="Allowed local unix sockets",
    )


# ==============================================================================
# Profile Config (one YAML file = one SandboxConfig)
# ==============================================================================


class SandboxConfig(BaseModel):
    """Configuration for a sandbox profile.

    Example YAML::

        name: default-linux
        type: bubblewrap
        os: linux
        filesystem:
          read: [".", "/usr", "/lib", "/lib64", "/etc/resolv.conf"]
          write: ["."]
          hidden: ["~/.ssh", "~/.aws", ".env"]
        network:
          remote: ["*"]
    """

    name: str = Field(description="Unique sandbox profile name")
    type: SandboxType = Field(description="Sandbox backend type")
    os: SandboxOS = Field(description="Target operating system")
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)

    @model_validator(mode="after")
    def validate_os_compatibility(self) -> SandboxConfig:
        """Validate that sandbox type matches OS."""
        if self.type == SandboxType.BUBBLEWRAP and self.os != SandboxOS.LINUX:
            raise ValueError("bubblewrap sandbox type requires os: linux")
        return self

    def validate_current_os(self) -> None:
        """Raise ``RuntimeError`` if the host OS does not match *self.os*."""
        current_os = SandboxOS.LINUX if sys.platform == "linux" else None
        if current_os != self.os:
            raise RuntimeError(
                f"Sandbox '{self.name}' requires {self.os.value}, "
                f"but running on {sys.platform}"
            )


# ==============================================================================
# Profile Binding (maps tool patterns → sandbox profile)
# ==============================================================================


class SandboxProfileBinding(BaseModel):
    """Binds tool patterns to a sandbox profile.

    ``sandbox: null`` means the matched tools execute without sandbox.
    """

    sandbox: SandboxConfig | None = Field(
        default=None,
        description="Sandbox config (None = unsandboxed passthrough)",
    )
    patterns: list[str] = Field(
        description="Tool patterns to match (same 3-part format as agent tools config)",
    )


# ==============================================================================
# Agent-level Sandbox Section
# ==============================================================================


class AgentSandboxConfig(BaseModel):
    """Agent-level sandbox configuration (embedded inside agent YAML).

    Example::

        sandbox:
          enabled: true
          profiles:
            - patterns: ["impl:terminal:*", "impl:file_system:*"]
              sandbox:
                name: default-linux
                type: bubblewrap
                os: linux
                filesystem:
                  read: [".", "/usr", "/lib", "/lib64"]
                  write: ["."]
                  hidden: ["~/.ssh", "~/.aws", ".env"]
                network:
                  remote: ["*"]
            - patterns: ["internal:*:*"]
              sandbox: null
    """

    enabled: bool = Field(default=False, description="Enable sandboxing for this agent")
    profiles: list[SandboxProfileBinding] = Field(
        default_factory=list,
        description="Ordered profile bindings (first match wins)",
    )
