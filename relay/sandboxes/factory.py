"""Factory for creating sandbox backends and resolving tool bindings.

``SandboxFactory`` takes the declarative ``AgentSandboxConfig`` (from
YAML or Python) and produces a list of ``SandboxBinding`` objects that
the ``SandboxMiddleware`` uses to route tool calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from relay.configs.sandbox import SandboxType
from relay.sandboxes.backend import SandboxBackend, SandboxBinding
from relay.sandboxes.bubblewrap import BubblewrapBackend

if TYPE_CHECKING:
    from relay.configs.sandbox import AgentSandboxConfig, SandboxConfig

logger = logging.getLogger(__name__)

BACKEND_TYPES: dict[SandboxType, type[SandboxBackend]] = {
    SandboxType.BUBBLEWRAP: BubblewrapBackend,
}


class SandboxFactory:
    """Create sandbox backend instances and resolve tool bindings."""

    def __init__(self) -> None:
        self._backends: dict[str, SandboxBackend] = {}

    def create_backend(
        self,
        config: SandboxConfig,
        working_dir: Path,
    ) -> SandboxBackend:
        """Create (or reuse) a backend for *config* + *working_dir*."""
        cache_key = f"{config.name}:{working_dir}"
        if cache_key in self._backends:
            return self._backends[cache_key]

        config.validate_current_os()

        backend_cls = BACKEND_TYPES.get(config.type)
        if backend_cls is None:
            raise ValueError(f"Unknown sandbox type: {config.type}")

        backend = backend_cls(config, working_dir)
        backend.validate_environment()
        self._backends[cache_key] = backend
        return backend

    def build_bindings(
        self,
        agent_config: AgentSandboxConfig,
        working_dir: Path,
    ) -> list[SandboxBinding]:
        """Build sandbox bindings from agent sandbox config profiles.

        Returns an empty list when sandboxing is disabled.
        """
        if not agent_config or not agent_config.enabled:
            return []

        bindings: list[SandboxBinding] = []
        for profile in agent_config.profiles:
            if profile.sandbox is None:
                backend = None
            else:
                backend = self.create_backend(profile.sandbox, working_dir)

            bindings.append(
                SandboxBinding(patterns=profile.patterns, backend=backend)
            )

        return bindings
