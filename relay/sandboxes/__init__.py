"""Sandbox execution abstraction.

Defines the ``SandboxBackend`` protocol that OS-specific backends
(bubblewrap, etc.) must implement, the ``SandboxFactory`` for building
backends from config, and the ``SandboxBinding`` that wires tool
patterns to backends.
"""

from relay.sandboxes.backend import SandboxBackend, SandboxBinding
from relay.sandboxes.bubblewrap import BubblewrapBackend
from relay.sandboxes.factory import SandboxFactory

__all__ = [
    "BubblewrapBackend",
    "SandboxBackend",
    "SandboxBinding",
    "SandboxFactory",
]
