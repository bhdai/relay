"""Sandbox execution abstraction.

Defines the ``SandboxBackend`` protocol that OS-specific backends
(bubblewrap, seatbelt, etc.) must implement, and the
``SandboxMiddleware`` that wires sandboxing into the agent middleware
stack.
"""

from relay.sandboxes.backend import SandboxBackend

__all__ = [
    "SandboxBackend",
]
