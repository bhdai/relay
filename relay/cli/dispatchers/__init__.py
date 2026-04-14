"""Dispatchers for routing user inputs."""

from relay.cli.dispatchers.commands import dispatch_command
from relay.cli.dispatchers.messages import MessageDispatcher

__all__ = ["MessageDispatcher", "dispatch_command"]
