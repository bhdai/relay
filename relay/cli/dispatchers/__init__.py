"""Dispatchers for routing user inputs."""

from relay.cli.dispatchers.commands import CommandDispatcher
from relay.cli.dispatchers.messages import MessageDispatcher

__all__ = ["CommandDispatcher", "MessageDispatcher"]
