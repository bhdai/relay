"""Command-line entry point for the relay REPL."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Sequence

from relay.settings import load_environment

load_environment()

from relay.cli.core.session import Session


def create_parser() -> argparse.ArgumentParser:
    """Create the relay command-line parser."""
    parser = argparse.ArgumentParser(
        prog="relay",
        description="Interactive Relay coding assistant",
    )
    parser.add_argument(
        "-w",
        "--working-dir",
        type=str,
        default=os.getcwd(),
        help="Working directory for configs, tools, and memory (default: current directory)",
    )
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        default=None,
        help="Top-level agent profile from .relay/agents or packaged defaults",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model override; takes precedence over the agent YAML and .env default",
    )
    return parser


async def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI args, start a session, and return a process exit code."""
    parser = create_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    session = Session(
        working_dir=Path(args.working_dir).resolve(),
        agent_name=args.agent,
        model_name=args.model,
    )
    await session.start()
    return 0


def cli(argv: Sequence[str] | None = None) -> int:
    """Run the CLI synchronously for console scripts."""
    try:
        return asyncio.run(main(argv))
    except KeyboardInterrupt:
        return 130