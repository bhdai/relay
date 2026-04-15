"""Module entry point for ``python -m relay``."""

from relay.cli.bootstrap.app import cli


if __name__ == "__main__":
    raise SystemExit(cli())