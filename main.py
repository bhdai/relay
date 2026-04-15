import asyncio

from relay.cli.bootstrap.app import cli as bootstrap_cli
from relay.cli.bootstrap.app import main as bootstrap_main


async def main(argv: list[str] | None = None) -> int:
    return await bootstrap_main(argv=argv)


def cli(argv: list[str] | None = None) -> int:
    return bootstrap_cli(argv=argv)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
