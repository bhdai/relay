# Load .env into os.environ before any LangChain/LangSmith imports so that
# tracing variables (LANGSMITH_API_KEY, LANGSMITH_TRACING, etc.) are visible
# to the LangSmith SDK, which reads os.environ directly.
from dotenv import load_dotenv

load_dotenv()

import asyncio

from relay.cli.session import Session


async def main():
    session = Session()
    await session.start()


if __name__ == "__main__":
    asyncio.run(main())
