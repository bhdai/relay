# Load .env into os.environ before any LangChain/LangSmith imports so that
# tracing variables (LANGSMITH_API_KEY, LANGSMITH_TRACING, etc.) are visible
# to the LangSmith SDK, which reads os.environ directly.
from dotenv import load_dotenv

load_dotenv()

import asyncio
from uuid import uuid4

from langchain_core.messages import AIMessageChunk, HumanMessage

from relay.graph import build_graph


async def main():
    # No checkpointer yet — conversation state lives only in-memory for the
    # current process.  Pass `InMemorySaver()` to enable cross-turn memory.
    graph = build_graph()
    thread_id = str(uuid4())

    while True:
        try:
            user_input = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input in ("exit", "quit", "stop"):
            break
        if not user_input:
            continue

        config = {"configurable": {"thread_id": thread_id}}

        # `stream_mode="messages"` yields (chunk, metadata) tuples.  We only
        # care about AI content tokens for the REPL output.
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk[0], AIMessageChunk):
                print(chunk[0].content, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
