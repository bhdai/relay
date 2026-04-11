from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import create_react_agent

from relay.settings import get_settings
from relay.tools.filesystem import FILE_SYSTEM_TOOLS
from relay.tools.terminal import TERMINAL_TOOLS
from relay.tools.web import WEB_TOOLS


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Construct the ReAct agent graph."""
    settings = get_settings()
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=settings.llm.openai_api_key,
    )
    return create_react_agent(
        llm,
        tools=[*FILE_SYSTEM_TOOLS, *TERMINAL_TOOLS, *WEB_TOOLS],
        checkpointer=checkpointer,
    )
