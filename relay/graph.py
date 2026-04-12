from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from relay.agents.context import AgentContext
from relay.agents.state import AgentState
from relay.settings import get_settings
from relay.tools.filesystem import FILE_SYSTEM_TOOLS
from relay.tools.memory import MEMORY_TOOLS
from relay.tools.terminal import TERMINAL_TOOLS
from relay.tools.todo import TODO_TOOLS
from relay.tools.web import WEB_TOOLS


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Construct the ReAct agent graph."""
    settings = get_settings()
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=settings.llm.openai_api_key,
    )
    return create_agent(
        llm,
        tools=[
            *FILE_SYSTEM_TOOLS,
            *TERMINAL_TOOLS,
            *WEB_TOOLS,
            *MEMORY_TOOLS,
            *TODO_TOOLS,
        ],
        state_schema=AgentState,
        context_schema=AgentContext,
        checkpointer=checkpointer,
    )
