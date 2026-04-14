"""Planning tools for the coordinator agent.

The coordinator's ``think`` tool does NOT have ``return_direct=True``
so it can reflect and continue working.  Subagents use the variant in
``relay.tools.subagents.task`` which has ``return_direct=True`` — there
it acts as a deliberate exit ramp that returns the reflection to the
coordinator.
"""

from langchain_core.tools import tool


@tool
def think(reflection: str) -> str:
    """Tool for strategic reflection on progress and decision-making.

    Use this tool to pause and reason about the current state of the
    task before deciding next steps.

    When to use:

    - After receiving results: What key information did I find?
    - Before deciding next steps: Do I have enough to proceed?
    - When assessing gaps: What specific information is still missing?
    - Before delegating: Which subagent is best suited for this?
    """
    return f"Reflection recorded: {reflection}"


think.metadata = {"approval_config": {"always_approve": True}}

PLANNING_TOOLS = [think]