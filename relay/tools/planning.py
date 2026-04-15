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

    Always use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing gaps: What specific information am I still missing?
    - Before concluding: Can I provide a complete answer now?
    - How complex is the question: Have I reached the number of search limits?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


think.metadata = {"approval_config": {"always_approve": True}}

PLANNING_TOOLS = [think]

