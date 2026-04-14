"""Skill catalog — dynamic skill discovery.

These tools let the LLM discover and read skills from the
``AgentContext.skill_catalog`` at runtime.
"""

from __future__ import annotations

import json
import re

from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool

from relay.agents.context import AgentContext


@tool
async def fetch_skills(
    runtime: ToolRuntime[AgentContext], pattern: str | None = None
) -> str:
    """Discover and search for available skills in the catalog.

    WITHOUT pattern: Returns ALL available skills.
    WITH pattern: Returns ONLY matching skills (case-insensitive regex on names, categories, and descriptions).

    Args:
        pattern: Optional regex pattern to filter skills.

    Returns:
        JSON array of skill objects with: category, name, description.
    """
    skills = runtime.context.skill_catalog

    if not skills:
        return json.dumps([])

    if pattern is None:
        result = [
            {
                "category": skill.category,
                "name": skill.name,
                "description": skill.description,
            }
            for skill in skills
        ]
        return json.dumps(result, indent=2)

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ToolException(f"Invalid regex pattern: {e}") from e

    matches = []
    for skill in skills:
        if (
            regex.search(skill.name)
            or regex.search(skill.category)
            or regex.search(skill.description)
        ):
            matches.append(
                {
                    "category": skill.category,
                    "name": skill.name,
                    "description": skill.description,
                }
            )

    return json.dumps(matches, indent=2)


fetch_skills.metadata = {"approval_config": {"always_approve": True}}


@tool
async def get_skill(
    category: str, name: str, runtime: ToolRuntime[AgentContext]
) -> str:
    """Read the full content of a specific skill.

    Args:
        category: Category of the skill (from fetch_skills output).
        name: Name of the skill (from fetch_skills output).

    Returns:
        Complete SKILL.md content.
    """
    skills = runtime.context.skill_catalog
    skill = next((s for s in skills if s.category == category and s.name == name), None)

    if not skill:
        raise ToolException(f"Skill '{category}/{name}' not found")

    content = skill.read_content()
    if not content:
        raise ToolException(f"Failed to read skill '{category}/{name}'")

    return content


get_skill.metadata = {"approval_config": {"always_approve": True}}


SKILL_CATALOG_TOOLS = [fetch_skills, get_skill]
