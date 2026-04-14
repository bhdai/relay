"""Middleware for rendering prompt templates from runtime context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware import dynamic_prompt

from relay.agents.context import AgentContext

if TYPE_CHECKING:
    from langchain.agents.middleware import ModelRequest


def render_prompt_template(template: str, context: AgentContext) -> str:
    """Render *template* with ``AgentContext.template_vars``.

    Missing or invalid format fields leave the template unchanged so a
    partial runtime context does not break graph construction.
    """
    try:
        return template.format(**context.template_vars)
    except (KeyError, ValueError):
        return template


def create_dynamic_prompt_middleware(template: str):
    """Return a dynamic prompt middleware for *template*."""

    @dynamic_prompt
    def render_prompt(request: ModelRequest) -> str:
        context = request.runtime.context
        if not isinstance(context, AgentContext):
            return template
        return render_prompt_template(template, context)

    return render_prompt