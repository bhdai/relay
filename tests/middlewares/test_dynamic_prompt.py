"""Tests for dynamic prompt rendering middleware helpers."""

from relay.agents.context import AgentContext
from relay.middlewares.dynamic_prompt import render_prompt_template


def test_render_prompt_template_uses_context_variables():
    context = AgentContext(
        working_dir="/workspace",
        platform="linux",
        shell="/bin/fish",
        current_date_time_zoned="2026-04-13T12:34:56+00:00",
        user_memory="remember this",
    )

    rendered = render_prompt_template(
        "cwd={working_dir} platform={platform} shell={shell} memory={user_memory}",
        context,
    )

    assert rendered == (
        "cwd=/workspace platform=linux shell=/bin/fish memory=remember this"
    )


def test_render_prompt_template_leaves_unknown_fields_unchanged():
    context = AgentContext(working_dir="/workspace")

    rendered = render_prompt_template(
        "cwd={working_dir} missing={missing_value}",
        context,
    )

    assert rendered == "cwd={working_dir} missing={missing_value}"