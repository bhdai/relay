"""Tests for Phase 4 subagent permission inheritance in ``create_task_tool``.

These cases cover the default and custom ``SubAgentRuntime.permission_ruleset``
field, coordinator and subagent ruleset merging, read-only explorer-style
overrides, worker inheritance, and forwarding from ``create_deep_agent``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from relay.agents.context import AgentContext
from relay.agents.deep_agent import create_deep_agent
from relay.agents.state import AgentState
from relay.permission.config import from_config, merge
from relay.permission.evaluate import evaluate
from relay.permission.schema import PermissionRule, Ruleset
from relay.tools.subagents.task import SubAgentRuntime, create_task_tool


# ==============================================================================
# Helpers
# ==============================================================================


def _rule(permission: str, pattern: str, action: str) -> PermissionRule:
    return PermissionRule(permission=permission, pattern=pattern, action=action)


def _allow(permission: str, pattern: str = "*") -> PermissionRule:
    return _rule(permission, pattern, "allow")


def _deny(permission: str, pattern: str = "*") -> PermissionRule:
    return _rule(permission, pattern, "deny")


@tool
def _noop() -> str:
    """A no-op tool for testing."""
    return "ok"


def _make_mock_runtime() -> MagicMock:
    """Build a minimal mock ToolRuntime for triggering lazy subagent compilation."""
    runtime = MagicMock(spec=ToolRuntime)
    runtime.state = {"messages": []}
    runtime.context = AgentContext()
    runtime.tool_call_id = "call-1"
    runtime.stream_writer = MagicMock()
    runtime.config = {"configurable": {"thread_id": "t-test"}}
    return runtime


def _make_subagent_config(
    name: str = "worker",
    permission_ruleset: Ruleset | None = None,
) -> SubAgentRuntime:
    return SubAgentRuntime(
        name=name,
        description=f"{name} description.",
        tools=[_noop],
        prompt=f"{name} prompt.",
        permission_ruleset=permission_ruleset or [],
    )


# ==============================================================================
# SubAgentRuntime.permission_ruleset field
# ==============================================================================


class TestSubAgentRuntimePermissionRuleset:
    """SubAgentRuntime has a permission_ruleset field that defaults to []."""

    def test_permission_ruleset_defaults_to_empty(self):
        runtime = SubAgentRuntime(
            name="explorer",
            description="Explorer.",
            tools=[_noop],
            prompt="Explore.",
        )
        assert runtime.permission_ruleset == []

    def test_permission_ruleset_can_be_set(self):
        ruleset: Ruleset = [_allow("read"), _deny("bash")]
        runtime = SubAgentRuntime(
            name="readonly",
            description="Read-only.",
            tools=[_noop],
            prompt="Read only.",
            permission_ruleset=ruleset,
        )
        assert len(runtime.permission_ruleset) == 2
        assert runtime.permission_ruleset[0].permission == "read"
        assert runtime.permission_ruleset[0].action == "allow"
        assert runtime.permission_ruleset[1].permission == "bash"
        assert runtime.permission_ruleset[1].action == "deny"


# ==============================================================================
# Ruleset merge logic (verify semantics without running a full graph)
# ==============================================================================


class TestRulesetMergeSemantics:
    """Verify last-match-wins merge correctly composes coordinator + subagent rules."""

    def test_coordinator_allow_bash_subagent_deny_bash(self):
        """Subagent deny rule overrides coordinator allow for bash."""
        coordinator: Ruleset = [_allow("bash")]
        subagent: Ruleset = [_deny("bash")]

        child = merge(coordinator, subagent)

        assert evaluate("bash", "*", child).action == "deny"

    def test_worker_empty_override_inherits_coordinator(self):
        """Worker subagent with no overrides fully inherits coordinator allow."""
        coordinator: Ruleset = [_allow("bash"), _allow("edit")]
        subagent: Ruleset = []

        child = merge(coordinator, subagent)

        assert evaluate("bash", "*", child).action == "allow"
        assert evaluate("edit", "*", child).action == "allow"

    def test_explorer_style_deny_all_allow_read(self):
        """Explorer subagent denies everything except read (deny * + allow read)."""
        coordinator: Ruleset = [_allow("*")]  # coordinator allows all
        # Explorer overrides: deny * first, then re-allow specific read permissions.
        # The last rules win, so allow("read") overrides deny("*") for read calls.
        subagent: Ruleset = [_deny("*"), _allow("read"), _allow("glob")]

        child = merge(coordinator, subagent)

        # bash is denied because deny("*") comes after coordinator allow("*")
        # and no later rule re-allows it.
        assert evaluate("bash", "*", child).action == "deny"
        # read and glob are allowed because their allow rules come after deny.
        assert evaluate("read", "*", child).action == "allow"
        assert evaluate("glob", "*", child).action == "allow"

    def test_coordinator_specific_deny_survives_empty_subagent_override(self):
        """Coordinator deny rules are preserved when the subagent adds nothing."""
        coordinator: Ruleset = [
            _allow("*"),
            _deny("bash", "rm -rf *"),
        ]
        subagent: Ruleset = []

        child = merge(coordinator, subagent)

        assert evaluate("bash", "rm -rf /", child).action == "deny"
        assert evaluate("bash", "git status", child).action == "allow"


# ==============================================================================
# create_task_tool passes coordinator_ruleset into create_react_agent
# ==============================================================================


class TestCreateTaskToolPassesRuleset:
    """Verify the child ruleset is forwarded into ``create_react_agent``."""

    def _run_task_tool_with_captured_ruleset(
        self,
        subagent_name: str,
        subagent_permission: Ruleset,
        coordinator_ruleset: Ruleset | None,
    ) -> Ruleset | None:
        """Build a task tool and capture the forwarded permission ruleset.

        Args:
            subagent_name: Name of the delegated subagent.
            subagent_permission: Ruleset applied directly to the subagent.
            coordinator_ruleset: Inherited coordinator ruleset, if any.

        Returns:
            The ruleset that ``create_react_agent`` received for the child
            graph.
        """
        config = _make_subagent_config(subagent_name, subagent_permission)
        captured: list[Ruleset | None] = []

        # Async generator that yields nothing — satisfies `async for` in task().
        async def _empty_astream(*args, **kwargs):
            # Annotate as async generator by using yield in a false branch.
            if False:  # pragma: no cover
                yield

        fake_subagent = MagicMock()
        fake_subagent.astream = _empty_astream

        def _fake_create_react_agent(*args, **kwargs):
            captured.append(kwargs.get("permission_ruleset"))
            return fake_subagent

        with patch(
            "relay.tools.subagents.task.create_react_agent",
            side_effect=_fake_create_react_agent,
        ):
            task_fn = create_task_tool(
                subagent_configs=[config],
                model_provider=lambda _cfg: MagicMock(),
                coordinator_ruleset=coordinator_ruleset,
            )

            runtime = _make_mock_runtime()

            # The task tool is async; invoke via its coroutine attribute.
            # ToolRuntime is passed directly since we're bypassing the
            # LangChain framework's injection mechanism.
            async def _run():
                try:
                    await task_fn.coroutine(
                        description="do something",
                        subagent_type=subagent_name,
                        runtime=runtime,
                    )
                except Exception:
                    pass

            asyncio.run(_run())

        assert len(captured) == 1, "create_react_agent called exactly once (lazy)"
        return captured[0]

    def test_subagent_deny_overrides_coordinator_allow(self):
        """Subagent deny(bash) wins over coordinator allow(bash) in the child ruleset."""
        coordinator: Ruleset = [_allow("bash")]
        subagent: Ruleset = [_deny("bash")]

        child = self._run_task_tool_with_captured_ruleset(
            "restricted", subagent, coordinator
        )

        assert child is not None
        assert evaluate("bash", "*", child).action == "deny"

    def test_worker_inherits_coordinator_with_no_overrides(self):
        """Worker subagent (empty override) inherits coordinator allow rules."""
        coordinator: Ruleset = [_allow("bash"), _allow("edit")]
        subagent: Ruleset = []

        child = self._run_task_tool_with_captured_ruleset("worker", subagent, coordinator)

        assert child is not None
        assert evaluate("bash", "*", child).action == "allow"
        assert evaluate("edit", "*", child).action == "allow"

    def test_no_coordinator_ruleset_uses_subagent_only(self):
        """When coordinator_ruleset is None, subagent only has its own ruleset."""
        subagent: Ruleset = [_allow("read")]

        child = self._run_task_tool_with_captured_ruleset("slim", subagent, None)

        assert child is not None
        # No coordinator or subagent rule for bash → default ask.
        assert evaluate("bash", "*", child).action == "ask"
        assert evaluate("read", "*", child).action == "allow"

    def test_empty_coordinator_and_subagent_yields_default_ask(self):
        """No rules → default ask for any permission."""
        child = self._run_task_tool_with_captured_ruleset("empty", [], [])

        assert child is not None
        assert evaluate("bash", "*", child).action == "ask"


# ==============================================================================
# create_deep_agent forwards permission_ruleset to create_task_tool
# ==============================================================================


class TestDeepAgentPermissionForwarding:
    """create_deep_agent passes its permission_ruleset into create_task_tool."""

    def test_coordinator_ruleset_forwarded_to_task_tool(self):
        """create_deep_agent forwards permission_ruleset to create_task_tool."""
        coordinator_ruleset: Ruleset = [_allow("bash")]
        subagent_config = _make_subagent_config("worker")

        captured: list[Ruleset | None] = []
        original_create_task_tool = create_task_tool

        def _capturing_create_task_tool(*args, **kwargs):
            captured.append(kwargs.get("coordinator_ruleset"))
            return original_create_task_tool(*args, **kwargs)

        with patch(
            "relay.agents.deep_agent.create_task_tool",
            side_effect=_capturing_create_task_tool,
        ):
            create_deep_agent(
                model=MagicMock(),
                tools=[_noop],
                prompt="Coordinator.",
                subagent_configs=[subagent_config],
                state_schema=AgentState,
                context_schema=AgentContext,
                name="coordinator",
                permission_ruleset=coordinator_ruleset,
            )

        assert len(captured) == 1
        assert captured[0] == coordinator_ruleset

    def test_none_permission_ruleset_forwards_empty_list(self):
        """When permission_ruleset=None, create_deep_agent forwards []."""
        subagent_config = _make_subagent_config("worker")

        captured: list[Ruleset | None] = []
        original_create_task_tool = create_task_tool

        def _capturing_create_task_tool(*args, **kwargs):
            captured.append(kwargs.get("coordinator_ruleset"))
            return original_create_task_tool(*args, **kwargs)

        with patch(
            "relay.agents.deep_agent.create_task_tool",
            side_effect=_capturing_create_task_tool,
        ):
            create_deep_agent(
                model=MagicMock(),
                tools=[_noop],
                prompt="Coordinator.",
                subagent_configs=[subagent_config],
                state_schema=AgentState,
                context_schema=AgentContext,
                name="coordinator",
                permission_ruleset=None,
            )

        assert len(captured) == 1
        assert captured[0] == [], "None permission_ruleset should forward as []"



# ==============================================================================
# Helpers
