"""Tests for PermissionMiddleware and related helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from relay.agents.context import AgentContext
from relay.middlewares.permission import (
    PermissionInterruptPayload,
    PermissionMiddleware,
)
from relay.permission.config import DEFAULT_PERMISSION, from_config
from relay.permission.schema import PermissionRule, Ruleset


# ==============================================================================
# Test helpers
# ==============================================================================


def _make_ruleset(*rules: PermissionRule) -> Ruleset:
    """Build a ruleset from individual rules."""
    return list(rules)


def _allow(permission: str = "*", pattern: str = "*") -> PermissionRule:
    return PermissionRule(permission=permission, pattern=pattern, action="allow")


def _deny(permission: str = "*", pattern: str = "*") -> PermissionRule:
    return PermissionRule(permission=permission, pattern=pattern, action="deny")


def _ask(permission: str = "*", pattern: str = "*") -> PermissionRule:
    return PermissionRule(permission=permission, pattern=pattern, action="ask")


def _make_request(
    tool_name: str = "test_tool",
    call_id: str = "call_1",
    args: dict | None = None,
    *,
    thread_id: str = "thread-1",
    tool_metadata: dict | None = None,
    tool_catalog: list | None = None,
    permission_ruleset: list[dict] | None = None,
) -> Mock:
    """Build a mock ToolCallRequest for PermissionMiddleware tests.

    The mock simulates the LangGraph runtime injected into middleware by
    providing a ``runtime`` with ``config`` (for thread_id), ``context``
    (AgentContext), and a ``tool`` with the specified metadata.
    """
    mock_tool = Mock()
    mock_tool.metadata = tool_metadata or {}

    request = Mock(spec=ToolCallRequest)
    request.tool_call = {
        "id": call_id,
        "name": tool_name,
        "args": args or {},
    }
    request.tool = mock_tool
    request.runtime = Mock()
    request.runtime.config = {"configurable": {"thread_id": thread_id}}
    request.runtime.context = AgentContext(
        tool_catalog=tool_catalog or [],
        permission_ruleset=permission_ruleset or [],
    )
    return request


def _make_handler(content: str = "ok", tool_name: str = "test_tool", call_id: str = "call_1"):
    """Return an async handler that yields a successful ToolMessage."""
    return AsyncMock(
        return_value=ToolMessage(
            name=tool_name,
            content=content,
            tool_call_id=call_id,
        )
    )


# ==============================================================================
# Allow rule → tool executes
# ==============================================================================


class TestAllowDecision:
    """PermissionMiddleware with an allow rule executes the tool immediately."""

    @pytest.mark.asyncio
    async def test_allow_rule_executes_tool(self):
        ruleset = _make_ruleset(_allow("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request()
        handler = _make_handler()

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "ok"
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_permission_allows_tool(self):
        """DEFAULT_PERMISSION has '*': 'allow'; all tools pass without interruption."""
        ruleset = from_config(DEFAULT_PERMISSION)
        middleware = PermissionMiddleware(ruleset)
        request = _make_request()
        handler = _make_handler()

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_specific_permission_key_allow(self):
        """Rule matching the tool's specific permission key allows execution."""
        ruleset = _make_ruleset(_allow("bash", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request(
            tool_metadata={
                "permission_config": {
                    "permission": "bash",
                    "patterns_fn": lambda args: ["git status"],
                }
            }
        )
        handler = _make_handler()

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        handler.assert_called_once()


# ==============================================================================
# Deny rule → return_direct error message
# ==============================================================================


class TestDenyDecision:
    """PermissionMiddleware with a deny rule returns an error without executing."""

    @pytest.mark.asyncio
    async def test_deny_rule_blocks_tool(self):
        # Deny all: '*' permission, '*' pattern.
        ruleset = _make_ruleset(_deny("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request()
        handler = _make_handler()

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_deny_rule_sets_return_direct(self):
        """Denied tool message must carry return_direct so the graph stops."""
        ruleset = _make_ruleset(_deny("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request()
        handler = _make_handler()

        result = await middleware.awrap_tool_call(request, handler)

        # return_direct is a first-class attribute on ToolMessage.
        assert getattr(result, "return_direct", False) is True

    @pytest.mark.asyncio
    async def test_deny_specific_pattern_blocks_matching_command(self):
        """Deny rule for 'bash:rm *' blocks 'rm -rf /' but not 'git status'."""
        ruleset = _make_ruleset(
            _allow("bash", "*"),        # base: allow all bash
            _deny("bash", "rm *"),      # override: deny rm commands
        )
        middleware = PermissionMiddleware(ruleset)

        # rm command → should be denied.
        request_rm = _make_request(
            call_id="call_rm",
            tool_metadata={
                "permission_config": {
                    "permission": "bash",
                    "patterns_fn": lambda args: [args.get("command", "")],
                }
            },
            args={"command": "rm -rf /tmp"},
        )
        handler_rm = _make_handler(call_id="call_rm")

        result_rm = await middleware.awrap_tool_call(request_rm, handler_rm)
        assert result_rm.status == "error"
        handler_rm.assert_not_called()

        # git status → should be allowed (different session so fresh service).
        middleware2 = PermissionMiddleware(ruleset)
        request_git = _make_request(
            call_id="call_git",
            tool_metadata={
                "permission_config": {
                    "permission": "bash",
                    "patterns_fn": lambda args: [args.get("command", "")],
                }
            },
            args={"command": "git status"},
        )
        handler_git = _make_handler(call_id="call_git")

        result_git = await middleware2.awrap_tool_call(request_git, handler_git)
        assert isinstance(result_git, ToolMessage)
        assert result_git.status != "error"
        handler_git.assert_called_once()


# ==============================================================================
# NeedsAsk → interrupt → reply handling
# ==============================================================================


class TestNeedsAskDecision:
    """PermissionMiddleware with an ask rule triggers an interrupt."""

    @pytest.mark.asyncio
    async def test_ask_rule_fires_interrupt(self):
        """When evaluation yields NeedsAsk, ``interrupt`` should be called."""
        ruleset = _make_ruleset(_ask("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request()
        handler = _make_handler()

        with patch(
            "relay.middlewares.permission.interrupt",
            return_value="once",
        ) as interrupt_mock:
            result = await middleware.awrap_tool_call(request, handler)

        interrupt_mock.assert_called_once()
        assert isinstance(interrupt_mock.call_args[0][0], PermissionInterruptPayload)

    @pytest.mark.asyncio
    async def test_once_reply_executes_tool(self):
        """Replying 'once' executes the tool without persisting any rules."""
        ruleset = _make_ruleset(_ask("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request()
        handler = _make_handler()

        with patch("relay.middlewares.permission.interrupt", return_value="once"):
            result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "ok"
        handler.assert_called_once()
        # "once" must not persist any rules to context.
        assert request.runtime.context.permission_ruleset == []

    @pytest.mark.asyncio
    async def test_always_reply_executes_tool_and_persists_rules(self):
        """Replying 'always' executes the tool and persists approved rules to context."""
        ruleset = _make_ruleset(_ask("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request(
            tool_metadata={
                "permission_config": {
                    "permission": "bash",
                    "patterns_fn": lambda args: ["git status"],
                    "always_fn": lambda args: ["git *"],
                }
            }
        )
        handler = _make_handler()

        with patch("relay.middlewares.permission.interrupt", return_value="always"):
            result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        handler.assert_called_once()

        # "always" must persist the 'always_fn' patterns to context.
        persisted = request.runtime.context.permission_ruleset
        assert len(persisted) >= 1
        assert any(r["permission"] == "bash" and "git" in r["pattern"] for r in persisted)

    @pytest.mark.asyncio
    async def test_reject_reply_returns_error_without_executing(self):
        """Replying 'reject' returns an error ToolMessage without running the tool."""
        ruleset = _make_ruleset(_ask("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request()
        handler = _make_handler()

        with patch("relay.middlewares.permission.interrupt", return_value="reject"):
            result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "rejected" in result.content.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_interrupt_payload_contains_correct_fields(self):
        """The interrupt payload carries the expected permission, patterns, and options."""
        ruleset = _make_ruleset(_ask("bash", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request(
            tool_name="run_command",
            args={"command": "git push --force"},
            tool_metadata={
                "permission_config": {
                    "permission": "bash",
                    "patterns_fn": lambda args: [args.get("command", "")],
                    "always_fn": lambda args: ["git push *"],
                    "metadata_fn": lambda args: {"command": args.get("command", "")},
                }
            },
        )
        handler = _make_handler()

        captured_payload: list[PermissionInterruptPayload] = []

        def _capture(payload):
            captured_payload.append(payload)
            return "once"

        with patch("relay.middlewares.permission.interrupt", side_effect=_capture):
            await middleware.awrap_tool_call(request, handler)

        assert len(captured_payload) == 1
        payload = captured_payload[0]
        assert payload.permission == "bash"
        assert "git push --force" in payload.patterns
        assert "git push *" in payload.always_patterns
        assert payload.metadata == {"command": "git push --force"}
        assert set(payload.options) == {"once", "always", "reject"}


# ==============================================================================
# Session isolation
# ==============================================================================


class TestSessionIsolation:
    """PermissionService instances are isolated by thread_id."""

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(self):
        """An 'always' reply in session A must not affect session B."""
        ruleset = _make_ruleset(_ask("*", "*"))
        middleware = PermissionMiddleware(ruleset)

        request_a = _make_request(thread_id="session-A")
        request_b = _make_request(thread_id="session-B", call_id="call_b")
        handler_a = _make_handler()
        handler_b = _make_handler(call_id="call_b")

        # Session A approves with "always".
        with patch("relay.middlewares.permission.interrupt", return_value="always"):
            await middleware.awrap_tool_call(request_a, handler_a)

        # Session B should still ask (the rule from A must not propagate).
        with patch(
            "relay.middlewares.permission.interrupt", return_value="once"
        ) as interrupt_b:
            await middleware.awrap_tool_call(request_b, handler_b)

        interrupt_b.assert_called_once()

    @pytest.mark.asyncio
    async def test_persisted_rules_seed_new_service(self):
        """Rules stored in context.permission_ruleset seed a new session service."""
        # Pre-populate context with a persisted allow rule for "bash".
        persisted = [{"permission": "bash", "pattern": "*", "action": "allow"}]
        ruleset = _make_ruleset(_ask("bash", "*"))  # config ruleset would ask
        middleware = PermissionMiddleware(ruleset)

        request = _make_request(
            thread_id="seeded-session",
            permission_ruleset=persisted,
            tool_metadata={
                "permission_config": {
                    "permission": "bash",
                    "patterns_fn": lambda args: ["git status"],
                }
            },
        )
        handler = _make_handler()

        # No interrupt should fire because the persisted 'always' rule covers bash.
        with patch(
            "relay.middlewares.permission.interrupt"
        ) as interrupt_mock:
            result = await middleware.awrap_tool_call(request, handler)

        interrupt_mock.assert_not_called()
        assert isinstance(result, ToolMessage)


# ==============================================================================
# Catalog proxy look-through
# ==============================================================================


class TestCatalogProxyLookthrough:
    """is_catalog_proxy routes evaluation to the underlying tool."""

    @pytest.mark.asyncio
    async def test_catalog_proxy_uses_underlying_permission(self):
        """run_tool should evaluate the underlying tool's permission config."""
        underlying_tool = Mock()
        underlying_tool.name = "edit_file"
        underlying_tool.metadata = {
            "permission_config": {
                "permission": "edit",
                "patterns_fn": lambda args: [args.get("file_path", "*")],
                "always_fn": lambda args: ["*"],
            }
        }

        ruleset = _make_ruleset(
            _allow("edit", "*"),  # allow all edit ops
        )
        middleware = PermissionMiddleware(ruleset)
        request = _make_request(
            tool_name="run_tool",
            args={"tool_name": "edit_file", "tool_args": {"file_path": "src/main.py"}},
            tool_metadata={"permission_config": {"is_catalog_proxy": True}},
            tool_catalog=[underlying_tool],
        )
        handler = _make_handler(tool_name="run_tool")

        with patch("relay.middlewares.permission.interrupt") as interrupt_mock:
            result = await middleware.awrap_tool_call(request, handler)

        # The underlying edit permission is allowed → no interrupt.
        interrupt_mock.assert_not_called()
        assert isinstance(result, ToolMessage)
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_catalog_proxy_unknown_tool_falls_back(self):
        """If the underlying tool is not in the catalog, fall back gracefully."""
        ruleset = _make_ruleset(_allow("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request(
            tool_name="run_tool",
            args={"tool_name": "nonexistent", "tool_args": {}},
            tool_metadata={"permission_config": {"is_catalog_proxy": True}},
            tool_catalog=[],  # empty catalog
        )
        handler = _make_handler(tool_name="run_tool")

        # Should not crash; falls back to default 'allow' from ruleset.
        with patch("relay.middlewares.permission.interrupt") as interrupt_mock:
            result = await middleware.awrap_tool_call(request, handler)

        interrupt_mock.assert_not_called()
        handler.assert_called_once()


# ==============================================================================
# Permission config fallback (no metadata)
# ==============================================================================


class TestNoPermissionConfigFallback:
    """Tools without permission_config use the tool name as permission key."""

    @pytest.mark.asyncio
    async def test_tool_name_used_as_permission_key_when_no_config(self):
        """Absence of permission_config falls back to tool name + '*' pattern."""
        # Ruleset explicitly allows the tool by name.
        ruleset = _make_ruleset(_allow("my_custom_tool", "*"))
        middleware = PermissionMiddleware(ruleset)
        # No permission_config in metadata.
        request = _make_request(tool_name="my_custom_tool", tool_metadata={})
        handler = _make_handler(tool_name="my_custom_tool")

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_without_metadata_uses_defaults(self):
        """A tool with no metadata at all should use safe fallback defaults."""
        ruleset = _make_ruleset(_allow("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request(tool_metadata=None)
        # Simulate tool with no metadata at all.
        request.tool.metadata = None
        handler = _make_handler()

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        handler.assert_called_once()


# ==============================================================================
# Exception handling
# ==============================================================================


class TestExceptionHandling:
    """Non-interrupt exceptions produce an error ToolMessage."""

    @pytest.mark.asyncio
    async def test_exception_in_handler_returns_error(self):
        ruleset = _make_ruleset(_allow("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request()
        handler = AsyncMock(side_effect=RuntimeError("boom"))

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "boom" in result.content

    @pytest.mark.asyncio
    async def test_graph_interrupt_propagates(self):
        """LangGraph GraphInterrupt must not be swallowed; it must propagate."""
        from langgraph.errors import GraphInterrupt

        ruleset = _make_ruleset(_ask("*", "*"))
        middleware = PermissionMiddleware(ruleset)
        request = _make_request()
        handler = _make_handler()

        # interrupt() itself raises GraphInterrupt internally; simulate that.
        with patch(
            "relay.middlewares.permission.interrupt",
            side_effect=GraphInterrupt(),
        ):
            with pytest.raises(GraphInterrupt):
                await middleware.awrap_tool_call(request, handler)


# ==============================================================================
# Stack ordering: PermissionMiddleware before CompressToolOutputMiddleware
# ==============================================================================


class TestStackOrdering:
    """create_react_agent stacks PermissionMiddleware before Compress; no Sandbox."""

    def test_stack_contains_permission_middleware(self):
        """create_react_agent must include PermissionMiddleware in wrapToolCall."""
        from unittest.mock import MagicMock

        from relay.agents.react_agent import create_react_agent
        from relay.middlewares.compress_tool_output import CompressToolOutputMiddleware

        captured_middleware: list = []

        def _fake_create_agent(**kwargs):
            captured_middleware.extend(kwargs.get("middleware", []))
            return MagicMock()

        with patch("relay.agents.react_agent.create_agent", side_effect=_fake_create_agent):
            create_react_agent(
                model=MagicMock(),
                tools=[],
                prompt="test",
                permission_ruleset=from_config(DEFAULT_PERMISSION),
            )

        wrap_types = [type(m).__name__ for m in captured_middleware]
        assert "PermissionMiddleware" in wrap_types
        assert "CompressToolOutputMiddleware" in wrap_types
        assert "SandboxMiddleware" not in wrap_types

    def test_permission_middleware_before_compress(self):
        """PermissionMiddleware must appear before CompressToolOutputMiddleware."""
        from unittest.mock import MagicMock

        from relay.agents.react_agent import create_react_agent
        from relay.middlewares.compress_tool_output import CompressToolOutputMiddleware

        captured_middleware: list = []

        def _fake_create_agent(**kwargs):
            captured_middleware.extend(kwargs.get("middleware", []))
            return MagicMock()

        with patch("relay.agents.react_agent.create_agent", side_effect=_fake_create_agent):
            create_react_agent(
                model=MagicMock(),
                tools=[],
                prompt="test",
            )

        perm_idx = next(
            i for i, m in enumerate(captured_middleware)
            if type(m).__name__ == "PermissionMiddleware"
        )
        compress_idx = next(
            i for i, m in enumerate(captured_middleware)
            if type(m).__name__ == "CompressToolOutputMiddleware"
        )
        assert perm_idx < compress_idx


# ==============================================================================
# _command_prefix helper (in terminal tool)
# ==============================================================================


class TestCommandPrefix:
    """_command_prefix extracts a meaningful prefix from shell commands."""

    def test_git_push_with_flags(self):
        from relay.tools.impl.terminal import _command_prefix

        assert _command_prefix("git push --force origin main") == "git push"

    def test_npm_install_with_package(self):
        from relay.tools.impl.terminal import _command_prefix

        assert _command_prefix("npm install foo bar") == "npm install"

    def test_ls_with_flag(self):
        from relay.tools.impl.terminal import _command_prefix

        # Second token starts with '-', so only the command is returned.
        assert _command_prefix("ls -la /tmp") == "ls"

    def test_rm_with_flag(self):
        from relay.tools.impl.terminal import _command_prefix

        assert _command_prefix("rm -rf /") == "rm"

    def test_empty_command(self):
        from relay.tools.impl.terminal import _command_prefix

        assert _command_prefix("") == "*"

    def test_stops_at_shell_operator(self):
        from relay.tools.impl.terminal import _command_prefix

        # Compound command: prefix is from the first segment only.
        assert _command_prefix("git fetch && git rebase") == "git fetch"

    def test_single_token(self):
        from relay.tools.impl.terminal import _command_prefix

        assert _command_prefix("make") == "make"


# ==============================================================================
# Tool permission_config metadata
# ==============================================================================


class TestToolPermissionConfig:
    """Verify that tool permission_config metadata is set correctly."""

    def test_run_command_has_bash_permission(self):
        from relay.tools.impl.terminal import run_command

        cfg = run_command.metadata["permission_config"]
        assert cfg["permission"] == "bash"
        assert callable(cfg["patterns_fn"])
        assert callable(cfg["always_fn"])
        assert callable(cfg["metadata_fn"])

    def test_run_command_patterns_fn_extracts_command(self):
        from relay.tools.impl.terminal import run_command

        cfg = run_command.metadata["permission_config"]
        patterns = cfg["patterns_fn"]({"command": "git status"})
        assert patterns == ["git status"]

    def test_run_command_always_fn_produces_prefix_wildcard(self):
        from relay.tools.impl.terminal import run_command

        cfg = run_command.metadata["permission_config"]
        always = cfg["always_fn"]({"command": "git push --force"})
        assert always == ["git push *"]

    def test_read_file_has_read_permission(self):
        from relay.tools.impl.filesystem import read_file

        cfg = read_file.metadata["permission_config"]
        assert cfg["permission"] == "read"

    def test_edit_file_has_edit_permission(self):
        from relay.tools.impl.filesystem import edit_file

        cfg = edit_file.metadata["permission_config"]
        assert cfg["permission"] == "edit"

    def test_write_file_has_edit_permission(self):
        from relay.tools.impl.filesystem import write_file

        cfg = write_file.metadata["permission_config"]
        assert cfg["permission"] == "edit"

    def test_delete_file_has_edit_permission(self):
        from relay.tools.impl.filesystem import delete_file

        cfg = delete_file.metadata["permission_config"]
        assert cfg["permission"] == "edit"

    def test_glob_files_has_glob_permission(self):
        from relay.tools.impl.filesystem import glob_files

        cfg = glob_files.metadata["permission_config"]
        assert cfg["permission"] == "glob"

    def test_ls_has_list_permission(self):
        from relay.tools.impl.filesystem import ls

        cfg = ls.metadata["permission_config"]
        assert cfg["permission"] == "list"

    def test_run_tool_has_catalog_proxy_flag(self):
        from relay.tools.catalog.tools import run_tool

        cfg = run_tool.metadata["permission_config"]
        assert cfg.get("is_catalog_proxy") is True


# ==============================================================================
# Factory: _build_permission_ruleset
# ==============================================================================


class TestFactoryPermissionRuleset:
    """AgentFactory._build_permission_ruleset merges defaults with agent config."""

    def test_defaults_present_in_ruleset(self):
        from relay.agents.factory import AgentFactory
        from relay.configs.agent import AgentConfig

        agent_cfg = AgentConfig(name="test", permission=None)
        ruleset = AgentFactory._build_permission_ruleset(agent_cfg)

        # DEFAULT_PERMISSION has '*': 'allow' as first entry.
        assert any(r.permission == "*" and r.action == "allow" for r in ruleset)

    def test_agent_overrides_appended(self):
        from relay.agents.factory import AgentFactory
        from relay.configs.agent import AgentConfig

        agent_cfg = AgentConfig(name="test", permission={"bash": "deny"})
        ruleset = AgentFactory._build_permission_ruleset(agent_cfg)

        # Agent's deny rule must appear after defaults (last-match-wins).
        bash_rules = [r for r in ruleset if r.permission == "bash"]
        assert bash_rules, "Expected at least one bash rule"
        # The last bash rule should be 'deny' (agent override wins over default).
        assert bash_rules[-1].action == "deny"

    def test_empty_agent_permission_uses_defaults_only(self):
        from relay.agents.factory import AgentFactory
        from relay.configs.agent import AgentConfig

        agent_cfg = AgentConfig(name="test", permission={})
        ruleset = AgentFactory._build_permission_ruleset(agent_cfg)

        # Should equal the default ruleset exactly.
        default_ruleset = from_config(DEFAULT_PERMISSION)
        assert len(ruleset) == len(default_ruleset)
