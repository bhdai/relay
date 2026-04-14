"""Tests for ApprovalMiddleware."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from relay.agents.context import AgentContext
from relay.configs.approval import ApprovalMode, ToolApprovalConfig, ToolApprovalRule
from relay.middlewares.approval import (
    ALLOW,
    ALWAYS_ALLOW,
    ALWAYS_DENY,
    DENY,
    ApprovalMiddleware,
    create_field_extractor,
    create_field_transformer,
)


# ==============================================================================
# Rule checking
# ==============================================================================


class TestCheckApprovalRules:
    """Tests for ApprovalMiddleware._check_approval_rules."""

    def test_deny_takes_highest_priority(self):
        config = ToolApprovalConfig(
            always_deny=[ToolApprovalRule(name="tool", args=None)],
            always_allow=[ToolApprovalRule(name="tool", args=None)],
        )
        decision, is_ask = ApprovalMiddleware._check_approval_rules(config, "tool", {})
        assert decision is False
        assert is_ask is False

    def test_allow_when_matched(self):
        config = ToolApprovalConfig(
            always_allow=[ToolApprovalRule(name="tool", args={"q": "ok"})],
        )
        decision, is_ask = ApprovalMiddleware._check_approval_rules(
            config, "tool", {"q": "ok"}
        )
        assert decision is True
        assert is_ask is False

    def test_always_ask_returns_none_with_flag(self):
        config = ToolApprovalConfig(
            always_ask=[ToolApprovalRule(name="tool", args={"cmd": "rm.*"})],
        )
        decision, is_ask = ApprovalMiddleware._check_approval_rules(
            config, "tool", {"cmd": "rm -rf /"}
        )
        assert decision is None
        assert is_ask is True

    def test_no_match_returns_none(self):
        config = ToolApprovalConfig()
        decision, is_ask = ApprovalMiddleware._check_approval_rules(
            config, "unknown", {}
        )
        assert decision is None
        assert is_ask is False


# ==============================================================================
# Mode bypass
# ==============================================================================


class TestCheckApprovalModeBypass:
    """Tests for ApprovalMiddleware._check_approval_mode_bypass."""

    def test_semi_active_never_bypasses(self):
        config = ToolApprovalConfig()
        assert (
            ApprovalMiddleware._check_approval_mode_bypass(
                ApprovalMode.SEMI_ACTIVE, config, "tool", {}
            )
            is False
        )

    def test_active_bypasses_by_default(self):
        config = ToolApprovalConfig()
        assert (
            ApprovalMiddleware._check_approval_mode_bypass(
                ApprovalMode.ACTIVE, config, "tool", {}
            )
            is True
        )

    def test_active_respects_deny(self):
        config = ToolApprovalConfig(
            always_deny=[ToolApprovalRule(name="tool", args=None)],
        )
        assert (
            ApprovalMiddleware._check_approval_mode_bypass(
                ApprovalMode.ACTIVE, config, "tool", {}
            )
            is False
        )

    def test_active_respects_always_ask(self):
        config = ToolApprovalConfig(
            always_ask=[ToolApprovalRule(name="critical", args=None)],
        )
        assert (
            ApprovalMiddleware._check_approval_mode_bypass(
                ApprovalMode.ACTIVE, config, "critical", {}
            )
            is False
        )
        # Other tools still bypass.
        assert (
            ApprovalMiddleware._check_approval_mode_bypass(
                ApprovalMode.ACTIVE, config, "safe", {}
            )
            is True
        )

    def test_aggressive_bypasses_unless_denied(self):
        config = ToolApprovalConfig(
            always_deny=[ToolApprovalRule(name="blocked", args=None)],
        )
        assert (
            ApprovalMiddleware._check_approval_mode_bypass(
                ApprovalMode.AGGRESSIVE, config, "tool", {}
            )
            is True
        )
        assert (
            ApprovalMiddleware._check_approval_mode_bypass(
                ApprovalMode.AGGRESSIVE, config, "blocked", {}
            )
            is False
        )


# ==============================================================================
# Persistence
# ==============================================================================


class TestSaveApprovalDecision:
    """Tests for ApprovalMiddleware._save_approval_decision."""

    def test_save_allow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.approval.json"
            config = ToolApprovalConfig()
            config.save_to_json_file(config_file)

            ApprovalMiddleware._save_approval_decision(
                config, config_file, "tool", {"q": "x"}, allow=True
            )

            loaded = ToolApprovalConfig.from_json_file(config_file)
            assert any(r.name == "tool" for r in loaded.always_allow)

    def test_save_deny(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.approval.json"
            config = ToolApprovalConfig()
            config.save_to_json_file(config_file)

            ApprovalMiddleware._save_approval_decision(
                config, config_file, "tool", None, allow=False
            )

            loaded = ToolApprovalConfig.from_json_file(config_file)
            assert any(r.name == "tool" for r in loaded.always_deny)

    def test_replaces_existing_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.approval.json"
            config = ToolApprovalConfig(
                always_allow=[ToolApprovalRule(name="tool", args={"q": "x"})],
            )
            config.save_to_json_file(config_file)

            ApprovalMiddleware._save_approval_decision(
                config, config_file, "tool", {"q": "x"}, allow=False
            )

            loaded = ToolApprovalConfig.from_json_file(config_file)
            assert len(loaded.always_allow) == 0
            assert len(loaded.always_deny) == 1


# ==============================================================================
# awrap_tool_call
# ==============================================================================


def _make_request(
    tool_name: str = "test_tool",
    call_id: str = "call_1",
    args: dict | None = None,
    *,
    approval_mode: ApprovalMode = ApprovalMode.AGGRESSIVE,
    working_dir: str = "/tmp",
    tool_metadata: dict | None = None,
) -> Mock:
    """Build a mock ToolCallRequest."""
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
    request.runtime.context = AgentContext(
        approval_mode=approval_mode,
        working_dir=working_dir,
    )
    return request


class TestAwrapToolCall:
    """Tests for ApprovalMiddleware.awrap_tool_call."""

    @pytest.mark.asyncio
    async def test_allow_executes_tool(self):
        middleware = ApprovalMiddleware()
        request = _make_request(approval_mode=ApprovalMode.AGGRESSIVE)
        handler = AsyncMock(
            return_value=ToolMessage(
                name="test_tool", content="result", tool_call_id="call_1"
            )
        )

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "result"
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_deny_blocks_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ToolApprovalConfig(
                always_deny=[ToolApprovalRule(name="test_tool", args=None)],
            )
            config.save_to_json_file(
                Path(tmpdir) / ".relay" / "config.approval.json"
            )

            middleware = ApprovalMiddleware()
            request = _make_request(
                approval_mode=ApprovalMode.SEMI_ACTIVE,
                working_dir=tmpdir,
            )

            with patch(
                "relay.middlewares.approval.interrupt", return_value=DENY
            ):
                handler = AsyncMock()
                result = await middleware.awrap_tool_call(request, handler)

            assert isinstance(result, ToolMessage)
            assert result.status == "error"
            assert "denied" in result.content.lower()
            handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_returns_error_message(self):
        middleware = ApprovalMiddleware()
        request = _make_request(approval_mode=ApprovalMode.AGGRESSIVE)
        handler = AsyncMock(side_effect=RuntimeError("boom"))

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "boom" in result.content

    @pytest.mark.asyncio
    async def test_cache_returns_same_result(self):
        middleware = ApprovalMiddleware()
        request = _make_request(approval_mode=ApprovalMode.AGGRESSIVE)
        handler = AsyncMock(
            return_value=ToolMessage(
                name="test_tool", content="first", tool_call_id="call_1"
            )
        )

        first_result = await middleware.awrap_tool_call(request, handler)
        # Second call with same tool_call_id should hit cache.
        second_result = await middleware.awrap_tool_call(request, handler)

        assert first_result.content == "first"
        assert second_result.content == "first"
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_always_approve_metadata_skips_rules(self):
        """Tool with always_approve=True in metadata bypasses all checks."""
        middleware = ApprovalMiddleware()
        request = _make_request(
            approval_mode=ApprovalMode.SEMI_ACTIVE,
            tool_metadata={"approval_config": {"always_approve": True}},
        )
        handler = AsyncMock(
            return_value=ToolMessage(
                name="test_tool", content="ok", tool_call_id="call_1"
            )
        )

        result = await middleware.awrap_tool_call(request, handler)
        assert result.content == "ok"
        handler.assert_called_once()


# ==============================================================================
# Arg formatting helpers
# ==============================================================================


class TestFieldExtractor:
    def test_extract_single_field(self):
        extractor = create_field_extractor({"command": r"(?P<command>\S+)"})
        result = extractor({"command": "echo hello world"})
        assert result["command"] == "echo"

    def test_no_match_keeps_original(self):
        extractor = create_field_extractor({"command": r"(?P<command>XYZ)"})
        result = extractor({"command": "echo hello"})
        assert result["command"] == "echo hello"


class TestFieldTransformer:
    def test_transform_field(self):
        transformer = create_field_transformer(
            {"command": lambda x: x.split()[0]}
        )
        result = transformer({"command": "git push origin", "other": "kept"})
        assert result["command"] == "git"
        assert result["other"] == "kept"

    def test_failure_keeps_original(self):
        transformer = create_field_transformer(
            {"command": lambda x: x.split()[99]}  # IndexError
        )
        result = transformer({"command": "hello"})
        assert result["command"] == "hello"
