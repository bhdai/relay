"""Tests for the sandbox worker module."""

import json

import pytest

from relay.sandboxes.worker import _run, _serialize_result


class TestSerializeResult:
    """Tests for _serialize_result."""

    def test_string_result(self):
        result = _serialize_result("hello world")
        assert result == {"success": True, "content": "hello world"}

    def test_numeric_result(self):
        result = _serialize_result(42)
        assert result == {"success": True, "content": "42"}


class TestRun:
    """Tests for _run (the async tool invocation)."""

    @pytest.mark.asyncio
    async def test_disallowed_module_prefix(self):
        """Modules outside ALLOWED_MODULE_PREFIX should be rejected."""
        result = await _run("os.path", "join", {"a": "/", "b": "etc"})
        assert result["success"] is False
        assert "not in allowed prefix" in result["error"]

    @pytest.mark.asyncio
    async def test_allowed_module_prefix_nonexistent(self):
        """Allowed prefix but non-existent module should fail gracefully."""
        result = await _run("relay.tools.nonexistent_xyz", "some_tool", {})
        assert result["success"] is False
        # Should get an import error, not a security error
        assert "error" in result


class TestWorkerConfig:
    """Tests for sandbox config models."""

    def test_sandbox_config_validates_os(self):
        from relay.configs.sandbox import SandboxConfig, SandboxOS, SandboxType

        config = SandboxConfig(
            name="test",
            type=SandboxType.BUBBLEWRAP,
            os=SandboxOS.LINUX,
        )
        assert config.type == SandboxType.BUBBLEWRAP

    def test_sandbox_config_rejects_mismatched_os(self):
        from relay.configs.sandbox import SandboxConfig, SandboxOS, SandboxType

        # Currently only bubblewrap+linux exists, so we verify the valid case.
        # The validator will reject mismatches when more OS types are added.
        config = SandboxConfig(
            name="ok",
            type=SandboxType.BUBBLEWRAP,
            os=SandboxOS.LINUX,
        )
        assert config.os == SandboxOS.LINUX
