"""Tests for SandboxFactory."""

from pathlib import Path
from unittest.mock import patch

import pytest

from relay.configs.sandbox import (
    AgentSandboxConfig,
    FilesystemConfig,
    NetworkConfig,
    SandboxConfig,
    SandboxOS,
    SandboxProfileBinding,
    SandboxType,
)
from relay.sandboxes.factory import SandboxFactory


def _make_sandbox_config(name: str = "test") -> SandboxConfig:
    return SandboxConfig(
        name=name,
        type=SandboxType.BUBBLEWRAP,
        os=SandboxOS.LINUX,
        filesystem=FilesystemConfig(read=["."], write=["."]),
        network=NetworkConfig(remote=["*"]),
    )


class TestSandboxFactory:
    """Tests for SandboxFactory.build_bindings."""

    @patch("relay.sandboxes.bubblewrap.shutil.which", return_value="/usr/bin/bwrap")
    def test_build_bindings_enabled(self, _mock_which, tmp_path: Path):
        factory = SandboxFactory()
        agent_config = AgentSandboxConfig(
            enabled=True,
            profiles=[
                SandboxProfileBinding(
                    patterns=["impl:terminal:*"],
                    sandbox=_make_sandbox_config(),
                ),
                SandboxProfileBinding(
                    patterns=["internal:*:*"],
                    sandbox=None,
                ),
            ],
        )
        bindings = factory.build_bindings(agent_config, tmp_path)
        assert len(bindings) == 2
        assert bindings[0].backend is not None
        assert bindings[1].backend is None

    def test_build_bindings_disabled(self, tmp_path: Path):
        factory = SandboxFactory()
        agent_config = AgentSandboxConfig(enabled=False, profiles=[])
        bindings = factory.build_bindings(agent_config, tmp_path)
        assert bindings == []

    @patch("relay.sandboxes.bubblewrap.shutil.which", return_value="/usr/bin/bwrap")
    def test_backend_reused_for_same_config(self, _mock_which, tmp_path: Path):
        factory = SandboxFactory()
        config = _make_sandbox_config()
        b1 = factory.create_backend(config, tmp_path)
        b2 = factory.create_backend(config, tmp_path)
        assert b1 is b2
