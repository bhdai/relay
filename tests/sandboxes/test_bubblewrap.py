"""Tests for BubblewrapBackend command building."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from relay.configs.sandbox import (
    FilesystemConfig,
    NetworkConfig,
    SandboxConfig,
    SandboxOS,
    SandboxType,
)
from relay.sandboxes.bubblewrap import BubblewrapBackend


def _make_config(
    *,
    read: list[str] | None = None,
    write: list[str] | None = None,
    hidden: list[str] | None = None,
    remote: list[str] | None = None,
    local: list[str] | None = None,
) -> SandboxConfig:
    return SandboxConfig(
        name="test",
        type=SandboxType.BUBBLEWRAP,
        os=SandboxOS.LINUX,
        filesystem=FilesystemConfig(
            read=read or ["."],
            write=write or ["."],
            hidden=hidden or [],
        ),
        network=NetworkConfig(
            remote=remote or [],
            local=local or [],
        ),
    )


class TestBubblewrapCommandBuilding:
    """Test that build_command produces correct bwrap arguments."""

    def test_basic_command_includes_bwrap(self, tmp_path: Path):
        config = _make_config()
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["python", "-m", "relay.sandboxes.worker"])
        assert cmd[0] == "bwrap"
        assert "--clearenv" in cmd
        assert "python" in cmd

    def test_tmpfs_root(self, tmp_path: Path):
        config = _make_config()
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        # Should have --tmpfs / for empty root
        idx = cmd.index("--tmpfs")
        assert cmd[idx + 1] == "/"

    def test_read_mount_for_working_dir(self, tmp_path: Path):
        config = _make_config(read=["."], write=[])
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        resolved = str(tmp_path.resolve())
        assert "--ro-bind" in cmd
        ro_idx = cmd.index("--ro-bind")
        assert cmd[ro_idx + 1] == resolved

    def test_write_mount_for_working_dir(self, tmp_path: Path):
        config = _make_config(read=[], write=["."])
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        resolved = str(tmp_path.resolve())
        # --bind (rw) should appear for write
        assert "--bind" in cmd
        bind_idx = cmd.index("--bind")
        assert cmd[bind_idx + 1] == resolved

    def test_network_unshared_by_default(self, tmp_path: Path):
        config = _make_config(remote=[])
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        assert "--unshare-net" in cmd
        assert "--share-net" not in cmd

    def test_network_shared_with_wildcard(self, tmp_path: Path):
        config = _make_config(remote=["*"])
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        assert "--share-net" in cmd
        assert "--unshare-net" not in cmd

    def test_namespace_isolation(self, tmp_path: Path):
        config = _make_config()
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        assert "--unshare-pid" in cmd
        assert "--unshare-ipc" in cmd
        assert "--unshare-uts" in cmd
        assert "--die-with-parent" in cmd
        assert "--new-session" in cmd

    def test_chdir_to_working_dir(self, tmp_path: Path):
        config = _make_config(read=["."])
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        chdir_idx = cmd.index("--chdir")
        assert cmd[chdir_idx + 1] == str(tmp_path.resolve())

    def test_hidden_file_mapped_to_dev_null(self, tmp_path: Path):
        secret = tmp_path / ".env"
        secret.write_text("SECRET=value")
        config = _make_config(hidden=[str(secret)])
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        # Hidden file should be overridden with /dev/null
        assert "/dev/null" in cmd

    def test_hidden_dir_mapped_to_tmpfs(self, tmp_path: Path):
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        config = _make_config(hidden=[str(ssh_dir)])
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        # The hidden dir should be covered by --tmpfs
        tmpfs_positions = [i for i, v in enumerate(cmd) if v == "--tmpfs"]
        hidden_covered = any(cmd[i + 1] == str(ssh_dir) for i in tmpfs_positions)
        assert hidden_covered

    def test_uid_gid_set(self, tmp_path: Path):
        config = _make_config()
        backend = BubblewrapBackend(config, tmp_path)
        cmd = backend.build_command(["echo"])
        uid_idx = cmd.index("--uid")
        assert cmd[uid_idx + 1] == str(os.getuid())
        gid_idx = cmd.index("--gid")
        assert cmd[gid_idx + 1] == str(os.getgid())


class TestBubblewrapValidation:
    """Test validate_environment checks."""

    def test_missing_bwrap_raises(self, tmp_path: Path):
        config = _make_config()
        backend = BubblewrapBackend(config, tmp_path)
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="bwrap not found"):
                backend.validate_environment()

    def test_present_bwrap_succeeds(self, tmp_path: Path):
        config = _make_config()
        backend = BubblewrapBackend(config, tmp_path)
        with patch("shutil.which", return_value="/usr/bin/bwrap"):
            backend.validate_environment()  # should not raise
