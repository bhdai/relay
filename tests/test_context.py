"""Unit tests for AgentContext defaults and template_vars."""

import os
import sys
from pathlib import Path

from relay.agents.context import AgentContext


class TestAgentContextDefaults:
    def test_default_working_dir(self):
        ctx = AgentContext()
        assert ctx.working_dir == str(Path.cwd())

    def test_default_platform(self):
        ctx = AgentContext()
        assert ctx.platform == sys.platform

    def test_default_shell(self):
        ctx = AgentContext()
        assert ctx.shell == os.environ.get("SHELL", "")

    def test_default_date_time_is_iso(self):
        ctx = AgentContext()
        # ISO 8601 strings contain 'T' as the date-time separator.
        assert "T" in ctx.current_date_time_zoned

    def test_custom_values(self):
        ctx = AgentContext(
            working_dir="/tmp/test",
            platform="linux",
            shell="/bin/zsh",
            user_memory="remember this",
            input_cost_per_mtok=3.0,
            output_cost_per_mtok=15.0,
        )
        assert ctx.working_dir == "/tmp/test"
        assert ctx.platform == "linux"
        assert ctx.shell == "/bin/zsh"
        assert ctx.user_memory == "remember this"
        assert ctx.input_cost_per_mtok == 3.0
        assert ctx.output_cost_per_mtok == 15.0


class TestTemplateVars:
    def test_keys_present(self):
        ctx = AgentContext(working_dir="/w", platform="test-os", shell="/bin/fish", user_memory="mem")
        tvars = ctx.template_vars
        assert tvars["working_dir"] == "/w"
        assert tvars["platform"] == "test-os"
        assert tvars["shell"] == "/bin/fish"
        assert tvars["user_memory"] == "mem"
        assert "current_date_time_zoned" in tvars

    def test_cost_fields_excluded(self):
        """Cost rates are runtime metadata, not prompt template vars."""
        ctx = AgentContext(input_cost_per_mtok=3.0, output_cost_per_mtok=15.0)
        tvars = ctx.template_vars
        assert "input_cost_per_mtok" not in tvars
        assert "output_cost_per_mtok" not in tvars
