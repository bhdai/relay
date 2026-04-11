"""Unit tests for terminal tool helpers — no subprocess calls."""

from relay.tools.terminal import _extract_command_parts, _format_output


# ==============================================================================
# _extract_command_parts
# ==============================================================================


class TestExtractCommandParts:
    def test_simple_command(self):
        parts = _extract_command_parts("ls -la")
        assert parts == ["ls -la"]

    def test_chained_and(self):
        parts = _extract_command_parts("cd src && make build")
        assert "cd src" in parts
        assert "make build" in parts

    def test_pipe(self):
        parts = _extract_command_parts("cat file.txt | grep error")
        assert "cat file.txt" in parts
        assert "grep error" in parts

    def test_nested_substitution(self):
        parts = _extract_command_parts("echo $(whoami)")
        assert "echo $(whoami)" in parts
        assert "whoami" in parts

    def test_empty_command(self):
        assert _extract_command_parts("") == []

    def test_semicolon_chain(self):
        parts = _extract_command_parts("cmd1 ; cmd2 ; cmd3")
        assert len(parts) == 3


# ==============================================================================
# _format_output
# ==============================================================================


class TestFormatOutput:
    def test_stdout_only(self):
        assert _format_output("hello\n", "") == "hello"

    def test_stderr_only(self):
        assert _format_output("", "warning\n") == "warning"

    def test_both(self):
        result = _format_output("out\n", "err\n")
        assert "out" in result
        assert "err" in result

    def test_empty(self):
        assert _format_output("", "") == "Command completed successfully"
