"""Tests for relay.tools.factory — ToolFactory registration and module maps."""

from relay.tools.factory import ToolFactory


class TestToolFactory:
    def test_impl_tools_are_registered(self):
        """All impl tool groups (filesystem, terminal, web) are present."""
        factory = ToolFactory()
        names = {t.name for t in factory.get_impl_tools()}

        # Spot-check representative tools from each group.
        assert "read_file" in names
        assert "write_file" in names
        assert "run_command" in names
        assert "fetch_web_content" in names

    def test_internal_tools_are_registered(self):
        """All internal tool groups (memory, todo) are present."""
        factory = ToolFactory()
        names = {t.name for t in factory.get_internal_tools()}

        assert "list_memory_files" in names
        assert "write_todos" in names
        assert "read_todos" in names

    def test_impl_module_map_has_correct_modules(self):
        """Impl module map uses the logical module names expected by YAML."""
        factory = ToolFactory()
        module_map = factory.get_impl_module_map()

        assert module_map["read_file"] == "file_system"
        assert module_map["glob_files"] == "file_system"
        assert module_map["grep_files"] == "file_system"
        assert module_map["ls"] == "file_system"

        assert module_map["run_command"] == "terminal"
        assert module_map["fetch_web_content"] == "web"

    def test_internal_module_map_has_correct_modules(self):
        """Internal module map records the source module for each tool."""
        factory = ToolFactory()
        module_map = factory.get_internal_module_map()

        assert module_map["list_memory_files"] == "memory"
        assert module_map["write_todos"] == "todo"
        assert module_map["read_todos"] == "todo"

    def test_no_overlap_between_impl_and_internal(self):
        """Impl and internal tool sets should be disjoint."""
        factory = ToolFactory()
        impl_names = {t.name for t in factory.get_impl_tools()}
        internal_names = {t.name for t in factory.get_internal_tools()}

        assert impl_names.isdisjoint(internal_names)
