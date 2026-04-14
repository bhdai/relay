"""Tests for relay.configs.utils — YAML loading and prompt content resolution."""

from pathlib import Path

import pytest

from relay.configs.utils import load_prompt_content, load_yaml_dir, load_yaml_file


# ==============================================================================
# load_yaml_dir
# ==============================================================================


class TestLoadYamlDir:
    async def test_loads_single_dict_files(self, tmp_path: Path):
        d = tmp_path / "cfg"
        d.mkdir()
        (d / "a.yml").write_text("name: alpha\nvalue: 1\n")
        (d / "b.yml").write_text("name: beta\nvalue: 2\n")

        items = await load_yaml_dir(d)
        assert len(items) == 2
        names = {i["name"] for i in items}
        assert names == {"alpha", "beta"}

    async def test_loads_list_files(self, tmp_path: Path):
        d = tmp_path / "cfg"
        d.mkdir()
        (d / "batch.yml").write_text("- name: one\n- name: two\n")

        items = await load_yaml_dir(d)
        assert len(items) == 2

    async def test_nonexistent_dir_returns_empty(self, tmp_path: Path):
        items = await load_yaml_dir(tmp_path / "nope")
        assert items == []

    async def test_ignores_non_yml_files(self, tmp_path: Path):
        d = tmp_path / "cfg"
        d.mkdir()
        (d / "readme.md").write_text("# not yaml")
        (d / "ok.yml").write_text("name: ok\n")

        items = await load_yaml_dir(d)
        assert len(items) == 1

    async def test_sorted_order(self, tmp_path: Path):
        d = tmp_path / "cfg"
        d.mkdir()
        (d / "z.yml").write_text("name: z\n")
        (d / "a.yml").write_text("name: a\n")

        items = await load_yaml_dir(d)
        assert items[0]["name"] == "a"
        assert items[1]["name"] == "z"


# ==============================================================================
# load_yaml_file
# ==============================================================================


class TestLoadYamlFile:
    async def test_loads_items_under_key(self, tmp_path: Path):
        f = tmp_path / "agents.yml"
        f.write_text("agents:\n  - name: general\n  - name: coder\n")

        items = await load_yaml_file(f, "agents")
        assert len(items) == 2

    async def test_missing_key_returns_empty(self, tmp_path: Path):
        f = tmp_path / "agents.yml"
        f.write_text("other:\n  - name: x\n")

        items = await load_yaml_file(f, "agents")
        assert items == []

    async def test_nonexistent_file_returns_empty(self, tmp_path: Path):
        items = await load_yaml_file(tmp_path / "nope.yml", "agents")
        assert items == []


# ==============================================================================
# load_prompt_content
# ==============================================================================


class TestLoadPromptContent:
    async def test_loads_single_file(self, tmp_path: Path):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "main.md").write_text("You are helpful.")

        result = await load_prompt_content(tmp_path, "prompts/main.md")
        assert result == "You are helpful."

    async def test_concatenates_multiple_files(self, tmp_path: Path):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "a.md").write_text("Part A")
        (prompt_dir / "b.md").write_text("Part B")

        result = await load_prompt_content(tmp_path, ["prompts/a.md", "prompts/b.md"])
        assert "Part A" in result
        assert "Part B" in result
        assert result == "Part A\n\nPart B"

    async def test_literal_string_passthrough(self, tmp_path: Path):
        result = await load_prompt_content(tmp_path, "You are a literal prompt.")
        assert result == "You are a literal prompt."

    async def test_mixed_files_and_literals(self, tmp_path: Path):
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "real.md").write_text("From file")

        result = await load_prompt_content(
            tmp_path, ["prompts/real.md", "literal text"]
        )
        assert "From file" in result
        assert "literal text" in result

    async def test_none_returns_empty(self, tmp_path: Path):
        result = await load_prompt_content(tmp_path, None)
        assert result == ""

    async def test_empty_string_returns_empty(self, tmp_path: Path):
        result = await load_prompt_content(tmp_path, "")
        assert result == ""
