"""Tests for relay.skills.factory — Skill model and SkillFactory."""

from pathlib import Path

import pytest

from relay.skills.factory import Skill, SkillFactory


def _write_skill(skills_dir: Path, category: str, name: str, **extra) -> Path:
    """Write a minimal SKILL.md file and return its path."""
    skill_dir = skills_dir / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"

    frontmatter = {"name": name, "description": f"A {name} skill", **extra}
    import yaml

    lines = [
        "---",
        yaml.dump(frontmatter, default_flow_style=False).strip(),
        "---",
        "",
        f"# {name}",
        "",
        "Skill content goes here.",
    ]
    skill_md.write_text("\n".join(lines))
    return skill_md


# ==============================================================================
# Skill model
# ==============================================================================


class TestSkill:
    async def test_from_file_parses_frontmatter(self, tmp_path: Path):
        skill_md = _write_skill(
            tmp_path, "coding", "python-lint", allowed_tools=["run_command"]
        )
        skill = await Skill.from_file(skill_md, "coding")

        assert skill is not None
        assert skill.name == "python-lint"
        assert skill.description == "A python-lint skill"
        assert skill.category == "coding"
        assert skill.allowed_tools == ["run_command"]

    async def test_from_file_returns_none_without_frontmatter(self, tmp_path: Path):
        skill_dir = tmp_path / "coding" / "bad"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# No frontmatter\n\nJust content.")

        result = await Skill.from_file(skill_md, "coding")
        assert result is None

    async def test_from_file_returns_none_missing_name(self, tmp_path: Path):
        skill_dir = tmp_path / "coding" / "noname"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\ndescription: test\n---\n\nContent.")

        result = await Skill.from_file(skill_md, "coding")
        assert result is None

    def test_read_content(self, tmp_path: Path):
        skill_md = _write_skill(tmp_path, "coding", "reader")
        skill = Skill(
            name="reader",
            description="test",
            category="coding",
            path=skill_md,
        )
        content = skill.read_content()
        assert "# reader" in content


# ==============================================================================
# SkillFactory
# ==============================================================================


class TestSkillFactory:
    async def test_load_skills_scans_categories(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "coding", "python-lint")
        _write_skill(skills_dir, "coding", "code-review")
        _write_skill(skills_dir, "general", "summarize")

        factory = SkillFactory()
        result = await factory.load_skills(skills_dir)

        assert "coding" in result
        assert "general" in result
        assert len(result["coding"]) == 2
        assert len(result["general"]) == 1

    async def test_load_skills_empty_dir(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        factory = SkillFactory()
        result = await factory.load_skills(skills_dir)
        assert result == {}

    async def test_load_skills_nonexistent_dir(self, tmp_path: Path):
        factory = SkillFactory()
        result = await factory.load_skills(tmp_path / "nonexistent")
        assert result == {}

    async def test_get_skill(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "coding", "python-lint")

        factory = SkillFactory()
        await factory.load_skills(skills_dir)

        skill = factory.get_skill("coding", "python-lint")
        assert skill is not None
        assert skill.name == "python-lint"

        missing = factory.get_skill("coding", "nonexistent")
        assert missing is None

    async def test_module_map(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "coding", "python-lint")

        factory = SkillFactory()
        await factory.load_skills(skills_dir)

        module_map = factory.get_module_map()
        assert "coding:python-lint" in module_map

    async def test_skips_non_directories(self, tmp_path: Path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        # Create a file at the category level (should be skipped).
        (skills_dir / "README.md").write_text("Not a category")
        _write_skill(skills_dir, "coding", "valid-skill")

        factory = SkillFactory()
        result = await factory.load_skills(skills_dir)
        assert "coding" in result
        # README.md should not appear as a category.
        assert "README.md" not in result
