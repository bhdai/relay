from pathlib import Path

from relay import settings as settings_module
from relay.settings import Settings


def _clear_llm_env(monkeypatch) -> None:
    monkeypatch.delenv("LLM__OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM__MODEL", raising=False)
    monkeypatch.delenv("LLM__INPUT_COST_PER_MTOK", raising=False)
    monkeypatch.delenv("LLM__OUTPUT_COST_PER_MTOK", raising=False)


def test_settings_loads_from_env(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM__OPENAI_API_KEY=sk-test\n"
        "LLM__INPUT_COST_PER_MTOK=1.25\n"
        "LLM__OUTPUT_COST_PER_MTOK=5.0\n"
    )
    s = Settings(_env_file=str(env_file))
    assert s.llm.openai_api_key.get_secret_value() == "sk-test"
    assert s.llm.input_cost_per_mtok == 1.25
    assert s.llm.output_cost_per_mtok == 5.0


def test_find_env_file_prefers_current_directory(monkeypatch, tmp_path):
    _clear_llm_env(monkeypatch)

    cwd = tmp_path / "project"
    cwd.mkdir()
    local_env = cwd / ".env"
    local_env.write_text("LLM__OPENAI_API_KEY=sk-local\n")

    config_home = tmp_path / "config"
    global_env = config_home / "relay" / ".env"
    global_env.parent.mkdir(parents=True)
    global_env.write_text("LLM__OPENAI_API_KEY=sk-global\n")

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv("RELAY_ENV_FILE", raising=False)

    assert settings_module._find_env_file() == str(local_env)


def test_find_env_file_uses_global_config_when_local_missing(monkeypatch, tmp_path):
    _clear_llm_env(monkeypatch)

    cwd = tmp_path / "project"
    cwd.mkdir()

    config_home = tmp_path / "config"
    global_env = config_home / "relay" / ".env"
    global_env.parent.mkdir(parents=True)
    global_env.write_text("LLM__OPENAI_API_KEY=sk-global\n")

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv("RELAY_ENV_FILE", raising=False)

    assert settings_module._find_env_file() == str(global_env)


def test_get_settings_reads_global_config(monkeypatch, tmp_path):
    _clear_llm_env(monkeypatch)

    cwd = tmp_path / "project"
    cwd.mkdir()

    config_home = tmp_path / "config"
    global_env = config_home / "relay" / ".env"
    global_env.parent.mkdir(parents=True)
    global_env.write_text(
        "LLM__OPENAI_API_KEY=sk-global\n"
        "LLM__MODEL=gpt-5.1-codex-mini\n"
    )

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv("RELAY_ENV_FILE", raising=False)
    settings_module.get_settings.cache_clear()

    settings = settings_module.get_settings()

    assert settings.llm.openai_api_key.get_secret_value() == "sk-global"
    assert settings.llm.model == "gpt-5.1-codex-mini"

    settings_module.get_settings.cache_clear()


def test_relay_env_file_overrides_other_locations(monkeypatch, tmp_path):
    _clear_llm_env(monkeypatch)

    cwd = tmp_path / "project"
    cwd.mkdir()

    explicit_env = tmp_path / "relay.env"
    explicit_env.write_text("LLM__OPENAI_API_KEY=sk-explicit\n")

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("RELAY_ENV_FILE", str(explicit_env))

    assert settings_module._find_env_file() == str(explicit_env)
