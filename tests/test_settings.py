from relay.settings import Settings


def test_settings_loads_from_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM__OPENAI_API_KEY=sk-test\n")
    s = Settings(_env_file=str(env_file))
    assert s.llm.openai_api_key.get_secret_value() == "sk-test"
