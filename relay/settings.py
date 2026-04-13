from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM__")

    model: str = "gpt-4.1-mini"
    openai_api_key: SecretStr


class Settings(BaseSettings):
    # extra="ignore": the .env file contains LANGCHAIN_* tracing vars that
    # don't map to any field here — ignore them instead of raising.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
