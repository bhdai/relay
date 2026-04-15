import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> str | None:
    """Find the env file for the current relay invocation.

    Precedence:
    1. ``RELAY_ENV_FILE`` when explicitly set.
    2. ``.env`` in the current working directory.
    3. ``$XDG_CONFIG_HOME/relay/.env``.
    4. ``~/.config/relay/.env``.
    5. ``~/.relay/.env``.
    """
    if explicit := os.environ.get("RELAY_ENV_FILE"):
        candidate = Path(explicit).expanduser()
        return str(candidate) if candidate.is_file() else None

    candidates = [
        Path.cwd() / ".env",
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "relay"
        / ".env",
        Path.home() / ".relay" / ".env",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None


def load_environment() -> str | None:
    """Load relay environment variables from the resolved env file, if any."""
    env_file = _find_env_file()
    if env_file is not None:
        load_dotenv(env_file)
    return env_file


class RateLimitSettings(BaseSettings):
    """Token-bucket rate limiting for LLM API calls.

    ``requests_per_second`` is the refill rate of the bucket.
    ``max_bucket_size`` is the burst capacity.  The limiter blocks
    until a token is available, preventing 429 errors from the
    provider.

    Defaults are conservative: ~5 requests/s with a burst of 10.
    These work for most OpenAI tiers without hitting per-minute
    token limits when subagents are active.
    """

    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT__")

    requests_per_second: float = 5.0
    check_every_n_seconds: float = 0.1
    max_bucket_size: int = 10


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM__")

    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    max_tokens: int = 10000
    temperature: float = 0.1
    streaming: bool = True
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0


class Settings(BaseSettings):
    # extra="ignore": the .env file contains LANGCHAIN_* tracing vars that
    # don't map to any field here — ignore them instead of raising.
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings
    rate_limit: RateLimitSettings = RateLimitSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env_file = _find_env_file()
    return Settings(_env_file=env_file)  # type: ignore[call-arg]
