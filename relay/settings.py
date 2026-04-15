from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    model: str = "gpt-4.1-mini"
    openai_api_key: SecretStr
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0


class Settings(BaseSettings):
    # extra="ignore": the .env file contains LANGCHAIN_* tracing vars that
    # don't map to any field here — ignore them instead of raising.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings
    rate_limit: RateLimitSettings = RateLimitSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
