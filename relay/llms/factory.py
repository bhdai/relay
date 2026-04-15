"""Provider-aware chat model factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.rate_limiters import InMemoryRateLimiter

from relay.configs.llm import LLMConfig, LLMProvider
from relay.settings import Settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


class LLMFactory:
    """Instantiate chat models from declarative LLM configs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[tuple, BaseChatModel] = {}

    @staticmethod
    def _cache_key(config: LLMConfig) -> tuple:
        reasoning = config.extended_reasoning or {}
        return (
            config.provider,
            config.model,
            config.alias,
            config.max_tokens,
            config.temperature,
            config.streaming,
            tuple(sorted(reasoning.items())),
            (
                None
                if config.rate_config is None
                else (
                    config.rate_config.requests_per_second,
                    config.rate_config.check_every_n_seconds,
                    config.rate_config.max_bucket_size,
                )
            ),
        )

    @staticmethod
    def _create_rate_limiter(config: LLMConfig) -> InMemoryRateLimiter | None:
        if config.rate_config is None:
            return None

        return InMemoryRateLimiter(
            requests_per_second=config.rate_config.requests_per_second,
            check_every_n_seconds=config.rate_config.check_every_n_seconds,
            max_bucket_size=config.rate_config.max_bucket_size,
        )

    def create(self, config: LLMConfig) -> BaseChatModel:
        cache_key = self._cache_key(config)
        if cache_key in self._cache:
            return self._cache[cache_key]

        rate_limiter = self._create_rate_limiter(config)

        if config.provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            api_key = self._settings.llm.openai_api_key
            if api_key is None:
                raise ValueError(
                    "OpenAI model selected but LLM__OPENAI_API_KEY is not configured"
                )

            kwargs = {
                "model": config.model,
                "api_key": api_key.get_secret_value(),
                "max_completion_tokens": config.max_tokens,
                "temperature": config.temperature,
                "streaming": config.streaming,
                "rate_limiter": rate_limiter,
            }
            if config.extended_reasoning:
                kwargs["reasoning"] = config.extended_reasoning
                kwargs["output_version"] = "responses/v1"

            model = ChatOpenAI(**kwargs)
        elif config.provider == LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            api_key = self._settings.llm.anthropic_api_key
            if api_key is None:
                raise ValueError(
                    "Anthropic model selected but LLM__ANTHROPIC_API_KEY is not configured"
                )

            kwargs = {
                "model": config.model,
                "anthropic_api_key": api_key.get_secret_value(),
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "streaming": config.streaming,
                "rate_limiter": rate_limiter,
            }
            if config.extended_reasoning:
                kwargs["thinking"] = config.extended_reasoning

            model = ChatAnthropic(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        self._cache[cache_key] = model
        return model