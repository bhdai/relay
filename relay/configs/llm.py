"""Declarative LLM configuration schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class LLMProvider(str, Enum):
    """Supported chat model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class RateConfig(BaseModel):
    """Provider request-throttling settings."""

    requests_per_second: float = Field(
        description="The maximum number of requests per second",
    )
    check_every_n_seconds: float = Field(
        description="The interval in seconds to check the rate limit",
    )
    max_bucket_size: int = Field(
        description="The maximum number of requests that can be stored in the bucket",
    )


class LLMConfig(BaseModel):
    """Provider-aware LLM configuration."""

    provider: LLMProvider = Field(description="The provider of the LLM")
    model: str = Field(description="The provider model identifier")
    alias: str = Field(
        default="",
        description="Human-friendly alias used from agent configs and the CLI",
    )
    max_tokens: int = Field(
        default=10000,
        description="Maximum completion tokens for a single response",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature",
    )
    streaming: bool = Field(
        default=True,
        description="Whether to stream responses from the provider",
    )
    rate_config: RateConfig | None = Field(
        default=None,
        description="Optional per-model rate limit settings",
    )
    input_cost_per_mtok: float | None = Field(
        default=None,
        description="Input token cost per million tokens",
    )
    output_cost_per_mtok: float | None = Field(
        default=None,
        description="Output token cost per million tokens",
    )
    extended_reasoning: dict[str, Any] | None = Field(
        default=None,
        description="Provider-native reasoning configuration",
    )

    @model_validator(mode="after")
    def set_alias_default(self) -> LLMConfig:
        """Use the model identifier as the default alias."""
        if not self.alias:
            self.alias = self.model
        return self


class BatchLLMConfig(BaseModel):
    """Collection of named LLM configs."""

    llms: list[LLMConfig] = Field(
        default_factory=list,
        description="All loaded LLM configs",
    )

    @property
    def llm_names(self) -> list[str]:
        return [llm.alias for llm in self.llms]

    def get_llm(self, alias: str) -> LLMConfig | None:
        return next((llm for llm in self.llms if llm.alias == alias), None)