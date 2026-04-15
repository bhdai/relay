"""Tests for provider-aware LLM construction."""

from __future__ import annotations

from unittest.mock import MagicMock

from relay.configs.llm import LLMConfig
from relay.llms.factory import LLMFactory


class TestLLMFactory:
    def test_create_openai_model_with_reasoning(self, monkeypatch):
        captured = {}

        class FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

        settings = MagicMock()
        settings.llm.openai_api_key.get_secret_value.return_value = "sk-test"

        factory = LLMFactory(settings)
        config = LLMConfig(
            provider="openai",
            model="gpt-5-mini-2025-08-07",
            alias="gpt-5-mini-thinking",
            extended_reasoning={"effort": "medium", "summary": "auto"},
        )

        factory.create(config)

        assert captured["model"] == "gpt-5-mini-2025-08-07"
        assert captured["reasoning"] == {"effort": "medium", "summary": "auto"}
        assert captured["output_version"] == "responses/v1"

    def test_create_anthropic_model_with_thinking(self, monkeypatch):
        captured = {}

        class FakeChatAnthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("langchain_anthropic.ChatAnthropic", FakeChatAnthropic)

        settings = MagicMock()
        settings.llm.anthropic_api_key.get_secret_value.return_value = "anthropic-test"

        factory = LLMFactory(settings)
        config = LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5",
            alias="haiku-4.5-thinking",
            extended_reasoning={"type": "enabled", "budget_tokens": 2000},
        )

        factory.create(config)

        assert captured["model"] == "claude-haiku-4-5"
        assert captured["thinking"] == {"type": "enabled", "budget_tokens": 2000}