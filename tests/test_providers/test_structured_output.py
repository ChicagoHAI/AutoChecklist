"""Integration tests for structured output (response_format) across providers.

Tests verify that response_format is correctly passed through and that
the fallback mechanism works when a provider doesn't support it.
"""

import os
from unittest.mock import MagicMock

import pytest

from autochecklist.models import (
    ChecklistResponse,
    to_response_format,
)


# ── Fixtures ──

CHECKLIST_RF = to_response_format(ChecklistResponse, "checklist_response")

GENERATION_PROMPT = (
    "Generate a checklist of 2 yes/no questions to evaluate the quality "
    "of a response to: 'Write a haiku about snow.'\n\n"
    "Return JSON with a 'questions' array where each item has a 'question' field."
)


# ── Integration Tests (require API keys) ──


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
class TestOpenRouterStructuredOutput:
    """Test structured output via OpenRouter (Chat Completions)."""

    def test_checklist_response(self):
        from autochecklist.providers.factory import get_client

        client = get_client(provider="openrouter")
        response = client.chat_completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": GENERATION_PROMPT}],
            temperature=0.0,
            max_tokens=512,
            response_format=CHECKLIST_RF,
        )
        content = response["choices"][0]["message"]["content"]
        parsed = ChecklistResponse.model_validate_json(content)
        assert len(parsed.questions) >= 1


@pytest.mark.openai_api
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestOpenAIChatStructuredOutput:
    """Test structured output via OpenAI Chat Completions."""

    def test_checklist_response(self):
        from autochecklist.providers.factory import get_client

        client = get_client(provider="openai", model="gpt-4o-mini")
        response = client.chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": GENERATION_PROMPT}],
            temperature=0.0,
            max_tokens=512,
            response_format=CHECKLIST_RF,
        )
        content = response["choices"][0]["message"]["content"]
        parsed = ChecklistResponse.model_validate_json(content)
        assert len(parsed.questions) >= 1


@pytest.mark.openai_api
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestOpenAIResponsesStructuredOutput:
    """Test structured output via OpenAI Responses API."""

    def test_checklist_response(self):
        from autochecklist.providers.factory import get_client

        client = get_client(
            provider="openai",
            model="gpt-4o-mini",
            api_format="responses",
        )
        response = client.chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": GENERATION_PROMPT}],
            temperature=0.0,
            max_tokens=512,
            response_format=CHECKLIST_RF,
        )
        content = response["choices"][0]["message"]["content"]
        parsed = ChecklistResponse.model_validate_json(content)
        assert len(parsed.questions) >= 1


@pytest.mark.vllm_offline
class TestVLLMOfflineStructuredOutput:
    """Test structured output via vLLM guided decoding (offline only).

    Offline mode uses GuidedDecodingParams for token-level JSON enforcement.
    Server mode passes response_format through but small models may not
    produce valid structured output without guided decoding.
    """

    def test_checklist_response(self, vllm_offline_client):
        from autochecklist.providers.vllm_offline import VLLMOfflineClient
        if not isinstance(vllm_offline_client, VLLMOfflineClient):
            pytest.skip("structured output guided decoding only tested in offline mode")
        response = vllm_offline_client.chat_completion(
            model=vllm_offline_client._test_model,
            messages=[{"role": "user", "content": GENERATION_PROMPT}],
            temperature=0.0,
            max_tokens=1024,
            response_format=CHECKLIST_RF,
        )
        content = response["choices"][0]["message"]["content"]
        parsed = ChecklistResponse.model_validate_json(content)
        assert len(parsed.questions) >= 1


# ── Fallback Tests (no API keys needed) ──


class TestStructuredOutputFallback:
    """Test _call_model fallback when response_format is unsupported."""

    def test_generator_fallback_on_error(self):
        """When response_format causes an error, _call_model retries without it."""
        from autochecklist.generators.base import InstanceChecklistGenerator
        from autochecklist.models import Checklist

        # Concrete subclass for testing
        class DummyGenerator(InstanceChecklistGenerator):
            @property
            def method_name(self):
                return "dummy"

            def generate(self, input, **kwargs):
                return Checklist(
                    items=[], source_method="dummy", generation_level="instance"
                )

        mock_client = MagicMock()
        call_count = 0

        def mock_chat_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if "response_format" in kwargs:
                raise ValueError("response_format not supported")
            return {"choices": [{"message": {"content": "fallback response"}}]}

        mock_client.chat_completion = mock_chat_completion

        gen = DummyGenerator(client=mock_client)
        result = gen._call_model(
            "test prompt", response_format=CHECKLIST_RF
        )

        assert result == "fallback response"
        assert call_count == 2  # First try + fallback

    def test_generator_raises_without_response_format(self):
        """Error propagates normally when no response_format is provided."""
        from autochecklist.generators.base import InstanceChecklistGenerator
        from autochecklist.models import Checklist

        class DummyGenerator(InstanceChecklistGenerator):
            @property
            def method_name(self):
                return "dummy"

            def generate(self, input, **kwargs):
                return Checklist(
                    items=[], source_method="dummy", generation_level="instance"
                )

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = ValueError("some error")

        gen = DummyGenerator(client=mock_client)
        with pytest.raises(ValueError, match="some error"):
            gen._call_model("test prompt")

    def test_scorer_fallback_on_error(self):
        """Same fallback behavior works for scorers."""
        from autochecklist.scorers.base import ChecklistScorer
        from autochecklist.models import Score

        class DummyScorer(ChecklistScorer):
            @property
            def scoring_method(self):
                return "dummy"

            def score(self, checklist, target, input=None, **kwargs):
                return Score(
                    checklist_id="test",
                    item_scores=[],
                    total_score=0.0,
                    judge_model="test",
                    scoring_method="dummy",
                )

        mock_client = MagicMock()
        call_count = 0

        def mock_chat_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if "response_format" in kwargs:
                raise ValueError("response_format not supported")
            return {"choices": [{"message": {"content": "fallback"}}]}

        mock_client.chat_completion = mock_chat_completion

        scorer = DummyScorer(client=mock_client)
        result = scorer._call_model(
            "test prompt", response_format=CHECKLIST_RF
        )

        assert result == "fallback"
        assert call_count == 2
