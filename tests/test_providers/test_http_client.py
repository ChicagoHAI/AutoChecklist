"""Tests for the provider-agnostic HTTP client."""
import os

import pytest


class TestHTTPClientConstruction:
    """Test LLMHTTPClient initialization with different providers."""

    def test_openrouter_sets_correct_base_url(self):
        """OpenRouter client should use OpenRouter's API URL."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openrouter", api_key="test-key")
        assert "openrouter.ai" in client.base_url
        client.close()

    def test_openai_sets_correct_base_url(self):
        """OpenAI client should use OpenAI's API URL."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai", api_key="test-key")
        assert "api.openai.com" in client.base_url
        client.close()

    def test_vllm_server_uses_provided_base_url(self):
        """vLLM server mode should use the explicitly provided URL."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="vllm", base_url="http://myserver:9000/v1")
        assert client.base_url == "http://myserver:9000/v1"
        client.close()

    def test_custom_base_url_overrides_preset(self):
        """Explicit base_url should override preset for any provider."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai", api_key="test", base_url="http://custom/v1")
        assert client.base_url == "http://custom/v1"
        client.close()


class TestAPIKeyResolution:
    """Test API key resolution priority: explicit > env var > error."""

    def test_explicit_api_key_takes_priority(self, monkeypatch):
        """Explicitly passed api_key should be used even if env var exists."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openrouter", api_key="explicit-key")
        assert client.api_key == "explicit-key"
        client.close()

    def test_env_var_fallback_for_openrouter(self, monkeypatch):
        """Should fall back to OPENROUTER_API_KEY env var."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        # Reset config singleton to pick up new env
        from autochecklist import config as config_mod
        config_mod._config = None
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openrouter")
        assert client.api_key == "env-key"
        client.close()
        config_mod._config = None  # cleanup

    def test_env_var_fallback_for_openai(self, monkeypatch):
        """Should fall back to OPENAI_API_KEY env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai")
        assert client.api_key == "openai-env-key"
        client.close()

    def test_missing_api_key_raises_for_openrouter(self, monkeypatch):
        """OpenRouter should raise ValueError when no API key is available."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Prevent load_dotenv from re-populating the env var from .env file
        monkeypatch.setattr("autochecklist.config.load_dotenv", lambda: None)
        from autochecklist import config as config_mod
        config_mod._config = None
        from autochecklist.providers.http_client import LLMHTTPClient
        with pytest.raises(ValueError, match="API key"):
            LLMHTTPClient(provider="openrouter")
        config_mod._config = None

    def test_missing_api_key_raises_for_openai(self, monkeypatch):
        """OpenAI should raise ValueError when no API key is available."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from autochecklist.providers.http_client import LLMHTTPClient
        with pytest.raises(ValueError, match="API key"):
            LLMHTTPClient(provider="openai")

    def test_vllm_server_allows_no_api_key(self):
        """vLLM server mode should not require an API key."""
        from autochecklist.providers.http_client import LLMHTTPClient
        # Should not raise
        client = LLMHTTPClient(provider="vllm", base_url="http://localhost:8000/v1")
        client.close()


class TestLogprobsSupport:
    """Test supports_logprobs() behavior per provider."""

    def test_vllm_always_supports_logprobs(self):
        """vLLM should always report logprobs support."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="vllm", base_url="http://localhost:8000/v1")
        assert client.supports_logprobs("any-model") is True
        client.close()

    def test_openai_always_supports_logprobs(self):
        """OpenAI should always report logprobs support."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai", api_key="test-key")
        assert client.supports_logprobs("gpt-4o-mini") is True
        client.close()


class TestModelNameNormalization:
    """Test _normalize_model_name() strips provider prefixes correctly."""

    def test_openai_provider_strips_openai_prefix(self):
        """OpenAI provider should strip 'openai/' prefix."""
        from autochecklist.providers.http_client import _normalize_model_name
        assert _normalize_model_name("openai/gpt-5-mini", "openai") == "gpt-5-mini"

    def test_openai_provider_strips_any_prefix(self):
        """OpenAI provider should strip any org prefix."""
        from autochecklist.providers.http_client import _normalize_model_name
        assert _normalize_model_name("anthropic/claude-sonnet-4", "openai") == "claude-sonnet-4"

    def test_openai_provider_keeps_bare_name(self):
        """OpenAI provider should not modify names without prefix."""
        from autochecklist.providers.http_client import _normalize_model_name
        assert _normalize_model_name("gpt-5-mini", "openai") == "gpt-5-mini"

    def test_openrouter_provider_keeps_prefix(self):
        """OpenRouter provider should keep the org/model prefix."""
        from autochecklist.providers.http_client import _normalize_model_name
        assert _normalize_model_name("openai/gpt-5-mini", "openrouter") == "openai/gpt-5-mini"

    def test_vllm_provider_keeps_prefix(self):
        """vLLM provider should keep HuggingFace-style org/model names."""
        from autochecklist.providers.http_client import _normalize_model_name
        assert _normalize_model_name("meta-llama/Llama-3-8B", "vllm") == "meta-llama/Llama-3-8B"


class TestReasoningModelRequestShape:
    """Reasoning models require different request body fields."""

    def test_reasoning_model_uses_max_completion_tokens_without_temperature(self):
        from autochecklist.providers.http_client import LLMHTTPClient

        client = LLMHTTPClient(provider="openai", api_key="test-key")
        try:
            # Capture outgoing payload without making a network call.
            captured = {}

            class FakeResponse:
                is_success = True

                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                        "model": "gpt-5-mini",
                    }

            def fake_post(path, json):
                captured["path"] = path
                captured["json"] = json
                return FakeResponse()

            client._client.post = fake_post

            client.chat_completion(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.1,
                max_tokens=123,
            )

            assert captured["path"] == "/chat/completions"
            assert "max_completion_tokens" in captured["json"]
            assert captured["json"]["max_completion_tokens"] == 123
            assert "max_tokens" not in captured["json"]
            assert "temperature" not in captured["json"]
        finally:
            client.close()

    def test_non_reasoning_model_uses_max_tokens_with_temperature(self):
        from autochecklist.providers.http_client import LLMHTTPClient

        client = LLMHTTPClient(provider="openai", api_key="test-key")
        try:
            captured = {}

            class FakeResponse:
                is_success = True

                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                        "model": "gpt-4o-mini",
                    }

            def fake_post(path, json):
                captured["path"] = path
                captured["json"] = json
                return FakeResponse()

            client._client.post = fake_post

            client.chat_completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.3,
                max_tokens=77,
            )

            assert captured["path"] == "/chat/completions"
            assert "max_tokens" in captured["json"]
            assert captured["json"]["max_tokens"] == 77
            assert captured["json"]["temperature"] == 0.3
            assert "max_completion_tokens" not in captured["json"]
        finally:
            client.close()


class TestOpenAIIntegration:
    """Integration tests for OpenAI provider with real API calls.

    Skipped when OPENAI_API_KEY is not set.
    """

    pytestmark = pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )

    def test_chat_completion_with_bare_model_name(self):
        """Basic chat completion with a standard OpenAI model name."""
        from autochecklist.providers.http_client import LLMHTTPClient
        with LLMHTTPClient(provider="openai") as client:
            response = client.chat_completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello in one word"}],
                temperature=0.0,
                max_tokens=16,
            )
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    def test_chat_completion_with_prefixed_model_name(self):
        """OpenRouter-style 'openai/model' name should be normalized for OpenAI API."""
        from autochecklist.providers.http_client import LLMHTTPClient
        with LLMHTTPClient(provider="openai") as client:
            response = client.chat_completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello in one word"}],
                temperature=0.0,
                max_tokens=16,
            )
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_async_chat_completion_with_prefixed_model_name(self):
        """Async path should also normalize OpenRouter-style model names."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai")
        response = await client.chat_completion_async(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello in one word"}],
            temperature=0.0,
            max_tokens=16,
        )
        client.close()
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0


class TestOpenRouterIntegration:
    """Integration tests for OpenRouter provider with real API calls.

    Skipped when OPENROUTER_API_KEY is not set.
    """

    pytestmark = pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set",
    )

    def test_chat_completion_basic(self):
        """Basic chat completion through OpenRouter."""
        from autochecklist.providers.http_client import LLMHTTPClient
        with LLMHTTPClient(provider="openrouter") as client:
            response = client.chat_completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello in one word"}],
                temperature=0.0,
                max_tokens=16,
            )
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0
