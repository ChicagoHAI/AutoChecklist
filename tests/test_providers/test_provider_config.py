"""Tests for provider configuration and factory."""
import pytest


class TestProviderConfig:
    """Test ProviderConfig presets and resolution."""

    def test_openrouter_preset_has_correct_env_var(self):
        """OpenRouter preset should resolve API key from OPENROUTER_API_KEY."""
        from autochecklist.providers.base import PROVIDER_PRESETS
        config = PROVIDER_PRESETS["openrouter"]
        assert config.api_key_env_var == "OPENROUTER_API_KEY"
        assert config.requires_api_key is True

    def test_openai_preset_has_correct_env_var(self):
        """OpenAI preset should resolve API key from OPENAI_API_KEY."""
        from autochecklist.providers.base import PROVIDER_PRESETS
        config = PROVIDER_PRESETS["openai"]
        assert config.api_key_env_var == "OPENAI_API_KEY"
        assert config.requires_api_key is True

    def test_vllm_preset_does_not_require_api_key(self):
        """vLLM preset should not require an API key."""
        from autochecklist.providers.base import PROVIDER_PRESETS
        config = PROVIDER_PRESETS["vllm"]
        assert config.requires_api_key is False

    def test_vllm_preset_has_no_default_base_url(self):
        """vLLM preset should NOT have a hardcoded base URL — it's configurable."""
        from autochecklist.providers.base import PROVIDER_PRESETS
        config = PROVIDER_PRESETS["vllm"]
        assert config.base_url is None

    def test_get_provider_config_uses_custom_base_url(self):
        """Custom base_url should override the preset default."""
        from autochecklist.providers.base import get_provider_config
        config = get_provider_config("vllm", base_url="http://custom:9000/v1")
        assert config.base_url == "http://custom:9000/v1"

    def test_get_provider_config_preserves_preset_when_no_override(self):
        """Without overrides, should return the preset values."""
        from autochecklist.providers.base import get_provider_config
        config = get_provider_config("openrouter")
        assert "openrouter.ai" in config.base_url
        assert config.requires_api_key is True

    def test_openrouter_preset_has_referer_header(self):
        """OpenRouter needs HTTP-Referer header for their API."""
        from autochecklist.providers.base import PROVIDER_PRESETS
        config = PROVIDER_PRESETS["openrouter"]
        assert "HTTP-Referer" in config.default_headers

    def test_openai_preset_has_no_referer_header(self):
        """OpenAI should NOT have HTTP-Referer header."""
        from autochecklist.providers.base import PROVIDER_PRESETS
        config = PROVIDER_PRESETS["openai"]
        assert "HTTP-Referer" not in config.default_headers


class TestClientFactory:
    """Test the get_client() factory function."""

    def test_unknown_provider_raises_error(self):
        """Factory should raise ValueError for unknown providers."""
        from autochecklist.providers.factory import get_client
        with pytest.raises(ValueError, match="Unknown provider"):
            get_client(provider="nonexistent")

    def test_vllm_offline_without_model_raises_error(self):
        """vLLM offline mode requires a model name to load weights."""
        from autochecklist.providers.factory import get_client
        with pytest.raises(ValueError, match="model"):
            get_client(provider="vllm")  # No base_url = offline, but no model

    def test_vllm_with_base_url_returns_http_client(self):
        """vLLM with base_url should return HTTP client (server mode)."""
        from autochecklist.providers.factory import get_client
        from autochecklist.providers.http_client import LLMHTTPClient
        client = get_client(provider="vllm", base_url="http://localhost:8000/v1")
        assert isinstance(client, LLMHTTPClient)
        client.close()

    def test_openrouter_returns_http_client(self):
        """OpenRouter should return HTTP client."""
        from autochecklist.providers.factory import get_client
        from autochecklist.providers.http_client import LLMHTTPClient
        # Provide api_key to avoid env var lookup failure
        client = get_client(provider="openrouter", api_key="test-key")
        assert isinstance(client, LLMHTTPClient)
        client.close()

    def test_openai_returns_http_client(self):
        """OpenAI should return HTTP client."""
        from autochecklist.providers.factory import get_client
        from autochecklist.providers.http_client import LLMHTTPClient
        client = get_client(provider="openai", api_key="test-key")
        assert isinstance(client, LLMHTTPClient)
        client.close()
