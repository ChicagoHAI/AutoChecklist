"""Factory function for creating LLM clients."""

from typing import Any, Optional

from .base import LLMClient


def get_client(
    provider: str = "openrouter",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    api_format: Optional[str] = None,
    **kwargs: Any,
) -> LLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: Provider name ("openrouter", "openai", "vllm")
        base_url: Override the default base URL. For vLLM, None means offline mode.
        api_key: API key (resolved from env if not provided)
        model: Model name (required for vLLM offline mode)
        api_format: API format ("chat" or "responses"). Defaults to "responses"
            for OpenAI, "chat" for other providers.
        **kwargs: Additional kwargs passed to the client constructor

    Returns:
        An LLMClient instance

    Raises:
        ValueError: If provider is unknown or vLLM offline mode missing model
    """
    if provider == "vllm" and base_url is None:
        # Offline mode — direct Python inference
        from .vllm_offline import VLLMOfflineClient

        if model is None:
            raise ValueError(
                "model is required for vLLM offline mode. "
                "Pass model='your-model-name' or set base_url for server mode."
            )
        return VLLMOfflineClient(model=model, **kwargs)

    # HTTP mode (OpenRouter, OpenAI, vLLM server)
    from .http_client import LLMHTTPClient

    # Default to Responses API for OpenAI (handles reasoning models natively)
    if api_format is None:
        api_format = "responses" if provider == "openai" else "chat"

    return LLMHTTPClient(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        api_format=api_format,
        **kwargs,
    )
