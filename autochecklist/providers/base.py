"""Base types and configuration for LLM providers."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol that all LLM providers must satisfy.

    Returns OpenAI-format dicts everywhere so existing parsing code
    works unchanged regardless of provider.
    """

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    def get_logprobs(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, float]: ...

    def supports_logprobs(self, model: str) -> bool: ...

    def batch_completions(
        self,
        requests: List[Dict[str, Any]],
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]: ...

    def close(self) -> None: ...

    def __enter__(self) -> "LLMClient": ...

    def __exit__(self, *args: Any) -> None: ...


@dataclass
class ProviderConfig:
    """Configuration preset for an LLM provider."""

    name: str
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    default_headers: Dict[str, str] = field(default_factory=dict)
    requires_api_key: bool = True


PROVIDER_PRESETS: Dict[str, ProviderConfig] = {
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
        default_headers={"HTTP-Referer": "https://github.com/ChicagoHAI/AutoChecklist"},
        requires_api_key=True,
    ),
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env_var="OPENAI_API_KEY",
        default_headers={},
        requires_api_key=True,
    ),
    "vllm": ProviderConfig(
        name="vllm",
        base_url=None,  # Configurable — not hardcoded
        api_key_env_var=None,
        default_headers={},
        requires_api_key=False,
    ),
}


def get_provider_config(
    provider: str,
    base_url: Optional[str] = None,
) -> ProviderConfig:
    """Get provider config with optional overrides.

    Args:
        provider: Provider name ("openrouter", "openai", "vllm")
        base_url: Override the default base URL

    Returns:
        ProviderConfig with overrides applied
    """
    if provider not in PROVIDER_PRESETS:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {', '.join(PROVIDER_PRESETS.keys())}"
        )

    preset = PROVIDER_PRESETS[provider]

    if base_url is not None:
        # Return a copy with overridden base_url
        return ProviderConfig(
            name=preset.name,
            base_url=base_url,
            api_key_env_var=preset.api_key_env_var,
            default_headers=preset.default_headers,
            requires_api_key=preset.requires_api_key,
        )

    return preset
