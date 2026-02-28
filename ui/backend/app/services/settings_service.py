"""Settings management with 3-tier resolution: settings.json -> .env -> env vars."""

import os
from pathlib import Path
from typing import Optional

from .storage import DATA_DIR, read_json, write_json

SETTINGS_PATH = DATA_DIR / "settings.json"

# Root .env file: ui/backend/app/services -> app -> backend -> ui -> root
ENV_PATH = Path(__file__).parent.parent.parent.parent.parent / ".env"

DEFAULT_SETTINGS = {
    "default_provider": "openrouter",
    "default_model": "openai/gpt-4o-mini",
    "openrouter_api_key": "",
    "openai_api_key": "",
    "vllm_base_url": "",
}

# Map of settings keys to environment variable names
ENV_VAR_MAP = {
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "vllm_base_url": "VLLM_BASE_URL",
}


def _parse_env_file(path: Path) -> dict:
    """Parse a .env file into a dict."""
    result = {}
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            result[key.strip()] = value
    return result


def load_settings() -> dict:
    """Load settings with fallback chain: settings.json -> .env -> env vars."""
    settings = dict(DEFAULT_SETTINGS)

    # Layer 1: Load from settings.json
    saved = read_json(SETTINGS_PATH)
    if saved:
        settings.update({k: v for k, v in saved.items() if v})

    # Layer 2: Fill blanks from .env file
    env_file = _parse_env_file(ENV_PATH)
    for settings_key, env_var in ENV_VAR_MAP.items():
        if not settings.get(settings_key) and env_var in env_file:
            settings[settings_key] = env_file[env_var]

    # Layer 3: Fill blanks from environment variables
    for settings_key, env_var in ENV_VAR_MAP.items():
        if not settings.get(settings_key):
            settings[settings_key] = os.environ.get(env_var, "")

    return settings


def save_settings(updates: dict) -> dict:
    """Save settings updates to settings.json. Returns full settings."""
    current = load_settings()
    current.update({k: v for k, v in updates.items() if k in DEFAULT_SETTINGS})
    write_json(SETTINGS_PATH, current)
    return current


def get_masked_settings() -> dict:
    """Return settings with API keys masked."""
    settings = load_settings()
    for key in settings:
        if "api_key" in key and settings[key]:
            val = settings[key]
            settings[key] = val[:4] + "..." + val[-4:] if len(val) > 8 else "****"
    return settings


def get_provider_kwargs(
    provider_override: Optional[str] = None, model_override: Optional[str] = None
) -> dict:
    """Build kwargs compatible with ChecklistPipeline from settings.

    Returns keys that map directly to ChecklistPipeline.__init__:
    provider, api_key, base_url, and model.
    Callers that need generator_model/scorer_model should remap 'model' themselves.
    """
    s = load_settings()
    provider = provider_override or s.get("default_provider", "openrouter")
    model = model_override or s.get("default_model", "openai/gpt-4o-mini")

    kwargs: dict = {"provider": provider, "model": model}

    # Default reasoning_effort to "low" for reasoning models (cost/speed optimization)
    if _is_reasoning_model(model):
        kwargs["reasoning_effort"] = "low"

    if provider == "openrouter":
        api_key = s.get("openrouter_api_key")
        if api_key:
            kwargs["api_key"] = api_key
    elif provider == "openai":
        api_key = s.get("openai_api_key")
        if api_key:
            kwargs["api_key"] = api_key
    elif provider == "vllm":
        base_url = s.get("vllm_base_url")
        if base_url:
            kwargs["base_url"] = base_url

    return kwargs


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model that supports reasoning_effort."""
    # Normalize: strip provider prefix (e.g. "openai/gpt-5-mini" → "gpt-5-mini")
    name = model.rsplit("/", 1)[-1].lower()
    reasoning_patterns = ("o1", "o3", "o4", "gpt-5", "gpt5")
    return any(name.startswith(p) for p in reasoning_patterns)
