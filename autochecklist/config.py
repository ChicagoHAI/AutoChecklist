"""Configuration management for autochecklist."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    provider: str = "openrouter"
    model_id: str = "openai/gpt-4o-mini"
    api_key: Optional[str] = None  # Falls back to env var
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


class AutoChecklistsConfig(BaseModel):
    """Global configuration for autochecklist."""

    # Default models
    generator_model: ModelConfig = Field(default_factory=ModelConfig)
    scorer_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model_id="openai/gpt-4o-mini",
            temperature=0.0
        )
    )

    # API settings
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Behavior
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Custom prompts directory
    custom_prompts_dir: Optional[Path] = None


# Global configuration instance
_config: Optional[AutoChecklistsConfig] = None


def get_config() -> AutoChecklistsConfig:
    """Get current configuration."""
    global _config
    if _config is None:
        # Load .env file if present
        load_dotenv()

        _config = AutoChecklistsConfig()
        # Load from environment
        if api_key := os.getenv("OPENROUTER_API_KEY"):
            _config.openrouter_api_key = api_key
    return _config


def configure(**kwargs) -> None:
    """Update configuration.

    Example:
        configure(
            openrouter_api_key="sk-...",
            generator_model=ModelConfig(model_id="anthropic/claude-3-sonnet")
        )
    """
    global _config
    current = get_config()
    _config = current.model_copy(update=kwargs)
