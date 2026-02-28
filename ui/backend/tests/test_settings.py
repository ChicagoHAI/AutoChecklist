"""Tests for settings service."""

import json

import pytest

from app.services.settings_service import (
    load_settings,
    save_settings,
    get_masked_settings,
    get_provider_kwargs,
    _parse_env_file,
    ENV_VAR_MAP,
)


@pytest.fixture
def settings_env(tmp_data_dir, monkeypatch):
    """Fixture that patches DATA_DIR, SETTINGS_PATH, and ENV_PATH for settings tests."""
    import app.services.settings_service as ss

    monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_data_dir / "settings.json")
    monkeypatch.setattr(ss, "ENV_PATH", tmp_data_dir / ".env")
    # Clear env vars to avoid leaking real keys
    for env_var in ENV_VAR_MAP.values():
        monkeypatch.delenv(env_var, raising=False)
    return tmp_data_dir


class TestParseEnvFile:
    def test_parses_simple_env(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("KEY=value\nOTHER=123\n")
        result = _parse_env_file(env)
        assert result == {"KEY": "value", "OTHER": "123"}

    def test_skips_comments_and_blanks(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("# comment\n\nKEY=val\n")
        result = _parse_env_file(env)
        assert result == {"KEY": "val"}

    def test_strips_quotes(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("A='single'\nB=\"double\"\n")
        result = _parse_env_file(env)
        assert result["A"] == "single"
        assert result["B"] == "double"

    def test_missing_file(self, tmp_path):
        result = _parse_env_file(tmp_path / "nope")
        assert result == {}

    def test_handles_equals_in_value(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("KEY=val=ue=more\n")
        result = _parse_env_file(env)
        assert result["KEY"] == "val=ue=more"


class TestLoadSettings:
    def test_defaults_when_nothing_exists(self, settings_env):
        result = load_settings()
        assert result["default_provider"] == "openrouter"
        assert result["default_model"] == "openai/gpt-4o-mini"
        assert result["openrouter_api_key"] == ""

    def test_settings_json_overrides_defaults(self, settings_env):
        (settings_env / "settings.json").write_text(
            json.dumps({"default_model": "custom-model", "openrouter_api_key": "sk-from-json"})
        )
        result = load_settings()
        assert result["default_model"] == "custom-model"
        assert result["openrouter_api_key"] == "sk-from-json"

    def test_env_file_fills_blanks(self, settings_env):
        (settings_env / ".env").write_text("OPENROUTER_API_KEY=sk-from-env\n")
        result = load_settings()
        assert result["openrouter_api_key"] == "sk-from-env"

    def test_settings_json_takes_priority_over_env_file(self, settings_env):
        (settings_env / "settings.json").write_text(
            json.dumps({"openrouter_api_key": "sk-from-json"})
        )
        (settings_env / ".env").write_text("OPENROUTER_API_KEY=sk-from-env\n")
        result = load_settings()
        assert result["openrouter_api_key"] == "sk-from-json"

    def test_env_var_fills_blanks(self, settings_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-envvar")
        result = load_settings()
        assert result["openai_api_key"] == "sk-from-envvar"

    def test_env_file_takes_priority_over_env_var(self, settings_env, monkeypatch):
        (settings_env / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-envvar")
        result = load_settings()
        assert result["openai_api_key"] == "sk-from-dotenv"

    def test_empty_values_in_json_do_not_override(self, settings_env):
        """Empty string values in settings.json should not count as 'set'."""
        (settings_env / "settings.json").write_text(
            json.dumps({"openrouter_api_key": ""})
        )
        (settings_env / ".env").write_text("OPENROUTER_API_KEY=sk-from-env\n")
        result = load_settings()
        assert result["openrouter_api_key"] == "sk-from-env"


class TestSaveSettings:
    def test_save_and_reload(self, settings_env):
        save_settings({"default_model": "new-model"})
        result = load_settings()
        assert result["default_model"] == "new-model"

    def test_ignores_unknown_keys(self, settings_env):
        save_settings({"unknown_key": "value", "default_model": "m"})
        data = json.loads((settings_env / "settings.json").read_text())
        assert "unknown_key" not in data
        assert data["default_model"] == "m"

    def test_preserves_existing_settings(self, settings_env):
        save_settings({"default_model": "model-a"})
        save_settings({"default_provider": "openai"})
        result = load_settings()
        assert result["default_model"] == "model-a"
        assert result["default_provider"] == "openai"


class TestGetMaskedSettings:
    def test_masks_api_keys(self, settings_env):
        (settings_env / "settings.json").write_text(
            json.dumps({"openrouter_api_key": "sk-or-v1-abcdefghijklmnop"})
        )
        result = get_masked_settings()
        assert result["openrouter_api_key"] == "sk-o...mnop"

    def test_short_key_fully_masked(self, settings_env):
        (settings_env / "settings.json").write_text(
            json.dumps({"openrouter_api_key": "short"})
        )
        result = get_masked_settings()
        assert result["openrouter_api_key"] == "****"

    def test_empty_key_not_masked(self, settings_env):
        result = get_masked_settings()
        assert result["openrouter_api_key"] == ""

    def test_non_key_fields_not_masked(self, settings_env):
        (settings_env / "settings.json").write_text(
            json.dumps({"default_model": "openai/gpt-4o-mini"})
        )
        result = get_masked_settings()
        assert result["default_model"] == "openai/gpt-4o-mini"


class TestGetProviderKwargs:
    def test_default_provider(self, settings_env):
        kwargs = get_provider_kwargs()
        assert kwargs["provider"] == "openrouter"
        assert kwargs["model"] == "openai/gpt-4o-mini"

    def test_override_provider(self, settings_env):
        kwargs = get_provider_kwargs(provider_override="openai")
        assert kwargs["provider"] == "openai"

    def test_override_model(self, settings_env):
        kwargs = get_provider_kwargs(model_override="gpt-4")
        assert kwargs["model"] == "gpt-4"

    def test_includes_api_key_for_openrouter(self, settings_env):
        (settings_env / "settings.json").write_text(
            json.dumps({"openrouter_api_key": "sk-test"})
        )
        kwargs = get_provider_kwargs()
        assert kwargs["api_key"] == "sk-test"

    def test_includes_api_key_for_openai(self, settings_env):
        (settings_env / "settings.json").write_text(
            json.dumps({"openai_api_key": "sk-openai"})
        )
        kwargs = get_provider_kwargs(provider_override="openai")
        assert kwargs["api_key"] == "sk-openai"

    def test_includes_base_url_for_vllm(self, settings_env):
        (settings_env / "settings.json").write_text(
            json.dumps({"vllm_base_url": "http://localhost:8000/v1"})
        )
        kwargs = get_provider_kwargs(provider_override="vllm")
        assert kwargs["base_url"] == "http://localhost:8000/v1"

    def test_no_api_key_when_empty(self, settings_env):
        kwargs = get_provider_kwargs()
        assert "api_key" not in kwargs
