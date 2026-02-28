"""Tests for settings API endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.settings_service import ENV_VAR_MAP


@pytest.fixture
def client(tmp_data_dir, monkeypatch):
    """Create a test client with isolated settings."""
    import app.services.settings_service as ss

    monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_data_dir / "settings.json")
    monkeypatch.setattr(ss, "ENV_PATH", tmp_data_dir / ".env")
    for env_var in ENV_VAR_MAP.values():
        monkeypatch.delenv(env_var, raising=False)
    return TestClient(app)


class TestGetSettings:
    def test_returns_defaults(self, client):
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["default_provider"] == "openrouter"
        assert data["default_model"] == "openai/gpt-4o-mini"

    def test_returns_masked_keys(self, client, tmp_data_dir):
        (tmp_data_dir / "settings.json").write_text(
            json.dumps({"openrouter_api_key": "sk-or-v1-abcdefghijklmnop"})
        )
        resp = client.get("/api/settings")
        data = resp.json()
        assert "..." in data["openrouter_api_key"]
        assert "sk-or-v1" not in data["openrouter_api_key"]


class TestUpdateSettings:
    def test_update_model(self, client):
        resp = client.put("/api/settings", json={"default_model": "gpt-4"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["default_model"] == "gpt-4"

    def test_update_persists(self, client):
        client.put("/api/settings", json={"default_provider": "openai"})
        resp = client.get("/api/settings")
        assert resp.json()["default_provider"] == "openai"

    def test_partial_update(self, client):
        client.put("/api/settings", json={"default_model": "model-a"})
        client.put("/api/settings", json={"default_provider": "openai"})
        resp = client.get("/api/settings")
        data = resp.json()
        assert data["default_model"] == "model-a"
        assert data["default_provider"] == "openai"


class TestHealthEndpoint:
    """Verify existing endpoints still work."""

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

