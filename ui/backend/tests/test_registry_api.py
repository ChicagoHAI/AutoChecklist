"""Tests for registry API endpoints."""

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


class TestGenerators:
    def test_list_generators(self, client):
        resp = client.get("/api/registry/generators")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Check expected fields
        names = [g["name"] for g in data]
        assert "tick" in names
        # Each generator should have name, level, description
        for g in data:
            assert "name" in g
            assert "level" in g

class TestScorers:
    def test_list_scorers(self, client):
        resp = client.get("/api/registry/scorers")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        names = [s["name"] for s in data]
        assert "batch" in names


class TestDefaultScorers:
    def test_get_default_scorers(self, client):
        resp = client.get("/api/registry/default-scorers")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "tick" in data
        # Default scorers are now config dicts
        tick_cfg = data["tick"]
        assert isinstance(tick_cfg, dict)
        assert tick_cfg["mode"] == "batch"
        assert tick_cfg["primary_metric"] == "pass"
        assert "rlcf_direct" in data
        rlcf_cfg = data["rlcf_direct"]
        assert isinstance(rlcf_cfg, dict)
        assert rlcf_cfg["primary_metric"] == "weighted"
