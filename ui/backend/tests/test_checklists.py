"""Tests for checklists CRUD router."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(tmp_data_dir):
    """TestClient with patched DATA_DIR."""
    # Ensure the checklists subdirectory exists
    (tmp_data_dir / "checklists").mkdir(parents=True, exist_ok=True)

    # Also patch the CHECKLISTS_DIR in the router module
    import app.routers.checklists as checklists_mod
    original = checklists_mod.CHECKLISTS_DIR
    checklists_mod.CHECKLISTS_DIR = tmp_data_dir / "checklists"
    try:
        yield TestClient(app)
    finally:
        checklists_mod.CHECKLISTS_DIR = original


def _create_checklist(client, **overrides):
    """Helper to create a checklist via API."""
    payload = {
        "name": "Test Checklist",
        "method": "tick",
        "level": "instance",
        "model": "openai/gpt-4o-mini",
        "items": [
            {"question": "Is the response relevant?", "weight": 1.0},
            {"question": "Is the response clear?", "weight": 0.8},
        ],
        "metadata": {"tag": "test"},
    }
    payload.update(overrides)
    return client.post("/api/checklists", json=payload)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestListChecklists:
    def test_empty_list(self, client):
        resp = client.get("/api/checklists")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_summaries(self, client):
        _create_checklist(client, name="First")
        _create_checklist(client, name="Second")

        resp = client.get("/api/checklists")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        # Summaries should have these fields, not full items
        for summary in data:
            assert "id" in summary
            assert "name" in summary
            assert "method" in summary
            assert "level" in summary
            assert "item_count" in summary
            assert "created_at" in summary
            # Full items should NOT be present in summary
            assert "items" not in summary

    def test_list_item_count(self, client):
        _create_checklist(client, items=[
            {"question": "Q1", "weight": 1.0},
            {"question": "Q2", "weight": 1.0},
            {"question": "Q3", "weight": 1.0},
        ])
        resp = client.get("/api/checklists")
        assert resp.json()[0]["item_count"] == 3


class TestCreateChecklist:
    def test_create_generates_id(self, client):
        resp = _create_checklist(client)
        data = resp.json()
        assert "id" in data
        assert data["id"] == "test-checklist"  # slugified from "Test Checklist"

    def test_create_sets_timestamps(self, client):
        resp = _create_checklist(client)
        data = resp.json()
        assert "created_at" in data
        assert "updated_at" in data
        assert data["created_at"] == data["updated_at"]

    def test_create_stores_all_fields(self, client):
        resp = _create_checklist(
            client,
            name="My Checklist",
            method="rlcf",
            level="corpus",
            model="anthropic/claude-3.5-sonnet",
            items=[{"question": "Is X?", "weight": 0.5}],
            metadata={"source": "test"},
        )
        data = resp.json()
        assert data["name"] == "My Checklist"
        assert data["method"] == "rlcf"
        assert data["level"] == "corpus"
        assert data["model"] == "anthropic/claude-3.5-sonnet"
        assert len(data["items"]) == 1
        assert data["items"][0]["question"] == "Is X?"
        assert data["items"][0]["weight"] == 0.5
        assert data["metadata"] == {"source": "test"}

    def test_create_persists_to_disk(self, client, tmp_data_dir):
        resp = _create_checklist(client)
        cid = resp.json()["id"]
        path = tmp_data_dir / "checklists" / f"{cid}.json"
        assert path.exists()
        disk_data = json.loads(path.read_text())
        assert disk_data["id"] == cid

    def test_create_requires_name(self, client):
        resp = client.post("/api/checklists", json={
            "method": "tick",
            "items": [],
        })
        assert resp.status_code == 422


class TestGetChecklist:
    def test_get_existing(self, client):
        create_resp = _create_checklist(client)
        cid = create_resp.json()["id"]

        resp = client.get(f"/api/checklists/{cid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == cid
        assert data["name"] == "Test Checklist"
        assert len(data["items"]) == 2

    def test_get_missing_returns_404(self, client):
        resp = client.get("/api/checklists/nonexistent123")
        assert resp.status_code == 404


class TestUpdateChecklist:
    def test_update_name(self, client):
        cid = _create_checklist(client).json()["id"]

        resp = client.put(f"/api/checklists/{cid}", json={"name": "Updated Name"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"

    def test_update_items(self, client):
        cid = _create_checklist(client).json()["id"]

        new_items = [{"question": "New Q?", "weight": 2.0}]
        resp = client.put(f"/api/checklists/{cid}", json={"items": new_items})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["question"] == "New Q?"
        assert data["items"][0]["weight"] == 2.0

    def test_update_metadata(self, client):
        cid = _create_checklist(client).json()["id"]

        resp = client.put(f"/api/checklists/{cid}", json={"metadata": {"new_key": "value"}})
        assert resp.status_code == 200
        assert resp.json()["metadata"] == {"new_key": "value"}

    def test_update_changes_updated_at(self, client):
        create_data = _create_checklist(client).json()
        cid = create_data["id"]
        original_updated = create_data["updated_at"]

        resp = client.put(f"/api/checklists/{cid}", json={"name": "Changed"})
        assert resp.json()["updated_at"] >= original_updated

    def test_update_preserves_unchanged_fields(self, client):
        cid = _create_checklist(client, name="Original", method="tick").json()["id"]

        # Update only name
        resp = client.put(f"/api/checklists/{cid}", json={"name": "New Name"})
        data = resp.json()
        assert data["name"] == "New Name"
        assert data["method"] == "tick"  # Unchanged
        assert len(data["items"]) == 2   # Unchanged

    def test_update_missing_returns_404(self, client):
        resp = client.put("/api/checklists/nonexistent123", json={"name": "X"})
        assert resp.status_code == 404


class TestDeleteChecklist:
    def test_delete_existing(self, client, tmp_data_dir):
        cid = _create_checklist(client).json()["id"]
        path = tmp_data_dir / "checklists" / f"{cid}.json"
        assert path.exists()

        resp = client.delete(f"/api/checklists/{cid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert not path.exists()

    def test_delete_removes_from_list(self, client):
        cid = _create_checklist(client).json()["id"]
        client.delete(f"/api/checklists/{cid}")

        resp = client.get("/api/checklists")
        assert len(resp.json()) == 0

    def test_delete_missing_returns_404(self, client):
        resp = client.delete("/api/checklists/nonexistent123")
        assert resp.status_code == 404

    def test_get_after_delete_returns_404(self, client):
        cid = _create_checklist(client).json()["id"]
        client.delete(f"/api/checklists/{cid}")
        resp = client.get(f"/api/checklists/{cid}")
        assert resp.status_code == 404
