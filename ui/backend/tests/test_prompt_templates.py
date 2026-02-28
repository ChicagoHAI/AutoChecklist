"""Tests for the prompt templates CRUD router."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(tmp_data_dir, monkeypatch):
    """Test client with isolated data directory."""
    import app.routers.prompt_templates as pt_mod
    templates_dir = tmp_data_dir / "prompt_templates"
    templates_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(pt_mod, "TEMPLATES_DIR", templates_dir)
    return TestClient(app)


class TestCreatePromptTemplate:
    """POST /api/prompt-templates"""

    def test_create_basic(self, client):
        resp = client.post("/api/prompt-templates", json={
            "name": "My Eval",
            "prompt_text": "Evaluate the {target} given {input}.",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My Eval"
        assert data["prompt_text"] == "Evaluate the {target} given {input}."
        assert set(data["placeholders"]) == {"input", "target"}
        assert data["id"]
        assert data["created_at"]

    def test_create_with_reference_placeholder(self, client):
        resp = client.post("/api/prompt-templates", json={
            "name": "Ref Eval",
            "prompt_text": "Compare {target} to {reference} for {input}.",
            "description": "With reference",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert set(data["placeholders"]) == {"input", "target", "reference"}
        assert data["description"] == "With reference"

    def test_create_no_known_placeholders(self, client):
        resp = client.post("/api/prompt-templates", json={
            "name": "Plain",
            "prompt_text": "Just a plain prompt with {unknown}.",
        })
        assert resp.status_code == 201
        assert resp.json()["placeholders"] == []

    def test_create_validation_empty_name(self, client):
        resp = client.post("/api/prompt-templates", json={
            "name": "",
            "prompt_text": "Some prompt.",
        })
        assert resp.status_code == 422

    def test_create_validation_empty_prompt(self, client):
        resp = client.post("/api/prompt-templates", json={
            "name": "Name",
            "prompt_text": "",
        })
        assert resp.status_code == 422


class TestListPromptTemplates:
    """GET /api/prompt-templates"""

    def test_list_empty(self, client):
        resp = client.get("/api/prompt-templates")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_create(self, client):
        client.post("/api/prompt-templates", json={
            "name": "Template 1",
            "prompt_text": "Evaluate {target}.",
        })
        client.post("/api/prompt-templates", json={
            "name": "Template 2",
            "prompt_text": "Check {input} vs {target}.",
        })
        resp = client.get("/api/prompt-templates")
        assert resp.status_code == 200
        templates = resp.json()
        assert len(templates) == 2
        names = {t["name"] for t in templates}
        assert names == {"Template 1", "Template 2"}


class TestGetPromptTemplate:
    """GET /api/prompt-templates/{id}"""

    def test_get_existing(self, client):
        create_resp = client.post("/api/prompt-templates", json={
            "name": "My Template",
            "prompt_text": "Evaluate {input} → {target}.",
        })
        template_id = create_resp.json()["id"]

        resp = client.get(f"/api/prompt-templates/{template_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "My Template"
        assert data["prompt_text"] == "Evaluate {input} → {target}."

    def test_get_not_found(self, client):
        resp = client.get("/api/prompt-templates/nonexistent")
        assert resp.status_code == 404


class TestUpdatePromptTemplate:
    """PUT /api/prompt-templates/{id}"""

    def test_update_name(self, client):
        create_resp = client.post("/api/prompt-templates", json={
            "name": "Original",
            "prompt_text": "Prompt {input}.",
        })
        template_id = create_resp.json()["id"]

        resp = client.put(f"/api/prompt-templates/{template_id}", json={
            "name": "Updated",
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"
        assert resp.json()["prompt_text"] == "Prompt {input}."

    def test_update_prompt_text_refreshes_placeholders(self, client):
        create_resp = client.post("/api/prompt-templates", json={
            "name": "Test",
            "prompt_text": "Check {input}.",
        })
        template_id = create_resp.json()["id"]
        assert create_resp.json()["placeholders"] == ["input"]

        resp = client.put(f"/api/prompt-templates/{template_id}", json={
            "prompt_text": "Check {input} and {target} with {reference}.",
        })
        assert resp.status_code == 200
        assert set(resp.json()["placeholders"]) == {"input", "target", "reference"}

    def test_update_not_found(self, client):
        resp = client.put("/api/prompt-templates/nonexistent", json={"name": "X"})
        assert resp.status_code == 404


class TestDeletePromptTemplate:
    """DELETE /api/prompt-templates/{id}"""

    def test_delete_existing(self, client):
        create_resp = client.post("/api/prompt-templates", json={
            "name": "To Delete",
            "prompt_text": "Prompt.",
        })
        template_id = create_resp.json()["id"]

        resp = client.delete(f"/api/prompt-templates/{template_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone
        resp = client.get(f"/api/prompt-templates/{template_id}")
        assert resp.status_code == 404

    def test_delete_not_found(self, client):
        resp = client.delete("/api/prompt-templates/nonexistent")
        assert resp.status_code == 404
