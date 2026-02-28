"""Tests for batch evaluation backend (router + service)."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.settings_service import ENV_VAR_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_data(n=3):
    """Create sample batch data items."""
    return [
        {"input": f"Instruction {i}", "target": f"Response {i}"}
        for i in range(n)
    ]


def _make_mock_pipeline_result():
    """Create a mock PipelineResult for a single pipe() call."""
    from autochecklist.models import (
        Checklist,
        ChecklistItem,
        ChecklistItemAnswer,
        ItemScore,
        Score,
    )

    items = [
        ChecklistItem(question=f"Is criterion {j} met?")
        for j in range(2)
    ]
    checklist = Checklist(
        items=items,
        source_method="tick",
        generation_level="instance",
    )
    item_scores = [
        ItemScore(item_id=item.id, answer=ChecklistItemAnswer.YES, reasoning="Good")
        for item in items
    ]
    score = Score(
        checklist_id="cl_0",
        item_scores=item_scores,
        total_score=1.0,
        judge_model="test-model",
        scoring_method="binary",
    )

    mock_result = MagicMock()
    mock_result.score = score
    mock_result.checklist = checklist
    return mock_result


def _make_csv_content(data):
    """Create CSV bytes from data dicts."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(data[0].keys()))
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue().encode("utf-8")


def _make_jsonl_content(data):
    """Create JSONL bytes from data dicts."""
    lines = [json.dumps(item) for item in data]
    return "\n".join(lines).encode("utf-8")


def _make_json_content(data):
    """Create JSON array bytes from data dicts."""
    return json.dumps(data).encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_data_dir, monkeypatch):
    """Create a test client with isolated data dir and settings."""
    import app.services.settings_service as ss
    import app.services.batch_runner as br

    monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_data_dir / "settings.json")
    monkeypatch.setattr(ss, "ENV_PATH", tmp_data_dir / ".env")
    monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
    for env_var in ENV_VAR_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    # Ensure subdirectories exist
    (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

    return TestClient(app)


@pytest.fixture
def sample_data():
    return _make_sample_data(3)


# ---------------------------------------------------------------------------
# Tests: Upload endpoint
# ---------------------------------------------------------------------------

class TestUploadEndpoint:
    """Tests for POST /api/batch/upload."""

    @pytest.mark.parametrize(
        "filename,mime,content_builder",
        [
            ("test.csv", "text/csv", _make_csv_content),
            ("test.jsonl", "application/jsonl", _make_jsonl_content),
            ("test.json", "application/json", _make_json_content),
        ],
    )
    def test_upload_supported_formats(self, client, sample_data, filename, mime, content_builder):
        content = content_builder(sample_data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": (filename, BytesIO(content), mime)},
            data={"method": "tick"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_items"] == len(sample_data)
        assert body["filename"] == filename
        assert "batch_id" in body

    def test_upload_rejects_unsupported_format(self, client):
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.txt", BytesIO(b"hello"), "text/plain")},
            data={"method": "tick"},
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_upload_rejects_missing_columns(self, client):
        data = [{"question": "Q1", "answer": "A1"}]
        content = _make_json_content(data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        assert resp.status_code == 400
        assert "input" in resp.json()["detail"]

    def test_upload_rejects_empty_data(self, client):
        content = _make_json_content([])
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        assert resp.status_code == 400
        assert "no data" in resp.json()["detail"]

    def test_upload_accepts_large_datasets(self, client):
        """No artificial row limit — this is a local tool."""
        data = _make_sample_data(1001)
        content = _make_json_content(data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        assert resp.status_code == 200
        assert resp.json()["total_items"] == 1001

    def test_upload_rejects_non_array_json(self, client):
        content = json.dumps({"key": "value"}).encode("utf-8")
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        assert resp.status_code == 400
        assert "array" in resp.json()["detail"]

    def test_upload_creates_directory_structure(self, client, tmp_data_dir, sample_data):
        content = _make_json_content(sample_data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        batch_id = resp.json()["batch_id"]
        batch_dir = tmp_data_dir / "batches" / batch_id

        assert batch_dir.exists()
        assert (batch_dir / "config.json").exists()
        assert (batch_dir / "input.jsonl").exists()
        assert (batch_dir / "summary.json").exists()

        config = json.loads((batch_dir / "config.json").read_text())
        assert config["method"] == "tick"
        assert config["total_items"] == 3

        summary = json.loads((batch_dir / "summary.json").read_text())
        assert summary["status"] == "pending"
        assert summary["total"] == 3
        assert summary["completed"] == 0


# ---------------------------------------------------------------------------
# Tests: Batch service layer (create, status, list)
# ---------------------------------------------------------------------------

class TestBatchService:
    """Tests for batch_runner service functions."""

    def test_create_batch(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
        (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

        data = _make_sample_data(2)
        batch_id = br.create_batch(
            method="tick", scorer=None, provider_kwargs={"model": "test"},
            filename="test.jsonl", data=data,
        )

        assert batch_id is not None
        batch_dir = tmp_data_dir / "batches" / batch_id
        assert batch_dir.exists()

        config = json.loads((batch_dir / "config.json").read_text())
        assert config["method"] == "tick"
        assert config["model"] == "test"  # provider_kwargs included (minus api_key)
        assert config["total_items"] == 2

    def test_create_batch_excludes_api_key(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
        (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

        data = _make_sample_data(1)
        batch_id = br.create_batch(
            method="tick", scorer=None,
            provider_kwargs={"model": "test", "api_key": "secret-key"},
            filename="test.jsonl", data=data,
        )

        config = json.loads((tmp_data_dir / "batches" / batch_id / "config.json").read_text())
        assert "api_key" not in config

    def test_get_batch_status(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
        (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

        data = _make_sample_data(2)
        batch_id = br.create_batch(
            method="tick", scorer=None, provider_kwargs={},
            filename="test.jsonl", data=data,
        )

        status = br.get_batch_status(batch_id)
        assert status is not None
        assert status["batch_id"] == batch_id
        assert status["status"] == "pending"
        assert status["total"] == 2
        assert "config" in status
        assert status["config"]["method"] == "tick"

    def test_get_batch_status_not_found(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)

        status = br.get_batch_status("nonexistent")
        assert status is None

    def test_list_batches(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
        (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

        data = _make_sample_data(1)
        id1 = br.create_batch(method="tick", scorer=None, provider_kwargs={}, filename="a.jsonl", data=data)
        id2 = br.create_batch(method="rocketeval", scorer=None, provider_kwargs={}, filename="b.jsonl", data=data)

        # Set explicit mtimes on batch directories to guarantee ordering (id2 newer)
        import os
        os.utime(tmp_data_dir / "batches" / id1, (1000, 1000))
        os.utime(tmp_data_dir / "batches" / id2, (2000, 2000))

        batches = br.list_batches()
        assert len(batches) == 2
        # Most recent first
        assert batches[0]["batch_id"] == id2
        assert batches[1]["batch_id"] == id1

    def test_list_batches_empty(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)

        batches = br.list_batches()
        assert batches == []

    def test_get_batch_results_empty(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)

        results = br.get_batch_results("nonexistent")
        assert results == []

    def test_get_batch_results_with_data(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
        (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

        data = _make_sample_data(1)
        batch_id = br.create_batch(method="tick", scorer=None, provider_kwargs={}, filename="test.jsonl", data=data)

        # Write fake results
        results_path = tmp_data_dir / "batches" / batch_id / "results.jsonl"
        records = [
            {"index": 0, "input": "I0", "target": "R0", "pass_rate": 1.0, "total_score": 1.0},
            {"index": 1, "input": "I1", "target": "R1", "pass_rate": 0.5, "total_score": 0.5},
            {"index": 2, "input": "I2", "target": "R2", "pass_rate": 0.0, "total_score": 0.0},
        ]
        with open(results_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # All results
        results = br.get_batch_results(batch_id)
        assert len(results) == 3

        # With offset and limit
        results = br.get_batch_results(batch_id, offset=1, limit=1)
        assert len(results) == 1
        assert results[0]["index"] == 1

    def test_cancel_batch_not_running(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)

        result = br.cancel_batch("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Start batch (with mocked pipeline)
# ---------------------------------------------------------------------------

class TestStartBatch:
    """Tests for starting batch jobs with mocked pipeline."""

    def test_start_batch_runs_pipeline(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
        (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

        data = _make_sample_data(2)
        batch_id = br.create_batch(method="tick", scorer=None, provider_kwargs={}, filename="test.jsonl", data=data)

        mock_pipeline_result = _make_mock_pipeline_result()

        with patch("app.services.batch_runner.ChecklistPipeline") as MockPipeline:
            mock_pipe = MagicMock()
            mock_pipe.return_value = mock_pipeline_result  # pipe() per-item call
            MockPipeline.return_value = mock_pipe

            success = br.start_batch(batch_id, {"provider": "openrouter", "model": "openai/gpt-4o-mini"})
            assert success is True

            # Wait for thread to finish
            if batch_id in br._running_jobs:
                br._running_jobs[batch_id].join(timeout=5)

        # Check summary updated
        summary = br.get_batch_status(batch_id)
        assert summary["status"] == "completed"
        assert summary["completed"] == 2
        assert summary["macro_pass_rate"] == 1.0

        # Check results written
        results = br.get_batch_results(batch_id)
        assert len(results) == 2
        assert results[0]["pass_rate"] == 1.0

    def test_start_batch_handles_pipeline_error(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
        (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

        data = _make_sample_data(1)
        batch_id = br.create_batch(method="tick", scorer=None, provider_kwargs={}, filename="test.jsonl", data=data)

        with patch("app.services.batch_runner.ChecklistPipeline") as MockPipeline:
            MockPipeline.side_effect = ValueError("Bad config")

            success = br.start_batch(batch_id, {"provider": "openrouter", "model": "test"})
            assert success is True

            # Wait for thread
            if batch_id in br._running_jobs:
                br._running_jobs[batch_id].join(timeout=5)

        summary = br.get_batch_status(batch_id)
        assert summary["status"] == "failed"
        assert "Bad config" in summary.get("error", "")

    def test_start_batch_handles_run_error(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
        (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

        data = _make_sample_data(1)
        batch_id = br.create_batch(method="tick", scorer=None, provider_kwargs={}, filename="test.jsonl", data=data)

        with patch("app.services.batch_runner.ChecklistPipeline") as MockPipeline:
            mock_pipe = MagicMock()
            mock_pipe.side_effect = RuntimeError("API error")  # pipe() per-item call
            MockPipeline.return_value = mock_pipe

            br.start_batch(batch_id, {"provider": "openrouter", "model": "test"})

            if batch_id in br._running_jobs:
                br._running_jobs[batch_id].join(timeout=5)

        summary = br.get_batch_status(batch_id)
        assert summary["status"] == "failed"
        assert "API error" in summary.get("error", "")

    def test_start_batch_not_found(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)

        success = br.start_batch("nonexistent", {})
        assert success is False

    def test_cancel_running_batch(self, tmp_data_dir, monkeypatch):
        import app.services.batch_runner as br

        monkeypatch.setattr(br, "DATA_DIR", tmp_data_dir)
        (tmp_data_dir / "batches").mkdir(parents=True, exist_ok=True)

        data = _make_sample_data(1)
        batch_id = br.create_batch(method="tick", scorer=None, provider_kwargs={}, filename="test.jsonl", data=data)

        # Simulate a running job by adding to _running_jobs
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        br._running_jobs[batch_id] = mock_thread
        br._cancel_flags[batch_id] = MagicMock()

        success = br.cancel_batch(batch_id)
        assert success is True

        summary = br.get_batch_status(batch_id)
        assert summary["status"] == "cancelled"

        # Clean up
        br._running_jobs.pop(batch_id, None)
        br._cancel_flags.pop(batch_id, None)


# ---------------------------------------------------------------------------
# Tests: Batch API endpoints
# ---------------------------------------------------------------------------

class TestBatchAPI:
    """Tests for batch API router endpoints."""

    def test_get_batch_not_found(self, client):
        resp = client.get("/api/batch/nonexistent")
        assert resp.status_code == 404

    def test_upload_and_get_status(self, client, sample_data):
        # Upload
        content = _make_json_content(sample_data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        assert resp.status_code == 200
        batch_id = resp.json()["batch_id"]

        # Get status
        resp = client.get(f"/api/batch/{batch_id}")
        assert resp.status_code == 200
        status = resp.json()
        assert status["status"] == "pending"
        assert status["total"] == 3
        assert status["config"]["method"] == "tick"

    def test_upload_and_list(self, client, sample_data):
        content = _make_json_content(sample_data)
        client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )

        resp = client.get("/api/batch")
        assert resp.status_code == 200
        batches = resp.json()
        assert len(batches) == 1
        assert batches[0]["status"] == "pending"

    def test_start_batch_endpoint(self, client, sample_data, monkeypatch):
        # Upload first
        content = _make_json_content(sample_data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        batch_id = resp.json()["batch_id"]

        mock_pipeline_result = _make_mock_pipeline_result()

        with patch("app.services.batch_runner.ChecklistPipeline") as MockPipeline:
            mock_pipe = MagicMock()
            mock_pipe.return_value = mock_pipeline_result  # pipe() per-item call
            MockPipeline.return_value = mock_pipe

            resp = client.post(f"/api/batch/{batch_id}/start", json={})
            assert resp.status_code == 200
            assert resp.json()["status"] == "started"

            # Wait for background thread
            import app.services.batch_runner as br
            if batch_id in br._running_jobs:
                br._running_jobs[batch_id].join(timeout=5)

        # Check completed
        resp = client.get(f"/api/batch/{batch_id}")
        assert resp.json()["status"] == "completed"

    def test_results_endpoint(self, client, tmp_data_dir, sample_data):
        # Upload
        content = _make_json_content(sample_data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        batch_id = resp.json()["batch_id"]

        # Write fake results
        results_path = tmp_data_dir / "batches" / batch_id / "results.jsonl"
        records = [
            {"index": i, "input": f"I{i}", "target": f"R{i}", "pass_rate": 1.0, "total_score": 1.0}
            for i in range(3)
        ]
        with open(results_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        resp = client.get(f"/api/batch/{batch_id}/results")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3

        # With pagination
        resp = client.get(f"/api/batch/{batch_id}/results?offset=1&limit=1")
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["index"] == 1

    def test_cancel_endpoint_not_running(self, client, sample_data):
        content = _make_json_content(sample_data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        batch_id = resp.json()["batch_id"]

        # Cancel when not running should fail
        resp = client.post(f"/api/batch/{batch_id}/cancel")
        assert resp.status_code == 400

    def test_export_json(self, client, tmp_data_dir, sample_data):
        content = _make_json_content(sample_data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        batch_id = resp.json()["batch_id"]

        # Write fake results
        results_path = tmp_data_dir / "batches" / batch_id / "results.jsonl"
        records = [
            {"index": 0, "input": "I0", "target": "R0", "pass_rate": 1.0, "total_score": 1.0}
        ]
        with open(results_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        resp = client.get(f"/api/batch/{batch_id}/export?format=json")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        exported = json.loads(resp.content)
        assert len(exported) == 1
        assert exported[0]["pass_rate"] == 1.0

    def test_export_csv(self, client, tmp_data_dir, sample_data):
        content = _make_json_content(sample_data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        batch_id = resp.json()["batch_id"]

        # Write fake results
        results_path = tmp_data_dir / "batches" / batch_id / "results.jsonl"
        records = [
            {"index": 0, "input": "I0", "target": "R0", "pass_rate": 1.0, "total_score": 1.0}
        ]
        with open(results_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        resp = client.get(f"/api/batch/{batch_id}/export?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "pass_rate" in lines[0]

    def test_export_no_results(self, client, sample_data):
        content = _make_json_content(sample_data)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.json", BytesIO(content), "application/json")},
            data={"method": "tick"},
        )
        batch_id = resp.json()["batch_id"]

        resp = client.get(f"/api/batch/{batch_id}/export")
        assert resp.status_code == 404


class TestUploadPath:
    """Tests for the /upload-path endpoint that reads files by path."""

    @pytest.mark.parametrize(
        "suffix,writer",
        [
            (".json", lambda data: json.dumps(data)),
            (".csv", None),
            (".jsonl", lambda data: "\n".join(json.dumps(item) for item in data)),
        ],
    )
    def test_upload_path_supported_formats(self, client, tmp_data_dir, sample_data, suffix, writer):
        test_file = tmp_data_dir / f"test_input{suffix}"

        if suffix == ".csv":
            import csv as csv_mod
            import io

            buf = io.StringIO()
            csv_writer = csv_mod.DictWriter(buf, fieldnames=["input", "target"])
            csv_writer.writeheader()
            csv_writer.writerows(sample_data)
            test_file.write_text(buf.getvalue())
        else:
            test_file.write_text(writer(sample_data))

        resp = client.post(
            "/api/batch/upload-path",
            json={"file_path": str(test_file), "method": "tick"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["total_items"] == len(sample_data)
        assert payload["filename"] == test_file.name

    def test_upload_path_not_found(self, client):
        resp = client.post(
            "/api/batch/upload-path",
            json={"file_path": "/nonexistent/file.json", "method": "tick"},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    def test_upload_path_missing_columns(self, client, tmp_data_dir):
        test_file = tmp_data_dir / "bad.json"
        test_file.write_text(json.dumps([{"foo": "bar"}]))

        resp = client.post(
            "/api/batch/upload-path",
            json={"file_path": str(test_file), "method": "tick"},
        )
        assert resp.status_code == 400
        assert "input" in resp.json()["detail"].lower()

