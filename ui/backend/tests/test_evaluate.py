"""Tests for the evaluate router and service layer."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.settings_service import ENV_VAR_MAP


# ---------------------------------------------------------------------------
# Helpers: build mock library objects
# ---------------------------------------------------------------------------

def _make_mock_checklist(method="tick", num_items=3):
    """Create a mock autochecklist.models.Checklist."""
    from autochecklist.models import (
        Checklist as LibChecklist,
        ChecklistItem as LibChecklistItem,
    )

    items = [
        LibChecklistItem(question=f"Question {i+1}?", weight=1.0)
        for i in range(num_items)
    ]
    return LibChecklist(
        items=items,
        source_method=method,
        generation_level="instance",
    )


def _make_mock_score(checklist, num_yes=2, num_no=1):
    """Create a mock autochecklist.models.Score."""
    from autochecklist.models import (
        Score,
        ItemScore,
        ChecklistItemAnswer,
    )

    item_scores = []
    for i, item in enumerate(checklist.items):
        answer = ChecklistItemAnswer.YES if i < num_yes else ChecklistItemAnswer.NO
        item_scores.append(
            ItemScore(item_id=item.id, answer=answer, confidence=0.9)
        )

    total = num_yes / (num_yes + num_no) if (num_yes + num_no) > 0 else 0.0
    return Score(
        checklist_id=checklist.id,
        item_scores=item_scores,
        total_score=total,
        judge_model="test-model",
        scoring_method="binary",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_data_dir, monkeypatch):
    """Create a test client with isolated data dir and settings."""
    import app.services.settings_service as ss

    monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_data_dir / "settings.json")
    monkeypatch.setattr(ss, "ENV_PATH", tmp_data_dir / ".env")
    for env_var in ENV_VAR_MAP.values():
        monkeypatch.delenv(env_var, raising=False)

    # Ensure evaluations subdir exists
    (tmp_data_dir / "evaluations").mkdir(parents=True, exist_ok=True)

    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: POST /api/evaluate/generate
# ---------------------------------------------------------------------------

class TestGenerateEndpoint:
    """Tests for the generate-only endpoint."""

    def test_generate_returns_checklist(self, client):
        mock_cl = _make_mock_checklist("tick")

        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_cl

            resp = client.post(
                "/api/evaluate/generate",
                json={
                    "input": "Write a haiku",
                    "method": "tick",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "tick"
        assert "items" in data["checklist"]
        assert len(data["checklist"]["items"]) == 3

    def test_generate_passes_provider_model(self, client):
        mock_cl = _make_mock_checklist("tick")

        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_cl

            resp = client.post(
                "/api/evaluate/generate",
                json={
                    "input": "Write a haiku",
                    "method": "tick",
                    "provider": "openai",
                    "model": "gpt-4o",
                },
            )

        assert resp.status_code == 200
        # Verify provider kwargs were passed through
        call_kwargs = mock_gen.call_args
        assert call_kwargs.kwargs.get("provider") == "openai"
        assert call_kwargs.kwargs.get("model") == "gpt-4o"

    def test_generate_with_reference(self, client):
        mock_cl = _make_mock_checklist("rocketeval")

        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_cl

            resp = client.post(
                "/api/evaluate/generate",
                json={
                    "input": "Write a haiku",
                    "method": "rocketeval",
                    "reference": "Some reference text",
                },
            )

        assert resp.status_code == 200
        call_kwargs = mock_gen.call_args
        assert call_kwargs.kwargs.get("reference") == "Some reference text"

    def test_generate_validation_empty_input(self, client):
        resp = client.post(
            "/api/evaluate/generate",
            json={"input": "", "method": "tick"},
        )
        assert resp.status_code == 422

    def test_generate_error_returns_500(self, client):
        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = ValueError("Unknown method: bad")

            resp = client.post(
                "/api/evaluate/generate",
                json={"input": "Write a haiku", "method": "bad"},
            )

        assert resp.status_code == 500
        assert "Generation failed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Tests: POST /api/evaluate/score
# ---------------------------------------------------------------------------

class TestScoreEndpoint:
    """Tests for the score-only endpoint."""

    def test_score_returns_result(self, client):
        mock_cl = _make_mock_checklist("custom", num_items=2)
        mock_score = _make_mock_score(mock_cl, num_yes=1, num_no=1)

        with patch(
            "app.routers.evaluate.score_only", new_callable=AsyncMock
        ) as mock_sc:
            mock_sc.return_value = mock_score

            resp = client.post(
                "/api/evaluate/score",
                json={
                    "target": "Some response text",
                    "checklist_items": [
                        {"question": "Is it good?"},
                        {"question": "Is it clear?", "weight": 2.0},
                    ],
                    "scorer": "batch",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        # The score dict should have item_results from the API schema
        assert "item_results" in data["score"]

    def test_score_passes_provider_model(self, client):
        mock_cl = _make_mock_checklist("custom", num_items=1)
        mock_score = _make_mock_score(mock_cl, num_yes=1, num_no=0)

        with patch(
            "app.routers.evaluate.score_only", new_callable=AsyncMock
        ) as mock_sc:
            mock_sc.return_value = mock_score

            resp = client.post(
                "/api/evaluate/score",
                json={
                    "target": "Some text",
                    "checklist_items": [{"question": "Q1?"}],
                    "provider": "openai",
                    "model": "gpt-4o",
                },
            )

        assert resp.status_code == 200
        call_kwargs = mock_sc.call_args
        assert call_kwargs.kwargs.get("provider") == "openai"
        assert call_kwargs.kwargs.get("model") == "gpt-4o"

    def test_score_custom_scorer(self, client):
        mock_cl = _make_mock_checklist("custom", num_items=1)
        mock_score = _make_mock_score(mock_cl, num_yes=1, num_no=0)

        with patch(
            "app.routers.evaluate.score_only", new_callable=AsyncMock
        ) as mock_sc:
            mock_sc.return_value = mock_score

            resp = client.post(
                "/api/evaluate/score",
                json={
                    "target": "text",
                    "checklist_items": [{"question": "Q?"}],
                    "scorer": "weighted",
                },
            )

        assert resp.status_code == 200
        call_kwargs = mock_sc.call_args
        assert call_kwargs.kwargs.get("scorer_name") == "weighted"

    def test_score_validation_empty_target(self, client):
        resp = client.post(
            "/api/evaluate/score",
            json={
                "target": "",
                "checklist_items": [{"question": "Q?"}],
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: POST /api/evaluate/stream  - SSE streaming
# ---------------------------------------------------------------------------

class TestStreamEndpoint:
    """Tests for the SSE streaming endpoint."""

    def _parse_sse(self, raw: str) -> list[dict]:
        """Parse SSE text into list of {event, data} dicts."""
        events = []
        current_event = None
        current_data = None
        for line in raw.split("\n"):
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                current_data = line[6:]
            elif line == "" and current_event is not None:
                events.append({
                    "event": current_event,
                    "data": json.loads(current_data) if current_data else {},
                })
                current_event = None
                current_data = None
        return events

    def test_stream_auto_selects_methods_no_ref(self, client):
        """Without reference, only no-ref methods are run."""
        mock_cl = _make_mock_checklist("tick")
        mock_score = _make_mock_score(mock_cl)

        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen, patch(
            "app.routers.evaluate.score_only", new_callable=AsyncMock
        ) as mock_sc:
            mock_gen.return_value = mock_cl
            mock_sc.return_value = mock_score

            resp = client.post(
                "/api/evaluate/stream",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                },
            )

        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        event_types = [e["event"] for e in events]
        assert "done" in event_types

        # Should have checklist events for no-ref methods only
        checklist_methods = [
            e["data"]["method"]
            for e in events
            if e["event"] == "checklist"
        ]
        assert "tick" in checklist_methods
        assert "rlcf_candidates_only" in checklist_methods
        assert "rocketeval" not in checklist_methods

    def test_stream_auto_selects_methods_with_ref(self, client):
        """With reference, all methods are run."""
        mock_cl = _make_mock_checklist("tick")
        mock_score = _make_mock_score(mock_cl)

        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen, patch(
            "app.routers.evaluate.score_only", new_callable=AsyncMock
        ) as mock_sc:
            mock_gen.return_value = mock_cl
            mock_sc.return_value = mock_score

            resp = client.post(
                "/api/evaluate/stream",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                    "reference": "Golden leaves drift",
                },
            )

        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        checklist_methods = [
            e["data"]["method"]
            for e in events
            if e["event"] == "checklist"
        ]
        assert "tick" in checklist_methods
        assert "rocketeval" in checklist_methods
        assert "rlcf_direct" in checklist_methods

    def test_stream_accepts_explicit_methods(self, client):
        """When methods list is provided, only those methods run."""
        mock_cl = _make_mock_checklist("tick")
        mock_score = _make_mock_score(mock_cl)

        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen, patch(
            "app.routers.evaluate.score_only", new_callable=AsyncMock
        ) as mock_sc:
            mock_gen.return_value = mock_cl
            mock_sc.return_value = mock_score

            resp = client.post(
                "/api/evaluate/stream",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                    "methods": ["tick"],
                },
            )

        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        checklist_methods = [
            e["data"]["method"]
            for e in events
            if e["event"] == "checklist"
        ]
        assert checklist_methods == ["tick"]

    def test_stream_accepts_provider_model(self, client):
        """Provider and model params are forwarded."""
        mock_cl = _make_mock_checklist("tick")
        mock_score = _make_mock_score(mock_cl)

        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen, patch(
            "app.routers.evaluate.score_only", new_callable=AsyncMock
        ) as mock_sc:
            mock_gen.return_value = mock_cl
            mock_sc.return_value = mock_score

            resp = client.post(
                "/api/evaluate/stream",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                    "methods": ["tick"],
                    "provider": "openai",
                    "model": "gpt-4o",
                },
            )

        assert resp.status_code == 200
        call_kwargs = mock_gen.call_args
        assert call_kwargs.kwargs.get("provider") == "openai"
        assert call_kwargs.kwargs.get("model") == "gpt-4o"

    def test_stream_saves_results(self, client, tmp_data_dir):
        """Results are saved to data/evaluations/ when complete."""
        mock_cl = _make_mock_checklist("tick")
        mock_score = _make_mock_score(mock_cl)

        eval_dir = tmp_data_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen, patch(
            "app.routers.evaluate.score_only", new_callable=AsyncMock
        ) as mock_sc, patch(
            "app.routers.evaluate.DATA_DIR", tmp_data_dir
        ):
            mock_gen.return_value = mock_cl
            mock_sc.return_value = mock_score

            resp = client.post(
                "/api/evaluate/stream",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                    "methods": ["tick"],
                },
            )

        assert resp.status_code == 200
        # Check that a file was saved
        saved_files = list(eval_dir.glob("*.json"))
        assert len(saved_files) >= 1

        saved = json.loads(saved_files[0].read_text())
        assert "request" in saved
        assert "results" in saved
        assert saved["request"]["input"] == "Write a haiku"

    def test_stream_handles_generation_error(self, client):
        """Generation errors appear as error events, not crashes."""
        with patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen, patch(
            "app.routers.evaluate.score_only", new_callable=AsyncMock
        ):
            mock_gen.side_effect = RuntimeError("LLM error")

            resp = client.post(
                "/api/evaluate/stream",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves",
                    "methods": ["tick"],
                },
            )

        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        assert error_events[0]["data"]["phase"] == "checklist"


# ---------------------------------------------------------------------------
# Tests: POST /api/evaluate  - Non-streaming
# ---------------------------------------------------------------------------

class TestEvaluateEndpoint:
    """Tests for the non-streaming evaluate endpoint."""

    def _patch_gen_score(self):
        """Return context managers that mock generate_only and score_only."""
        mock_gen = patch("app.routers.evaluate.generate_only", new_callable=AsyncMock)
        mock_score = patch("app.routers.evaluate.score_only", new_callable=AsyncMock)
        return mock_gen, mock_score

    def test_evaluate_returns_results(self, client):
        mock_cl = _make_mock_checklist("tick")
        mock_score_obj = _make_mock_score(mock_cl)

        mock_gen_ctx, mock_score_ctx = self._patch_gen_score()
        with mock_gen_ctx as mock_gen, mock_score_ctx as mock_score:
            mock_gen.return_value = mock_cl
            mock_score.return_value = mock_score_obj

            resp = client.post(
                "/api/evaluate",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "tick" in data["results"]
        assert "rlcf_candidates_only" in data["results"]
        assert data["results"]["tick"]["checklist"]["source_method"] == "tick"

    def test_evaluate_accepts_provider_model(self, client):
        mock_cl = _make_mock_checklist("tick")
        mock_score_obj = _make_mock_score(mock_cl)

        mock_gen_ctx, mock_score_ctx = self._patch_gen_score()
        with mock_gen_ctx as mock_gen, mock_score_ctx as mock_score:
            mock_gen.return_value = mock_cl
            mock_score.return_value = mock_score_obj

            resp = client.post(
                "/api/evaluate",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                    "provider": "openai",
                    "model": "gpt-4o",
                },
            )

        assert resp.status_code == 200
        # generate_only should receive provider kwargs
        gen_kwargs = mock_gen.call_args
        assert gen_kwargs.kwargs.get("provider") == "openai"
        assert gen_kwargs.kwargs.get("model") == "gpt-4o"

    def test_evaluate_uses_separate_scorer_model(self, client):
        """Verify that scorer gets its own provider kwargs when specified."""
        mock_cl = _make_mock_checklist("tick")
        mock_score_obj = _make_mock_score(mock_cl)

        mock_gen_ctx, mock_score_ctx = self._patch_gen_score()
        with mock_gen_ctx as mock_gen, mock_score_ctx as mock_score:
            mock_gen.return_value = mock_cl
            mock_score.return_value = mock_score_obj

            resp = client.post(
                "/api/evaluate",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                    "generator_provider": "openai",
                    "generator_model": "gpt-4o",
                    "scorer_provider": "openai",
                    "scorer_model": "gpt-4o-mini",
                },
            )

        assert resp.status_code == 200
        gen_kwargs = mock_gen.call_args
        score_kwargs = mock_score.call_args
        assert gen_kwargs.kwargs.get("model") == "gpt-4o"
        assert score_kwargs.kwargs.get("model") == "gpt-4o-mini"

    def test_evaluate_single_method(self, client):
        """Requesting a single method returns only that method."""
        mock_cl = _make_mock_checklist("tick")
        mock_score_obj = _make_mock_score(mock_cl)

        mock_gen_ctx, mock_score_ctx = self._patch_gen_score()
        with mock_gen_ctx as mock_gen, mock_score_ctx as mock_score:
            mock_gen.return_value = mock_cl
            mock_score.return_value = mock_score_obj

            resp = client.post(
                "/api/evaluate",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                    "methods": ["tick"],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert list(data["results"].keys()) == ["tick"]

    def test_evaluate_partial_failure(self, client):
        """If one method fails but another succeeds, return 200 with successes."""
        mock_cl = _make_mock_checklist("tick")
        mock_score_obj = _make_mock_score(mock_cl)

        call_count = 0

        async def gen_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("method") == "rlcf_candidates_only":
                raise RuntimeError("simulated failure")
            return mock_cl

        mock_gen_ctx, mock_score_ctx = self._patch_gen_score()
        with mock_gen_ctx as mock_gen, mock_score_ctx as mock_score:
            mock_gen.side_effect = gen_side_effect
            mock_score.return_value = mock_score_obj

            resp = client.post(
                "/api/evaluate",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "tick" in data["results"]
        assert "rlcf_candidates_only" not in data["results"]

    def test_evaluate_all_fail_returns_500(self, client):
        """If all methods fail, return 500."""
        mock_gen_ctx, mock_score_ctx = self._patch_gen_score()
        with mock_gen_ctx as mock_gen, mock_score_ctx:
            mock_gen.side_effect = RuntimeError("simulated failure")

            resp = client.post(
                "/api/evaluate",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                },
            )

        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Tests: Method auto-selection logic
# ---------------------------------------------------------------------------

class TestMethodSelection:
    """Test the _select_methods helper."""

    def test_no_ref_no_requested(self):
        from app.routers.evaluate import _select_methods

        methods = _select_methods(reference=None, requested=None)
        assert "tick" in methods
        assert "rlcf_candidates_only" in methods
        assert "rocketeval" not in methods
        assert "rlcf_direct" not in methods

    def test_with_ref_no_requested(self):
        from app.routers.evaluate import _select_methods

        methods = _select_methods(reference="some ref", requested=None)
        assert "tick" in methods
        assert "rlcf_candidates_only" in methods
        assert "rocketeval" in methods
        assert "rlcf_direct" in methods
        assert "rlcf_candidate" in methods

    def test_requested_overrides(self):
        from app.routers.evaluate import _select_methods

        methods = _select_methods(reference="ref", requested=["tick"])
        assert methods == ["tick"]

    def test_requested_no_ref(self):
        from app.routers.evaluate import _select_methods

        methods = _select_methods(reference=None, requested=["tick", "rocketeval"])
        assert methods == ["tick", "rocketeval"]


# ---------------------------------------------------------------------------
# Tests: Async evaluate endpoints
# ---------------------------------------------------------------------------


class TestAsyncEvaluateEndpoints:
    """Tests for /api/evaluate/async, /api/evaluate/recent, and /api/evaluate/{id}."""

    def test_async_endpoint_creates_summary_and_returns_eval_id(self, client, tmp_data_dir):
        class DummyTask:
            def add_done_callback(self, fn):
                # no-op in tests
                return None

        def _fake_create_task(coro):
            # Prevent "coroutine was never awaited" warnings in tests
            coro.close()
            return DummyTask()

        with patch("app.routers.evaluate.DATA_DIR", tmp_data_dir), patch(
            "app.routers.evaluate.asyncio.create_task",
            side_effect=_fake_create_task,
        ):
            resp = client.post(
                "/api/evaluate/async",
                json={
                    "input": "Write a haiku",
                    "target": "Leaves fall gently down",
                    "methods": ["tick"],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "eval_id" in data
        eval_id = data["eval_id"]

        summary_path = tmp_data_dir / "evaluations" / eval_id / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "running"
        assert summary["methods"] == ["tick"]

    def test_recent_endpoint_returns_latest_evaluations(self, client, tmp_data_dir):
        with patch("app.routers.evaluate.DATA_DIR", tmp_data_dir):
            eval_dir = tmp_data_dir / "evaluations"
            (eval_dir / "older").mkdir(parents=True, exist_ok=True)
            (eval_dir / "newer").mkdir(parents=True, exist_ok=True)

            (eval_dir / "older" / "summary.json").write_text(
                json.dumps({"eval_id": "older", "status": "completed"})
            )
            (eval_dir / "newer" / "summary.json").write_text(
                json.dumps({"eval_id": "newer", "status": "running"})
            )

            # Ensure deterministic sort order by mtime.
            (eval_dir / "older" / "summary.json").touch()
            (eval_dir / "newer" / "summary.json").touch()

            resp = client.get("/api/evaluate/recent")

        assert resp.status_code == 200
        payload = resp.json()
        assert isinstance(payload, list)
        ids = [x["eval_id"] for x in payload]
        assert "older" in ids
        assert "newer" in ids

    def test_get_eval_status_returns_404_when_missing(self, client, tmp_data_dir):
        with patch("app.routers.evaluate.DATA_DIR", tmp_data_dir):
            resp = client.get("/api/evaluate/does-not-exist")
        assert resp.status_code == 404

    def test_get_eval_status_returns_summary_when_present(self, client, tmp_data_dir):
        with patch("app.routers.evaluate.DATA_DIR", tmp_data_dir):
            eval_id = "20260211_abcdef12"
            eval_dir = tmp_data_dir / "evaluations" / eval_id
            eval_dir.mkdir(parents=True, exist_ok=True)
            expected = {"eval_id": eval_id, "status": "completed", "results": {"tick": {}}}
            (eval_dir / "summary.json").write_text(json.dumps(expected))

            resp = client.get(f"/api/evaluate/{eval_id}")

        assert resp.status_code == 200
        assert resp.json()["eval_id"] == eval_id
        assert resp.json()["status"] == "completed"

    @pytest.mark.anyio
    async def test_run_eval_job_full_loop_updates_summary(self, tmp_data_dir):
        from app.routers.evaluate import _run_eval_job
        from app.schemas import EvaluateRequest

        eval_id = "20260211_feedbeef"
        eval_dir = tmp_data_dir / "evaluations" / eval_id
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "summary.json").write_text(
            json.dumps(
                {
                    "eval_id": eval_id,
                    "status": "running",
                    "methods": ["tick"],
                    "completed_methods": [],
                    "results": {},
                    "request": {"input": "Write a haiku", "target": "Leaves"},
                }
            )
        )

        request = EvaluateRequest(
            input="Write a haiku",
            target="Leaves fall gently down",
            methods=["tick"],
        )

        mock_cl = _make_mock_checklist("tick")
        mock_score = _make_mock_score(mock_cl)

        with patch("app.routers.evaluate.DATA_DIR", tmp_data_dir), patch(
            "app.routers.evaluate.generate_only", new_callable=AsyncMock
        ) as mock_gen, patch("app.routers.evaluate.score_only", new_callable=AsyncMock) as mock_sc:
            mock_gen.return_value = mock_cl
            mock_sc.return_value = mock_score

            await _run_eval_job(
                eval_id=eval_id,
                request=request,
                gen_pk={"provider": "openrouter", "model": "openai/gpt-4o-mini"},
                score_pk={"provider": "openrouter", "model": "openai/gpt-4o-mini"},
                methods=["tick"],
            )

        summary = json.loads((eval_dir / "summary.json").read_text())
        assert summary["status"] == "completed"
        assert "tick" in summary["results"]
        assert summary["results"]["tick"]["checklist"]["source_method"] == "tick"
        assert summary["results"]["tick"]["score"] is not None
