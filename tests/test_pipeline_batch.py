"""Tests for generate_batch() and resumable run_batch().

Tests are organized into:
- TestGenerateBatch: generate_batch() for instance-level generators
- TestResumableBatch: run_batch() with output_path resume
- TestLoadCompletedIndices: JSONL parsing helper
"""

import json
import pytest

from autochecklist import (
    ChecklistPipeline,
    BatchResult,
    Checklist,
    ChecklistItem,
    Score,
    ItemScore,
    ChecklistItemAnswer,
)
from autochecklist.pipeline import (
    _load_completed_indices,
    _record_from_result,
    _checklist_record,
)


# ============================================================================
# Fixtures
# ============================================================================

def _make_checklist(input_text="test", method="tick"):
    """Create a simple checklist for testing."""
    return Checklist(
        items=[
            ChecklistItem(id="1", question="Is it good?"),
            ChecklistItem(id="2", question="Is it complete?"),
        ],
        source_method=method,
        generation_level="instance",
        input=input_text,
    )


def _make_score(checklist):
    """Create a simple score for testing."""
    return Score(
        checklist_id=checklist.id,
        item_scores=[
            ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
            ItemScore(item_id="2", answer=ChecklistItemAnswer.NO),
        ],
        total_score=0.5,
        judge_model="test",
        scoring_method="batch",
    )


# ============================================================================
# TestGenerateBatch
# ============================================================================


class TestGenerateBatch:
    """Tests for generate_batch() method."""

    def test_generate_batch_returns_checklists(self):
        """generate_batch returns List[Checklist] without scoring."""
        pipe = ChecklistPipeline(from_preset="tick")

        checklists_out = []
        def fake_generate(input=None, **kwargs):
            cl = _make_checklist(input)
            checklists_out.append(cl)
            return cl

        pipe.generate = fake_generate

        data = [
            {"input": "Write a haiku"},
            {"input": "Write a limerick"},
        ]
        result = pipe.generate_batch(data)

        assert len(result) == 2
        assert all(isinstance(c, Checklist) for c in result)

    def test_generate_batch_from_inputs_list(self):
        """generate_batch(inputs=[...]) convenience form."""
        pipe = ChecklistPipeline(from_preset="tick")
        pipe.generate = lambda input=None, **kwargs: _make_checklist(input)

        result = pipe.generate_batch(inputs=["Write a haiku", "Write a poem"])

        assert len(result) == 2

    def test_generate_batch_raises_for_corpus_level(self):
        """Corpus-level generator raises RuntimeError."""
        pipe = ChecklistPipeline(from_preset="feedback")

        with pytest.raises(RuntimeError, match="instance-level"):
            pipe.generate_batch(inputs=["test"])

    def test_generate_batch_calls_on_progress(self):
        """on_progress callback is called for each item."""
        pipe = ChecklistPipeline(from_preset="tick")
        pipe.generate = lambda input=None, **kwargs: _make_checklist(input)

        progress_calls = []
        pipe.generate_batch(
            inputs=["a", "b", "c"],
            on_progress=lambda completed, total: progress_calls.append((completed, total)),
        )

        assert progress_calls == [(1, 3), (2, 3), (3, 3)]

    def test_generate_batch_writes_to_jsonl(self, tmp_path):
        """generate_batch with output_path creates valid JSONL."""
        pipe = ChecklistPipeline(from_preset="tick")
        pipe.generate = lambda input=None, **kwargs: _make_checklist(input)

        output_file = tmp_path / "gen_results.jsonl"
        pipe.generate_batch(
            inputs=["Write a haiku", "Write a poem"],
            output_path=str(output_file),
        )

        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "index" in record
            assert "checklist" in record

    def test_generate_batch_resumes(self, tmp_path):
        """generate_batch resumes from partial file."""
        pipe = ChecklistPipeline(from_preset="tick")

        call_count = 0
        def counting_generate(input=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_checklist(input)

        pipe.generate = counting_generate

        # Write a partial file (first item completed)
        output_file = tmp_path / "gen_partial.jsonl"
        cl = _make_checklist("Write a haiku")
        record = _checklist_record(0, {"input": "Write a haiku"}, cl)
        output_file.write_text(json.dumps(record) + "\n")

        result = pipe.generate_batch(
            inputs=["Write a haiku", "Write a poem"],
            output_path=str(output_file),
        )

        assert len(result) == 2
        assert call_count == 1  # Only generated the second item


# ============================================================================
# TestResumableBatch
# ============================================================================


class TestResumableBatch:
    """Tests for run_batch() with output_path resume."""

    def test_run_batch_writes_to_jsonl(self, tmp_path):
        """output_path creates valid JSONL."""
        pipe = ChecklistPipeline(from_preset="tick")

        # Mock generate and score
        cl = _make_checklist("test")
        pipe.generate = lambda input=None, **kwargs: cl
        pipe.score = lambda checklist, target, input=None: _make_score(checklist)

        output_file = tmp_path / "results.jsonl"
        data = [
            {"input": "Write a haiku", "target": "Leaves fall..."},
            {"input": "Write a poem", "target": "Roses are red..."},
        ]
        result = pipe.run_batch(data, output_path=str(output_file))

        assert isinstance(result, BatchResult)
        assert len(result.scores) == 2

        # Verify JSONL
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "index" in record
            assert "checklist" in record
            assert "score" in record

    def test_run_batch_resumes_from_partial(self, tmp_path):
        """Skips completed indices, generates remaining."""
        pipe = ChecklistPipeline(from_preset="tick")

        gen_count = 0
        score_count = 0

        def counting_generate(input=None, **kwargs):
            nonlocal gen_count
            gen_count += 1
            return _make_checklist(input)

        def counting_score(checklist, target, input=None):
            nonlocal score_count
            score_count += 1
            return _make_score(checklist)

        pipe.generate = counting_generate
        pipe.score = counting_score

        # Write partial file (item 0 completed)
        output_file = tmp_path / "partial.jsonl"
        cl = _make_checklist("Write a haiku")
        sc = _make_score(cl)
        record = _record_from_result(0, {"input": "Write a haiku", "target": "Leaves..."}, cl, sc)
        output_file.write_text(json.dumps(record) + "\n")

        data = [
            {"input": "Write a haiku", "target": "Leaves..."},
            {"input": "Write a poem", "target": "Roses..."},
        ]
        result = pipe.run_batch(data, output_path=str(output_file))

        assert len(result.scores) == 2
        assert gen_count == 1  # Only generated item 1
        assert score_count == 1  # Only scored item 1

    def test_run_batch_full_resume_no_calls(self, tmp_path):
        """All done → zero generate/score calls."""
        pipe = ChecklistPipeline(from_preset="tick")

        gen_count = 0
        score_count = 0

        def counting_generate(input=None, **kwargs):
            nonlocal gen_count
            gen_count += 1
            return _make_checklist(input)

        def counting_score(checklist, target, input=None):
            nonlocal score_count
            score_count += 1
            return _make_score(checklist)

        pipe.generate = counting_generate
        pipe.score = counting_score

        # Write complete file
        output_file = tmp_path / "complete.jsonl"
        records = []
        for i, item in enumerate([
            {"input": "Write a haiku", "target": "Leaves..."},
            {"input": "Write a poem", "target": "Roses..."},
        ]):
            cl = _make_checklist(item["input"])
            sc = _make_score(cl)
            records.append(json.dumps(_record_from_result(i, item, cl, sc)))
        output_file.write_text("\n".join(records) + "\n")

        data = [
            {"input": "Write a haiku", "target": "Leaves..."},
            {"input": "Write a poem", "target": "Roses..."},
        ]
        result = pipe.run_batch(data, output_path=str(output_file))

        assert len(result.scores) == 2
        assert gen_count == 0
        assert score_count == 0

    def test_run_batch_output_path_with_shared_checklist_raises(self):
        """output_path + checklist raises ValueError (shared checklist doesn't make sense with resume)."""
        pipe = ChecklistPipeline(from_preset="tick")
        cl = _make_checklist("test")

        with pytest.raises(ValueError):
            pipe.run_batch(
                data=[{"input": "test", "target": "test"}],
                checklist=cl,
                output_path="/tmp/test.jsonl",
            )

    def test_run_batch_overwrite_deletes_existing(self, tmp_path):
        """overwrite=True starts fresh even if file has results."""
        pipe = ChecklistPipeline(from_preset="tick")

        gen_count = 0
        def counting_generate(input=None, **kwargs):
            nonlocal gen_count
            gen_count += 1
            return _make_checklist(input)

        pipe.generate = counting_generate
        pipe.score = lambda checklist, target, input=None: _make_score(checklist)

        # Write existing complete file
        output_file = tmp_path / "overwrite.jsonl"
        cl = _make_checklist("old")
        sc = _make_score(cl)
        record = _record_from_result(0, {"input": "old", "target": "old"}, cl, sc)
        output_file.write_text(json.dumps(record) + "\n")

        data = [
            {"input": "Write a haiku", "target": "Leaves..."},
        ]
        pipe.run_batch(data, output_path=str(output_file), overwrite=True)

        assert gen_count == 1  # Re-generated even though file existed


# ============================================================================
# TestLoadCompletedIndices
# ============================================================================


class TestLoadCompletedIndices:
    """Tests for _load_completed_indices helper."""

    def test_missing_file_returns_empty(self):
        """Non-existent file returns empty dict."""
        result = _load_completed_indices("/nonexistent/path.jsonl")
        assert result == {}

    def test_loads_valid_records(self, tmp_path):
        """Loads valid records keyed by index."""
        f = tmp_path / "test.jsonl"
        records = [
            {"index": 0, "checklist": {"items": []}, "score": {"item_scores": []}},
            {"index": 1, "checklist": {"items": []}, "score": {"item_scores": []}},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        result = _load_completed_indices(str(f))

        assert len(result) == 2
        assert 0 in result
        assert 1 in result

    def test_skips_malformed_lines(self, tmp_path):
        """Malformed JSON lines are skipped."""
        f = tmp_path / "test.jsonl"
        f.write_text('{"index": 0, "checklist": {}, "score": {}}\nnot json\n{"index": 2, "checklist": {}, "score": {}}\n')

        result = _load_completed_indices(str(f))

        assert len(result) == 2
        assert 0 in result
        assert 2 in result

    def test_skips_incomplete_records(self, tmp_path):
        """Records without 'index' are skipped."""
        f = tmp_path / "test.jsonl"
        f.write_text('{"checklist": {}, "score": {}}\n{"index": 1, "checklist": {}, "score": {}}\n')

        result = _load_completed_indices(str(f))

        assert len(result) == 1
        assert 1 in result
