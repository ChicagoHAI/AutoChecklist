"""Tests for on_progress callback in pipeline.run_batch()."""

import pytest
from unittest.mock import patch

from autochecklist.pipeline import ChecklistPipeline
from autochecklist.models import Checklist, ChecklistItem, Score, ItemScore, ChecklistItemAnswer


@pytest.fixture
def mock_checklist():
    """A simple checklist for testing."""
    return Checklist(
        items=[
            ChecklistItem(question="Is this good?"),
            ChecklistItem(question="Is this clear?"),
        ],
        source_method="tick",
        generation_level="instance",
    )


@pytest.fixture
def mock_score():
    """A simple score for testing."""
    return Score(
        checklist_id="test",
        total_score=1.0,
        judge_model="test-model",
        scoring_method="batch",
        item_scores=[
            ItemScore(item_id="0", answer=ChecklistItemAnswer.YES),
            ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
        ],
    )


class TestOnProgressCallback:
    """Test that on_progress fires correctly during batch operations."""

    def test_run_batch_generation_and_scoring_progress(self, mock_checklist, mock_score):
        """on_progress should fire after each generate+score pair."""
        progress_calls = []

        pipe = ChecklistPipeline(from_preset="tick", generator_model="openai/gpt-4o-mini", scorer_model="openai/gpt-4o-mini")

        # Mock generate and score to avoid real API calls
        with patch.object(pipe, "generate", return_value=mock_checklist), \
             patch.object(pipe, "score", return_value=mock_score):

            data = [
                {"input": "Write a haiku", "target": "Leaves fall..."},
                {"input": "Write a limerick", "target": "There once..."},
                {"input": "Write a sonnet", "target": "Shall I..."},
            ]

            pipe.run_batch(
                data=data,
                on_progress=lambda done, total: progress_calls.append((done, total)),
            )

        # Should have progress calls, with total = 3 (number of items)
        # Each item generates a call after generate+score completes
        assert len(progress_calls) >= 3
        # The last call should show all items complete
        assert progress_calls[-1] == (3, 3)
        # Calls should be monotonically increasing
        completed_values = [c[0] for c in progress_calls]
        assert completed_values == sorted(completed_values)

    def test_run_batch_scoring_only_progress(self, mock_checklist, mock_score):
        """on_progress should fire when using shared checklist (scoring only)."""
        progress_calls = []

        pipe = ChecklistPipeline(from_preset="tick", generator_model="openai/gpt-4o-mini", scorer_model="openai/gpt-4o-mini")

        with patch.object(pipe, "score", return_value=mock_score):
            data = [
                {"input": "Write a haiku", "target": "Leaves fall..."},
                {"input": "Write a limerick", "target": "There once..."},
            ]

            pipe.run_batch(
                data=data,
                checklist=mock_checklist,
                on_progress=lambda done, total: progress_calls.append((done, total)),
            )

        # Should have progress calls for scoring each item
        assert len(progress_calls) >= 2
        assert progress_calls[-1] == (2, 2)

    def test_no_progress_callback(self, mock_checklist, mock_score):
        """run_batch works fine without on_progress."""
        pipe = ChecklistPipeline(from_preset="tick", generator_model="openai/gpt-4o-mini", scorer_model="openai/gpt-4o-mini")

        with patch.object(pipe, "generate", return_value=mock_checklist), \
             patch.object(pipe, "score", return_value=mock_score):

            data = [{"input": "Test", "target": "Result"}]
            result = pipe.run_batch(data=data)

        assert len(result.scores) == 1

    def test_progress_total_matches_data_length(self, mock_checklist, mock_score):
        """Total in progress callback should match number of items."""
        progress_calls = []

        pipe = ChecklistPipeline(from_preset="tick", generator_model="openai/gpt-4o-mini", scorer_model="openai/gpt-4o-mini")

        with patch.object(pipe, "generate", return_value=mock_checklist), \
             patch.object(pipe, "score", return_value=mock_score):

            data = [{"input": f"Test {i}", "target": f"Result {i}"} for i in range(5)]
            pipe.run_batch(
                data=data,
                on_progress=lambda done, total: progress_calls.append((done, total)),
            )

        # All progress calls should report total=5
        for done, total in progress_calls:
            assert total == 5

    def test_score_batch_progress(self, mock_checklist, mock_score):
        """score_batch should also support on_progress."""
        progress_calls = []

        pipe = ChecklistPipeline(from_preset="tick", generator_model="openai/gpt-4o-mini", scorer_model="openai/gpt-4o-mini")

        with patch.object(pipe, "score", return_value=mock_score):
            pipe.score_batch(
                checklist=mock_checklist,
                targets=["Response 1", "Response 2", "Response 3"],
                on_progress=lambda done, total: progress_calls.append((done, total)),
            )

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)
