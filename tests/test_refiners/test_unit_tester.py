"""Tests for the UnitTester refiner.

The UnitTester validates that checklist questions are LLM-enforceable:
1. For each question, rewrite passing samples to fail
2. Score rewritten samples - should now get "No"
3. Enforceability rate = proportion correctly failing
4. Filter questions below threshold (default 0.7)
"""

import pytest
from unittest.mock import patch

from autochecklist.models import Checklist, ChecklistItem
from autochecklist.refiners.unit_tester import UnitTester


@pytest.fixture
def sample_checklist():
    """Basic checklist for testing."""
    return Checklist(
        items=[
            ChecklistItem(id="q1", question="Is the response grammatically correct?"),
            ChecklistItem(id="q2", question="Does the response address the main question?"),
            ChecklistItem(id="q3", question="Is the tone appropriate?"),
        ],
        source_method="test",
        generation_level="corpus",
    )


@pytest.fixture
def sample_data():
    """Sample evaluation data for unit testing."""
    return [
        {"id": "s1", "text": "This is a well-written response that addresses the question."},
        {"id": "s2", "text": "Another good response with proper grammar and tone."},
        {"id": "s3", "text": "A third sample that passes all criteria."},
    ]


@pytest.fixture
def passing_scores():
    """Mock scores where all samples pass all questions."""
    return {
        "s1": {"q1": "Yes", "q2": "Yes", "q3": "Yes"},
        "s2": {"q1": "Yes", "q2": "Yes", "q3": "Yes"},
        "s3": {"q1": "Yes", "q2": "Yes", "q3": "Yes"},
    }


class TestUnitTesterInit:
    """Tests for UnitTester initialization."""

    def test_default_threshold(self):
        """Default enforceability threshold should be 0.7."""
        tester = UnitTester()
        assert tester.enforceability_threshold == 0.7

    def test_custom_threshold(self):
        """Should accept custom threshold."""
        tester = UnitTester(enforceability_threshold=0.8)
        assert tester.enforceability_threshold == 0.8

    def test_refiner_name(self):
        """Should return correct refiner name."""
        tester = UnitTester()
        assert tester.refiner_name == "unit_tester"

    def test_max_samples_default(self):
        """Default max samples should be 10."""
        tester = UnitTester()
        assert tester.max_samples == 10


class TestUnitTesterHighEnforceability:
    """Tests when all questions have high enforceability."""

    def test_all_questions_pass_threshold(self, sample_checklist, sample_data, passing_scores):
        """Questions with high enforceability should be kept."""
        with patch.object(UnitTester, "_rewrite_sample") as mock_rewrite:
            with patch.object(UnitTester, "_score_sample") as mock_score:
                # All rewrites get "No" - perfect enforceability
                mock_rewrite.return_value = "Rewritten sample with error"
                mock_score.return_value = "No"

                tester = UnitTester(enforceability_threshold=0.7)
                result = tester.refine(
                    sample_checklist,
                    samples=sample_data,
                    sample_scores=passing_scores,
                )

        # All questions should pass
        assert len(result.items) == 3

    def test_metadata_tracks_enforceability(self, sample_checklist, sample_data, passing_scores):
        """Metadata should track enforceability rates."""
        with patch.object(UnitTester, "_rewrite_sample") as mock_rewrite:
            with patch.object(UnitTester, "_score_sample") as mock_score:
                mock_rewrite.return_value = "Rewritten sample"
                mock_score.return_value = "No"

                tester = UnitTester()
                result = tester.refine(
                    sample_checklist,
                    samples=sample_data,
                    sample_scores=passing_scores,
                )

        assert result.metadata["refined_by"] == "unit_tester"
        assert "enforceability_rates" in result.metadata


class TestUnitTesterLowEnforceability:
    """Tests when some questions have low enforceability."""

    def test_filters_low_enforceability_questions(self, sample_checklist, sample_data, passing_scores):
        """Questions with low enforceability should be filtered."""
        with patch.object(UnitTester, "_rewrite_sample") as mock_rewrite:
            with patch.object(UnitTester, "_score_sample") as mock_score:
                mock_rewrite.return_value = "Rewritten sample"
                # q1: All get No (rate=1.0) - pass
                # q2: All get Yes (rate=0.0) - fail
                # q3: 2/3 get No (rate=0.67) - fail with 0.7 threshold
                call_count = [0]

                def score_side_effect(question_id, *args, **kwargs):
                    call_count[0] += 1
                    # Each question scored 3 times (3 samples)
                    question_num = (call_count[0] - 1) // 3
                    sample_num = (call_count[0] - 1) % 3

                    if question_num == 0:  # q1: all No
                        return "No"
                    elif question_num == 1:  # q2: all Yes
                        return "Yes"
                    else:  # q3: 2 No, 1 Yes
                        return "No" if sample_num < 2 else "Yes"

                mock_score.side_effect = score_side_effect

                tester = UnitTester(enforceability_threshold=0.7)
                result = tester.refine(
                    sample_checklist,
                    samples=sample_data,
                    sample_scores=passing_scores,
                )

        # Only q1 should pass (rate=1.0)
        assert len(result.items) == 1
        assert result.items[0].id == "q1"

    def test_metadata_tracks_filtered_questions(self, sample_checklist, sample_data, passing_scores):
        """Metadata should track which questions were filtered."""
        with patch.object(UnitTester, "_rewrite_sample") as mock_rewrite:
            with patch.object(UnitTester, "_score_sample") as mock_score:
                mock_rewrite.return_value = "Rewritten"
                # q1 passes, q2 and q3 fail
                call_count = [0]

                def score_side_effect(*args, **kwargs):
                    call_count[0] += 1
                    question_num = (call_count[0] - 1) // 3
                    return "No" if question_num == 0 else "Yes"

                mock_score.side_effect = score_side_effect

                tester = UnitTester(enforceability_threshold=0.7)
                result = tester.refine(
                    sample_checklist,
                    samples=sample_data,
                    sample_scores=passing_scores,
                )

        assert result.metadata["filtered_count"] == 2


class TestUnitTesterEdgeCases:
    """Tests for edge cases."""

    def test_empty_checklist(self):
        """Should handle empty checklist."""
        empty = Checklist(
            items=[],
            source_method="test",
            generation_level="corpus",
        )
        tester = UnitTester()
        result = tester.refine(empty, samples=[], sample_scores={})

        assert len(result.items) == 0

    def test_no_passing_samples(self, sample_checklist):
        """Should handle questions with no passing samples."""
        samples = [{"id": "s1", "text": "Sample text"}]
        # No samples pass q1
        scores = {"s1": {"q1": "No", "q2": "No", "q3": "No"}}

        tester = UnitTester()
        result = tester.refine(
            sample_checklist,
            samples=samples,
            sample_scores=scores,
        )

        # Questions with no passing samples get filtered (can't test enforceability)
        assert len(result.items) == 0

    def test_preserves_item_weights(self, sample_data, passing_scores):
        """Should preserve item weights for passing questions."""
        checklist = Checklist(
            items=[
                ChecklistItem(id="q1", question="Question 1?", weight=0.8),
                ChecklistItem(id="q2", question="Question 2?", weight=0.5),
            ],
            source_method="test",
            generation_level="corpus",
        )

        with patch.object(UnitTester, "_rewrite_sample") as mock_rewrite:
            with patch.object(UnitTester, "_score_sample") as mock_score:
                mock_rewrite.return_value = "Rewritten"
                mock_score.return_value = "No"

                tester = UnitTester()
                result = tester.refine(
                    checklist,
                    samples=sample_data,
                    sample_scores=passing_scores,
                )

        weights = {item.question: item.weight for item in result.items}
        assert weights["Question 1?"] == 0.8
        assert weights["Question 2?"] == 0.5

    def test_max_samples_limit(self, sample_checklist):
        """Should limit samples tested per question."""
        # Create 15 samples
        samples = [{"id": f"s{i}", "text": f"Sample {i}"} for i in range(15)]
        scores = {f"s{i}": {"q1": "Yes", "q2": "Yes", "q3": "Yes"} for i in range(15)}

        with patch.object(UnitTester, "_rewrite_sample") as mock_rewrite:
            with patch.object(UnitTester, "_score_sample") as mock_score:
                mock_rewrite.return_value = "Rewritten"
                mock_score.return_value = "No"

                tester = UnitTester(max_samples=5)
                tester.refine(sample_checklist, samples=samples, sample_scores=scores)

        # Should only rewrite 5 samples per question (3 questions * 5 = 15)
        assert mock_rewrite.call_count == 15


class TestUnitTesterLLMIntegration:
    """Tests for LLM integration."""

    def test_rewrite_prompt_includes_question_and_sample(self, sample_checklist, sample_data, passing_scores):
        """Rewrite call should include question and sample."""
        with patch.object(UnitTester, "_rewrite_sample") as mock_rewrite:
            with patch.object(UnitTester, "_score_sample") as mock_score:
                mock_rewrite.return_value = "Rewritten"
                mock_score.return_value = "No"

                tester = UnitTester()
                tester.refine(
                    sample_checklist,
                    samples=sample_data,
                    sample_scores=passing_scores,
                )

        # Check that rewrite was called with question and sample
        calls = mock_rewrite.call_args_list
        assert len(calls) > 0
        # Each call should have question and sample info
        for call in calls:
            args = call[0]
            assert len(args) >= 2  # question, sample at minimum


class TestUnitTesterItemMetadata:
    """Tests for item-level enforceability metadata."""

    def test_items_have_enforceability_metadata(self, sample_checklist, sample_data, passing_scores):
        """Passing items should have enforceability rate in metadata."""
        with patch.object(UnitTester, "_rewrite_sample") as mock_rewrite:
            with patch.object(UnitTester, "_score_sample") as mock_score:
                mock_rewrite.return_value = "Rewritten"
                mock_score.return_value = "No"

                tester = UnitTester()
                result = tester.refine(
                    sample_checklist,
                    samples=sample_data,
                    sample_scores=passing_scores,
                )

        for item in result.items:
            assert item.metadata is not None
            assert "enforceability_rate" in item.metadata
            assert item.metadata["enforceability_rate"] == 1.0
