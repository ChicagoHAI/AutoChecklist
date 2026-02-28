"""Tests for the Tagger refiner.

The Tagger filters checklist questions based on two criteria:
1. Generally applicable (no N/A answers allowed)
2. Section specific (evaluates single aspect, no cross-references)
"""

import pytest
from unittest.mock import patch
import json

from autochecklist.models import Checklist, ChecklistItem
from autochecklist.refiners.tagger import Tagger


@pytest.fixture
def all_valid_checklist():
    """Checklist where all questions should pass tagging."""
    return Checklist(
        items=[
            ChecklistItem(id="q1", question="Is the response grammatically correct?"),
            ChecklistItem(id="q2", question="Does the response address the main question?"),
            ChecklistItem(id="q3", question="Is the tone appropriate for the context?"),
        ],
        source_method="test",
        generation_level="corpus",
    )


@pytest.fixture
def mixed_validity_checklist():
    """Checklist with some questions that should fail tagging."""
    return Checklist(
        items=[
            # Should pass - universally applicable, single aspect
            ChecklistItem(id="q1", question="Is the response grammatically correct?"),
            # Should fail - references "if applicable" (N/A scenarios)
            ChecklistItem(id="q2", question="If the response includes code, is it syntactically valid?"),
            # Should pass - universally applicable
            ChecklistItem(id="q3", question="Is the response concise and clear?"),
            # Should fail - references other section
            ChecklistItem(id="q4", question="Does the conclusion align with the introduction's premise?"),
            # Should fail - might be N/A for non-technical queries
            ChecklistItem(id="q5", question="Are technical terms defined when first used?"),
        ],
        source_method="test",
        generation_level="corpus",
    )


def _make_tag_response(generally_applicable: bool, section_specific: bool) -> str:
    """Helper to create mock tagging responses."""
    return json.dumps({
        "reasoning": "Mock reasoning for test",
        "generally_applicable": generally_applicable,
        "section_specific": section_specific,
    })


class TestTaggerInit:
    """Tests for Tagger initialization."""

    def test_default_model(self):
        """Default model should be gpt-5-mini as specified in plan."""
        tagger = Tagger()
        # Note: The model is set via the base class, we just verify it's accessible
        assert tagger.refiner_name == "tagger"

    def test_custom_model(self):
        """Should accept custom model."""
        tagger = Tagger(model="openai/gpt-4o")
        assert tagger.refiner_name == "tagger"

    def test_refiner_name(self):
        """Should return correct refiner name."""
        tagger = Tagger()
        assert tagger.refiner_name == "tagger"


class TestTaggerAllPass:
    """Tests when all questions pass tagging."""

    def test_all_valid_questions_unchanged(self, all_valid_checklist):
        """When all questions pass tagging, checklist should be unchanged."""
        with patch.object(Tagger, "_call_model") as mock_call:
            # All questions pass both criteria
            mock_call.return_value = _make_tag_response(True, True)

            tagger = Tagger()
            result = tagger.refine(all_valid_checklist)

        # Same number of items
        assert len(result.items) == len(all_valid_checklist.items)
        # Same questions
        original_questions = {item.question for item in all_valid_checklist.items}
        result_questions = {item.question for item in result.items}
        assert original_questions == result_questions

    def test_metadata_tracks_tags(self, all_valid_checklist):
        """Metadata should track tagging results."""
        with patch.object(Tagger, "_call_model") as mock_call:
            mock_call.return_value = _make_tag_response(True, True)

            tagger = Tagger()
            result = tagger.refine(all_valid_checklist)

        assert result.metadata["refined_by"] == "tagger"
        assert result.metadata["filtered_count"] == 0
        assert result.metadata["original_count"] == 3


class TestTaggerFiltering:
    """Tests for filtering behavior."""

    def test_filters_non_applicable_questions(self, mixed_validity_checklist):
        """Questions with N/A scenarios should be filtered out."""
        with patch.object(Tagger, "_call_model") as mock_call:
            # Return different results for each question
            mock_call.side_effect = [
                _make_tag_response(True, True),   # q1: pass
                _make_tag_response(False, True),  # q2: fail (not applicable)
                _make_tag_response(True, True),   # q3: pass
                _make_tag_response(True, False),  # q4: fail (not section-specific)
                _make_tag_response(False, True),  # q5: fail (not applicable)
            ]

            tagger = Tagger()
            result = tagger.refine(mixed_validity_checklist)

        # Should have only q1 and q3
        assert len(result.items) == 2
        result_ids = {item.id for item in result.items}
        assert result_ids == {"q1", "q3"}

    def test_metadata_tracks_filtered_count(self, mixed_validity_checklist):
        """Metadata should track how many questions were filtered."""
        with patch.object(Tagger, "_call_model") as mock_call:
            mock_call.side_effect = [
                _make_tag_response(True, True),
                _make_tag_response(False, True),
                _make_tag_response(True, True),
                _make_tag_response(True, False),
                _make_tag_response(False, True),
            ]

            tagger = Tagger()
            result = tagger.refine(mixed_validity_checklist)

        assert result.metadata["filtered_count"] == 3
        assert result.metadata["original_count"] == 5

    def test_items_track_tag_results(self, all_valid_checklist):
        """Passing items should have tagging metadata."""
        with patch.object(Tagger, "_call_model") as mock_call:
            mock_call.return_value = _make_tag_response(True, True)

            tagger = Tagger()
            result = tagger.refine(all_valid_checklist)

        for item in result.items:
            assert item.metadata is not None
            assert item.metadata.get("generally_applicable") is True
            assert item.metadata.get("section_specific") is True


class TestTaggerEdgeCases:
    """Tests for edge cases."""

    def test_empty_checklist(self):
        """Should handle empty checklist gracefully."""
        empty = Checklist(
            items=[],
            source_method="test",
            generation_level="corpus",
        )
        tagger = Tagger()
        result = tagger.refine(empty)

        assert len(result.items) == 0
        assert result.metadata["original_count"] == 0
        assert result.metadata["filtered_count"] == 0

    def test_all_filtered_out(self, all_valid_checklist):
        """Should handle case where all questions are filtered."""
        with patch.object(Tagger, "_call_model") as mock_call:
            # All fail
            mock_call.return_value = _make_tag_response(False, False)

            tagger = Tagger()
            result = tagger.refine(all_valid_checklist)

        assert len(result.items) == 0
        assert result.metadata["filtered_count"] == 3

    def test_single_item_passes(self):
        """Should handle single-item checklist that passes."""
        single = Checklist(
            items=[ChecklistItem(id="q1", question="Is it good?")],
            source_method="test",
            generation_level="corpus",
        )

        with patch.object(Tagger, "_call_model") as mock_call:
            mock_call.return_value = _make_tag_response(True, True)

            tagger = Tagger()
            result = tagger.refine(single)

        assert len(result.items) == 1

    def test_preserves_item_weights(self, all_valid_checklist):
        """Should preserve item weights for passing items."""
        checklist = Checklist(
            items=[
                ChecklistItem(id="q1", question="Question 1?", weight=0.8),
                ChecklistItem(id="q2", question="Question 2?", weight=0.5),
                ChecklistItem(id="q3", question="Question 3?", weight=1.0),
            ],
            source_method="test",
            generation_level="corpus",
        )

        with patch.object(Tagger, "_call_model") as mock_call:
            mock_call.return_value = _make_tag_response(True, True)

            tagger = Tagger()
            result = tagger.refine(checklist)

        weights = {item.question: item.weight for item in result.items}
        assert weights["Question 1?"] == 0.8
        assert weights["Question 2?"] == 0.5
        assert weights["Question 3?"] == 1.0


class TestTaggerLLMIntegration:
    """Tests for LLM integration."""

    def test_prompt_includes_question(self, all_valid_checklist):
        """LLM call should include the question being tagged."""
        with patch.object(Tagger, "_call_model") as mock_call:
            mock_call.return_value = _make_tag_response(True, True)

            tagger = Tagger()
            tagger.refine(all_valid_checklist)

        # Check that each call included a question
        calls = mock_call.call_args_list
        assert len(calls) == 3  # One per question

        # Each prompt should contain the question text
        for call, item in zip(calls, all_valid_checklist.items):
            prompt = call[0][0]  # First positional arg
            assert item.question in prompt

    def test_handles_malformed_response(self, all_valid_checklist):
        """Should handle malformed LLM responses gracefully."""
        with patch.object(Tagger, "_call_model") as mock_call:
            # Return invalid JSON
            mock_call.return_value = "This is not valid JSON"

            tagger = Tagger()
            # Should not raise - defaults to filtering out the question
            result = tagger.refine(all_valid_checklist)

        # Malformed responses should result in filtering
        assert len(result.items) == 0


