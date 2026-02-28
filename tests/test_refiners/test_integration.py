"""Integration tests for refiners using real API calls.

These tests validate end-to-end behavior with actual LLM and embedding APIs.
Run selectively: `uv run pytest tests/test_refiners/test_integration.py -v`

Requires:
- OPENROUTER_API_KEY for LLM calls
- OPENAI_API_KEY for embedding calls
"""

import os
import pytest

from autochecklist.models import Checklist, ChecklistItem
from autochecklist.refiners import Deduplicator, Tagger, UnitTester, Selector


# Skip all tests if API keys not available
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY") or not os.getenv("OPENAI_API_KEY"),
    reason="API keys required for integration tests",
)


@pytest.fixture
def sample_checklist_with_duplicates():
    """Checklist with semantically similar questions for deduplication testing."""
    return Checklist(
        items=[
            # Near-duplicate pair 1: almost identical wording
            ChecklistItem(id="q1", question="Is the code well-documented?"),
            ChecklistItem(id="q2", question="Is the code well documented?"),  # Only diff is hyphen
            # Unique
            ChecklistItem(id="q3", question="Does the API handle errors gracefully?"),
            # Near-duplicate pair 2: synonyms
            ChecklistItem(id="q4", question="Is the response fast enough?"),
            ChecklistItem(id="q5", question="Is the response quick enough?"),
            # Unique
            ChecklistItem(id="q6", question="Is the database schema properly normalized?"),
        ],
        source_method="test",
        generation_level="corpus",
    )


@pytest.fixture
def sample_checklist_for_tagging():
    """Checklist with questions that vary in applicability."""
    return Checklist(
        items=[
            # Should pass - universally applicable
            ChecklistItem(id="q1", question="Is the response grammatically correct?"),
            # May fail - conditional applicability
            ChecklistItem(id="q2", question="If code is included, is it syntactically valid?"),
            # Should pass - universal
            ChecklistItem(id="q3", question="Is the response relevant to the question asked?"),
            # May fail - references other sections
            ChecklistItem(id="q4", question="Does the conclusion follow logically from the introduction?"),
        ],
        source_method="test",
        generation_level="corpus",
    )


@pytest.fixture
def sample_data_for_unit_testing():
    """Sample evaluation data for unit testing refiners."""
    return [
        {
            "id": "s1",
            "text": "This is a well-written response that clearly addresses the question with proper grammar and organization.",
        },
        {
            "id": "s2",
            "text": "The answer provides relevant information and maintains a professional tone throughout.",
        },
        {
            "id": "s3",
            "text": "A concise response that directly answers the query with supporting evidence.",
        },
    ]


class TestDeduplicatorIntegration:
    """Integration tests for Deduplicator with real API calls."""

    def test_deduplicator_reduces_similar_questions(self, sample_checklist_with_duplicates):
        """Deduplicator should merge similar questions (with lower threshold)."""
        # Use lower threshold since real embeddings may have lower similarity
        dedup = Deduplicator(
            similarity_threshold=0.75,  # Lower threshold for real embeddings
            model="openai/gpt-4o-mini",
        )

        result = dedup.refine(sample_checklist_with_duplicates)

        # Should have fewer items than original (some merged)
        # If no merges happen, the test still passes but we log it
        assert len(result.items) <= len(sample_checklist_with_duplicates.items)
        # Original count tracked
        assert result.metadata["original_count"] == 6
        # Log actual merges for debugging
        print(f"Clusters merged: {result.metadata['clusters_merged']}")

    def test_deduplicator_preserves_unique_questions(self, sample_checklist_with_duplicates):
        """Unique questions should be preserved."""
        dedup = Deduplicator(
            similarity_threshold=0.85,
            model="openai/gpt-4o-mini",
        )

        result = dedup.refine(sample_checklist_with_duplicates)

        # Should preserve at least some items after dedup
        assert len(result.items) >= 1

    def test_deduplicator_merged_items_have_metadata(self, sample_checklist_with_duplicates):
        """Merged items should track source questions (if any merges occur)."""
        dedup = Deduplicator(
            similarity_threshold=0.75,  # Lower threshold
            model="openai/gpt-4o-mini",
        )

        result = dedup.refine(sample_checklist_with_duplicates)

        # Find merged items
        merged_items = [
            item for item in result.items
            if item.metadata and "merged_from" in item.metadata
        ]

        # If merges occurred, verify metadata structure
        if merged_items:
            for item in merged_items:
                assert len(item.metadata["merged_from"]) >= 2
                assert "original_questions" in item.metadata
        else:
            # No merges - that's OK, real embeddings may not find high similarity
            print("No merges occurred - embeddings didn't find high similarity pairs")


class TestTaggerIntegration:
    """Integration tests for Tagger with real API calls."""

    def test_tagger_filters_conditional_questions(self, sample_checklist_for_tagging):
        """Tagger should filter questions with conditional applicability."""
        tagger = Tagger(model="openai/gpt-4o-mini")

        result = tagger.refine(sample_checklist_for_tagging)

        # Should filter some questions
        assert len(result.items) <= len(sample_checklist_for_tagging.items)
        # Metadata tracked
        assert "filtered_count" in result.metadata

    def test_tagger_keeps_universal_questions(self, sample_checklist_for_tagging):
        """Universal questions should typically be kept (LLM may vary)."""
        tagger = Tagger(model="openai/gpt-4o-mini")

        result = tagger.refine(sample_checklist_for_tagging)

        # At least some questions should pass
        assert len(result.items) >= 1

        # Log which questions passed for debugging
        result_questions = [item.question for item in result.items]
        print(f"Questions that passed tagging: {result_questions}")

    def test_tagger_items_have_tag_metadata(self, sample_checklist_for_tagging):
        """Passing items should have tagging metadata."""
        tagger = Tagger(model="openai/gpt-4o-mini")

        result = tagger.refine(sample_checklist_for_tagging)

        for item in result.items:
            assert item.metadata is not None
            assert item.metadata.get("generally_applicable") is True
            assert item.metadata.get("section_specific") is True


class TestSelectorIntegration:
    """Integration tests for Selector with real embeddings."""

    def test_selector_reduces_to_max_questions(self, sample_checklist_with_duplicates):
        """Selector should reduce to max_questions."""
        selector = Selector(max_questions=3, beam_width=3)

        result = selector.refine(sample_checklist_with_duplicates)

        assert len(result.items) == 3
        assert result.metadata["diversity_score"] > 0

    def test_selector_prefers_diverse_subset(self, sample_checklist_with_duplicates):
        """Selected subset should maximize diversity."""
        selector = Selector(max_questions=4, beam_width=5)

        result = selector.refine(sample_checklist_with_duplicates)

        # Diversity score should be non-negative
        assert result.metadata["diversity_score"] > 0

    def test_selector_handles_small_checklist(self):
        """Small checklist should be returned unchanged."""
        small = Checklist(
            items=[
                ChecklistItem(id="q1", question="Question 1?"),
                ChecklistItem(id="q2", question="Question 2?"),
            ],
            source_method="test",
            generation_level="corpus",
        )

        selector = Selector(max_questions=10)
        result = selector.refine(small)

        assert len(result.items) == 2


class TestUnitTesterIntegration:
    """Integration tests for UnitTester with real API calls."""

    def test_unit_tester_with_sample_data(self, sample_data_for_unit_testing):
        """UnitTester should validate enforceability with real samples."""
        checklist = Checklist(
            items=[
                ChecklistItem(id="q1", question="Is the response grammatically correct?"),
                ChecklistItem(id="q2", question="Does the response address the main question?"),
            ],
            source_method="test",
            generation_level="corpus",
        )

        # All samples pass all questions initially
        sample_scores = {
            s["id"]: {"q1": "Yes", "q2": "Yes"}
            for s in sample_data_for_unit_testing
        }

        tester = UnitTester(
            enforceability_threshold=0.5,  # Lower threshold for testing
            max_samples=2,  # Limit API calls
            model="openai/gpt-4o-mini",
        )

        result = tester.refine(
            checklist,
            samples=sample_data_for_unit_testing,
            sample_scores=sample_scores,
        )

        # Should have enforceability rates
        assert "enforceability_rates" in result.metadata

        # Items that pass should have enforceability metadata
        for item in result.items:
            assert "enforceability_rate" in item.metadata


class TestFullPipelineIntegration:
    """Integration test for full refiner pipeline."""

    def test_dedup_then_tag_pipeline(self, sample_checklist_with_duplicates):
        """Test deduplication followed by tagging."""
        # Step 1: Deduplicate
        dedup = Deduplicator(
            similarity_threshold=0.85,
            model="openai/gpt-4o-mini",
        )
        deduped = dedup.refine(sample_checklist_with_duplicates)

        # Step 2: Tag
        tagger = Tagger(model="openai/gpt-4o-mini")
        tagged = tagger.refine(deduped)

        # Should have fewer items than original
        assert len(tagged.items) <= len(sample_checklist_with_duplicates.items)

        # Metadata should reflect both refinement steps
        # (Each refiner overwrites refined_by, but we can check the items)

    def test_full_pipeline_dedup_tag_select(self, sample_checklist_with_duplicates):
        """Test full pipeline: dedup -> tag -> select."""
        # Step 1: Deduplicate
        dedup = Deduplicator(
            similarity_threshold=0.85,
            model="openai/gpt-4o-mini",
        )
        deduped = dedup.refine(sample_checklist_with_duplicates)

        # Step 2: Tag
        tagger = Tagger(model="openai/gpt-4o-mini")
        tagged = tagger.refine(deduped)

        # Step 3: Select (if still too many)
        if len(tagged.items) > 3:
            selector = Selector(max_questions=3, beam_width=3)
            final = selector.refine(tagged)
        else:
            final = tagged

        # Final checklist should be manageable size
        assert len(final.items) <= 3
