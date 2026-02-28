"""Tests for ItemScorer.

Integration tests use real LLM calls via OpenRouter.
Requires OPENROUTER_API_KEY environment variable.
"""

import os
import pytest

from autochecklist import ItemScorer, Checklist, ChecklistItem, Score
from autochecklist.models import ChecklistItemAnswer


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)


class TestItemScorer:
    """Structural tests for item-by-item evaluation."""

    @pytest.fixture
    def scorer(self):
        """Create ItemScorer with cheap model."""
        return ItemScorer(model="openai/gpt-4o-mini", temperature=0.0)

    @pytest.fixture
    def sample_checklist(self):
        """Create a sample checklist for testing."""
        return Checklist(
            items=[
                ChecklistItem(question="Is the response a haiku (a 3-line poem)?"),
                ChecklistItem(question="Does the response mention nature or natural elements?"),
                ChecklistItem(question="Is the response about autumn or fall season?"),
            ],
            source_method="test",
            generation_level="instance",
            input="Write a haiku about autumn leaves.",
        )

    def test_makes_multiple_calls(self, scorer, sample_checklist):
        """Test that scorer makes one LLM call per question."""
        response = "Golden leaves descend / Dancing on autumn breeze / Nature's goodbye"

        score = scorer.score(sample_checklist, target=response)

        # Check metadata for number of calls
        assert "num_calls" in score.metadata
        assert score.metadata["num_calls"] == len(sample_checklist.items)

    def test_item_scores_have_valid_answers(self, scorer, sample_checklist):
        """Test that each item score has a valid YES/NO answer."""
        response = "Autumn wind whispers / Red maple leaves spiral down / Earth welcomes them home"

        score = scorer.score(sample_checklist, target=response)

        assert isinstance(score, Score)
        assert score.scoring_method == "item"
        assert len(score.item_scores) == 3
        for item_score in score.item_scores:
            assert item_score.answer in [
                ChecklistItemAnswer.YES,
                ChecklistItemAnswer.NO,
            ]
