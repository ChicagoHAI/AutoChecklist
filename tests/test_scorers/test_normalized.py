"""Tests for NormalizedScorer.

Integration tests (logprobs support) use real LLM calls via OpenRouter.
"""

import os
import pytest

from autochecklist import (
    Checklist,
    ChecklistItem,
    NormalizedScorer,
)


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="Requires OPENROUTER_API_KEY"
)
class TestNormalizedScorerLogprobsSupport:
    """Test logprobs support detection and fallback behavior."""

    def test_model_with_logprobs_support(self):
        """Test that gpt-4o-mini is detected as supporting logprobs."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer = NormalizedScorer(model="openai/gpt-4o-mini")

            # Should have logprobs available
            assert scorer._logprobs_available is True
            # No UserWarning about logprobs should be emitted
            logprobs_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning) and "logprobs" in str(x.message)
            ]
            assert len(logprobs_warnings) == 0

    def test_model_without_logprobs_support(self):
        """Test that gpt-5-mini is detected as NOT supporting logprobs."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer = NormalizedScorer(model="openai/gpt-5-mini")

            # Should NOT have logprobs available
            assert scorer._logprobs_available is False
            # UserWarning about logprobs should be emitted
            logprobs_warnings = [
                x for x in w
                if issubclass(x.category, UserWarning) and "does not support logprobs" in str(x.message)
            ]
            assert len(logprobs_warnings) == 1

    def test_no_logprobs_model_returns_none_confidence(self):
        """Test scoring with model that doesn't support logprobs returns confidence=None."""
        scorer = NormalizedScorer(model="openai/gpt-5-mini")

        # Create simple checklist
        checklist = Checklist(
            items=[
                ChecklistItem(question="Does the response say hello?"),
            ],
            source_method="test",
            generation_level="instance",
        )

        score = scorer.score(checklist, target="Hello world!")

        # All item scores should have confidence=None
        for item_score in score.item_scores:
            assert item_score.confidence is None
            assert item_score.confidence_level is None

        # normalized_score should equal pass_rate (since no confidence data)
        assert score.normalized_score == score.pass_rate
