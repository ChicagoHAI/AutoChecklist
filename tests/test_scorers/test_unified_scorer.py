"""Tests for unified ChecklistScorer API.

Tests the consolidated ChecklistScorer(mode=...) interface and
backward-compatibility factory functions.
"""

import warnings

import pytest

from autochecklist.models import (
    Checklist,
    ChecklistItem,
    ChecklistItemAnswer,
    ItemScore,
    Score,
)


# ── Score.primary_score property tests ─────────────────────────────────────


class TestScoreProperty:
    """Test Score.primary_score property routes to the correct metric."""

    def _make_score(self, primary_metric="pass", weighted=None, normalized=None):
        return Score(
            checklist_id="test",
            item_scores=[
                ItemScore(item_id="a", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="b", answer=ChecklistItemAnswer.NO),
            ],
            total_score=0.5,
            weighted_score=weighted,
            normalized_score=normalized,
            primary_metric=primary_metric,
            judge_model="test",
            scoring_method="batch",
        )

    def test_default_returns_pass_rate(self):
        score = self._make_score()
        assert score.primary_score == score.pass_rate

    def test_weighted_returns_weighted_score(self):
        score = self._make_score(primary_metric="weighted", weighted=0.75)
        assert score.primary_score == 0.75

    def test_normalized_returns_normalized_score(self):
        score = self._make_score(primary_metric="normalized", normalized=0.8)
        assert score.primary_score == 0.8

    def test_weighted_falls_back_to_pass_rate(self):
        score = self._make_score(primary_metric="weighted", weighted=None)
        assert score.primary_score == score.pass_rate

    def test_normalized_falls_back_to_pass_rate(self):
        score = self._make_score(primary_metric="normalized", normalized=None)
        assert score.primary_score == score.pass_rate

    def test_invalid_primary_metric_raises(self):
        """Invalid primary_metric should raise on construction."""
        with pytest.raises(ValueError):
            self._make_score(primary_metric="invalid")


# ── ChecklistScorer config tests (no API calls) ────────────────────────────


class TestChecklistScorerConfig:
    """Test ChecklistScorer constructor configuration."""

    def test_batch_mode(self):
        from autochecklist.scorers import ChecklistScorer
        scorer = ChecklistScorer(mode="batch", api_key="test")
        assert scorer.scoring_method == "batch"

    def test_item_mode(self):
        from autochecklist.scorers import ChecklistScorer
        scorer = ChecklistScorer(mode="item", api_key="test")
        assert scorer.scoring_method == "item"

    def test_item_mode_with_reasoning(self):
        from autochecklist.scorers import ChecklistScorer
        scorer = ChecklistScorer(mode="item", capture_reasoning=True, api_key="test")
        assert scorer.scoring_method == "item"

    def test_item_mode_with_logprobs(self):
        from autochecklist.scorers import ChecklistScorer
        scorer = ChecklistScorer(mode="item", use_logprobs=True, api_key="test")
        assert scorer.scoring_method == "normalized"

    def test_item_mode_weighted_primary(self):
        from autochecklist.scorers import ChecklistScorer
        scorer = ChecklistScorer(mode="item", primary_metric="weighted", api_key="test")
        assert scorer.scoring_method == "weighted"

    def test_invalid_mode_raises(self):
        from autochecklist.scorers import ChecklistScorer
        with pytest.raises(ValueError, match="mode must be"):
            ChecklistScorer(mode="invalid")

    def test_invalid_primary_metric_raises(self):
        from autochecklist.scorers import ChecklistScorer
        with pytest.raises(ValueError, match="primary_metric must be"):
            ChecklistScorer(mode="batch", primary_metric="invalid")

    def test_default_primary_metric(self):
        from autochecklist.scorers import ChecklistScorer
        scorer = ChecklistScorer(mode="batch", api_key="test")
        assert scorer.primary_metric == "pass"

    def test_prompt_text_property(self):
        from autochecklist.scorers import ChecklistScorer
        scorer = ChecklistScorer(mode="batch", api_key="test")
        assert len(scorer.prompt_text) > 0

    def test_custom_prompt_str(self):
        from autochecklist.scorers import ChecklistScorer
        custom = "Score this: {input} {target} {checklist}"
        scorer = ChecklistScorer(mode="batch", custom_prompt=custom, api_key="test")
        assert scorer.prompt_text == custom


# ── Deprecation factory tests ──────────────────────────────────────────────


class TestDeprecatedFactories:
    """Test that old class names emit DeprecationWarning."""

    def test_batch_scorer_deprecated(self):
        from autochecklist.scorers import BatchScorer
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer = BatchScorer(api_key="test")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "BatchScorer" in str(dep_warnings[0].message)
        assert scorer.scoring_method == "batch"

    def test_item_scorer_deprecated(self):
        from autochecklist.scorers import ItemScorer
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer = ItemScorer(api_key="test")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "ItemScorer" in str(dep_warnings[0].message)
        assert scorer.scoring_method == "item"

    def test_weighted_scorer_deprecated(self):
        from autochecklist.scorers import WeightedScorer
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer = WeightedScorer(api_key="test")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "WeightedScorer" in str(dep_warnings[0].message)
        assert scorer.scoring_method == "weighted"
        assert scorer.primary_metric == "weighted"

    def test_normalized_scorer_deprecated(self):
        from autochecklist.scorers import NormalizedScorer
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scorer = NormalizedScorer(api_key="test")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "NormalizedScorer" in str(dep_warnings[0].message)
        assert scorer.scoring_method == "normalized"
        assert scorer.primary_metric == "normalized"


# ── Weighted score calculation tests (no API) ──────────────────────────────


class TestWeightedScoreCalculation:
    """Test _calculate_weighted_score logic in unified scorer."""

    def _make_scorer(self):
        from autochecklist.scorers import ChecklistScorer
        return ChecklistScorer(mode="item", primary_metric="weighted", api_key="test")

    def test_all_yes(self):
        scorer = self._make_scorer()
        checklist = Checklist(
            items=[
                ChecklistItem(id="a", question="Q1?", weight=100),
                ChecklistItem(id="b", question="Q2?", weight=50),
            ],
            source_method="test",
            generation_level="instance",
        )
        item_scores = [
            ItemScore(item_id="a", answer=ChecklistItemAnswer.YES),
            ItemScore(item_id="b", answer=ChecklistItemAnswer.YES),
        ]
        weighted = scorer._calculate_weighted_score(item_scores, checklist)
        assert weighted == 1.0

    def test_all_no(self):
        scorer = self._make_scorer()
        checklist = Checklist(
            items=[
                ChecklistItem(id="a", question="Q1?", weight=100),
                ChecklistItem(id="b", question="Q2?", weight=50),
            ],
            source_method="test",
            generation_level="instance",
        )
        item_scores = [
            ItemScore(item_id="a", answer=ChecklistItemAnswer.NO),
            ItemScore(item_id="b", answer=ChecklistItemAnswer.NO),
        ]
        weighted = scorer._calculate_weighted_score(item_scores, checklist)
        assert weighted == 0.0

    def test_mixed(self):
        scorer = self._make_scorer()
        checklist = Checklist(
            items=[
                ChecklistItem(id="a", question="Q1?", weight=100),
                ChecklistItem(id="b", question="Q2?", weight=50),
            ],
            source_method="test",
            generation_level="instance",
        )
        item_scores = [
            ItemScore(item_id="a", answer=ChecklistItemAnswer.YES),
            ItemScore(item_id="b", answer=ChecklistItemAnswer.NO),
        ]
        weighted = scorer._calculate_weighted_score(item_scores, checklist)
        assert abs(weighted - 100 / 150) < 0.001



# ── Logprobs helper tests (no API) ────────────────────────────────────────


class TestLogprobsHelpers:
    """Test _interpret_probs and _confidence_to_level."""

    def _make_scorer(self):
        from autochecklist.scorers import ChecklistScorer
        return ChecklistScorer(mode="item", use_logprobs=True, api_key="test")

    def test_interpret_probs_high_yes(self):
        scorer = self._make_scorer()
        probs = {"yes": 0.9, "no": 0.1}
        confidence, answer, level = scorer._interpret_probs(probs)
        assert confidence == pytest.approx(0.9)
        assert answer == ChecklistItemAnswer.YES

    def test_interpret_probs_high_no(self):
        scorer = self._make_scorer()
        probs = {"yes": 0.1, "no": 0.9}
        confidence, answer, level = scorer._interpret_probs(probs)
        assert confidence == pytest.approx(0.1)
        assert answer == ChecklistItemAnswer.NO

    def test_interpret_probs_unsure(self):
        scorer = self._make_scorer()
        probs = {"yes": 0.5, "no": 0.5}
        confidence, answer, level = scorer._interpret_probs(probs)
        assert confidence == pytest.approx(0.5)

    def test_confidence_boundaries(self):
        from autochecklist.models import ConfidenceLevel
        scorer = self._make_scorer()
        cases = [
            (0.05, ChecklistItemAnswer.NO, ConfidenceLevel.NO_10),
            (0.25, ChecklistItemAnswer.NO, ConfidenceLevel.NO_30),
            (0.50, ChecklistItemAnswer.NO, ConfidenceLevel.UNSURE),
            (0.70, ChecklistItemAnswer.YES, ConfidenceLevel.YES_70),
            (0.95, ChecklistItemAnswer.YES, ConfidenceLevel.YES_90),
        ]
        for confidence, expected_answer, expected_level in cases:
            answer, level = scorer._confidence_to_level(confidence)
            assert answer == expected_answer
            assert level == expected_level
