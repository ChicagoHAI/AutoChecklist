"""Tests for pipeline grouped scoring integration (score_group, per_category_pass_rates)."""

import pytest

from autochecklist.models import (
    Checklist,
    ChecklistItem,
    ChecklistItemAnswer,
    ItemScore,
    Score,
)
from autochecklist.pipeline import BatchResult, ChecklistPipeline


# ── score_group() ─────────────────────────────────────────────────────────


class TestScoreGroup:
    """Test ChecklistPipeline.score_group() method."""

    def test_score_group_returns_grouped_score(self):
        """score_group should exist on ChecklistPipeline and return GroupedScore."""
        assert hasattr(ChecklistPipeline, "score_group")

    def test_score_group_signature(self):
        """score_group should accept sub_checklists dict, target, and input."""
        import inspect
        sig = inspect.signature(ChecklistPipeline.score_group)
        params = list(sig.parameters.keys())
        assert "sub_checklists" in params
        assert "target" in params
        assert "input" in params


# ── per_category_pass_rates() ─────────────────────────────────────────────


def _make_score(yes: int, no: int, checklist_id: str = "cl") -> Score:
    items = []
    for i in range(yes):
        items.append(ItemScore(item_id=f"y{i}", answer=ChecklistItemAnswer.YES))
    for i in range(no):
        items.append(ItemScore(item_id=f"n{i}", answer=ChecklistItemAnswer.NO))
    total = yes + no
    return Score(
        checklist_id=checklist_id,
        item_scores=items,
        total_score=yes / total if total > 0 else 0.0,
        judge_model="test",
        scoring_method="batch",
    )


def _make_checklist_with_categories():
    return Checklist(
        id="cl",
        items=[
            ChecklistItem(id="1", question="Q1", category="coherence"),
            ChecklistItem(id="2", question="Q2", category="coherence"),
            ChecklistItem(id="3", question="Q3", category="clarity"),
        ],
        source_method="checkeval",
        generation_level="corpus",
    )


class TestPerCategoryPassRates:

    def test_returns_list(self):
        """per_category_pass_rates should return List[Dict[str, float]]."""
        # Score where item 1=YES, 2=NO, 3=YES
        score = Score(
            checklist_id="cl",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.NO),
                ItemScore(item_id="3", answer=ChecklistItemAnswer.YES),
            ],
            total_score=0.67,
            judge_model="test",
            scoring_method="batch",
        )
        cl = _make_checklist_with_categories()
        result = BatchResult(
            scores=[score],
            data=[{"input": "test", "target": "test"}],
            checklists=[cl],
        )
        rates = result.per_category_pass_rates()
        assert isinstance(rates, list)
        # Should have one entry per data item
        assert len(rates) == 1

    def test_computes_per_category(self):
        """Should compute pass rates for each category within each example."""
        score = Score(
            checklist_id="cl",
            item_scores=[
                ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                ItemScore(item_id="2", answer=ChecklistItemAnswer.NO),
                ItemScore(item_id="3", answer=ChecklistItemAnswer.YES),
            ],
            total_score=0.67,
            judge_model="test",
            scoring_method="batch",
        )
        cl = _make_checklist_with_categories()
        result = BatchResult(
            scores=[score],
            data=[{"input": "test", "target": "test"}],
            checklists=[cl],
        )
        rates = result.per_category_pass_rates()
        # coherence: 1 YES + 1 NO = 0.5, clarity: 1 YES = 1.0
        assert rates[0]["coherence"] == pytest.approx(0.5)
        assert rates[0]["clarity"] == pytest.approx(1.0)

    def test_shared_checklist(self):
        """Should work with a shared checklist across all examples."""
        cl = _make_checklist_with_categories()
        scores = [
            Score(
                checklist_id="cl",
                item_scores=[
                    ItemScore(item_id="1", answer=ChecklistItemAnswer.YES),
                    ItemScore(item_id="2", answer=ChecklistItemAnswer.YES),
                    ItemScore(item_id="3", answer=ChecklistItemAnswer.NO),
                ],
                total_score=0.67,
                judge_model="test",
                scoring_method="batch",
            ),
            Score(
                checklist_id="cl",
                item_scores=[
                    ItemScore(item_id="1", answer=ChecklistItemAnswer.NO),
                    ItemScore(item_id="2", answer=ChecklistItemAnswer.NO),
                    ItemScore(item_id="3", answer=ChecklistItemAnswer.YES),
                ],
                total_score=0.33,
                judge_model="test",
                scoring_method="batch",
            ),
        ]
        result = BatchResult(
            scores=scores,
            data=[{"input": "a", "target": "a"}, {"input": "b", "target": "b"}],
            checklist=cl,
        )
        rates = result.per_category_pass_rates()
        assert len(rates) == 2
        # First example: coherence 2/2=1.0, clarity 0/1=0.0
        assert rates[0]["coherence"] == pytest.approx(1.0)
        assert rates[0]["clarity"] == pytest.approx(0.0)
        # Second example: coherence 0/2=0.0, clarity 1/1=1.0
        assert rates[1]["coherence"] == pytest.approx(0.0)
        assert rates[1]["clarity"] == pytest.approx(1.0)

    def test_empty_batch(self):
        result = BatchResult(scores=[], data=[])
        rates = result.per_category_pass_rates()
        assert rates == []
