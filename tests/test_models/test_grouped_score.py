"""Tests for GroupedScore model."""

import pytest

from autochecklist.models import (
    ChecklistItemAnswer,
    GroupedScore,
    ItemScore,
    Score,
)


def _make_score(yes_count: int, no_count: int, checklist_id: str = "cl1") -> Score:
    """Create a Score with the given number of yes/no answers."""
    item_scores = []
    for i in range(yes_count):
        item_scores.append(ItemScore(item_id=f"y{i}", answer=ChecklistItemAnswer.YES))
    for i in range(no_count):
        item_scores.append(ItemScore(item_id=f"n{i}", answer=ChecklistItemAnswer.NO))
    total = yes_count + no_count
    return Score(
        checklist_id=checklist_id,
        item_scores=item_scores,
        total_score=yes_count / total if total > 0 else 0.0,
        judge_model="test",
        scoring_method="batch",
    )


class TestGroupedScore:

    def test_per_group_pass_rates(self):
        gs = GroupedScore(scores={
            "coherence": _make_score(3, 1),  # 0.75
            "clarity": _make_score(2, 2),     # 0.50
        })
        rates = gs.per_group_pass_rates
        assert rates["coherence"] == pytest.approx(0.75)
        assert rates["clarity"] == pytest.approx(0.50)

    def test_pass_rate_macro_average(self):
        gs = GroupedScore(scores={
            "coherence": _make_score(3, 1),  # 0.75
            "clarity": _make_score(2, 2),     # 0.50
        })
        assert gs.pass_rate == pytest.approx(0.625)  # (0.75 + 0.50) / 2

    def test_micro_pass_rate_pooled(self):
        gs = GroupedScore(scores={
            "coherence": _make_score(3, 1),  # 3 yes, 1 no
            "clarity": _make_score(2, 2),     # 2 yes, 2 no
        })
        # total: 5 yes, 3 no => 5/8 = 0.625
        assert gs.micro_pass_rate == pytest.approx(0.625)

    def test_micro_pass_rate_unequal_sizes(self):
        gs = GroupedScore(scores={
            "a": _make_score(1, 0),  # 1 yes
            "b": _make_score(0, 3),  # 3 no
        })
        # total: 1 yes, 3 no => 1/4 = 0.25
        assert gs.micro_pass_rate == pytest.approx(0.25)

    def test_flatten(self):
        gs = GroupedScore(scores={
            "coherence": _make_score(3, 1),
            "clarity": _make_score(2, 2),
        })
        flat = gs.flatten()
        assert isinstance(flat, Score)
        assert len(flat.item_scores) == 8  # 4 + 4
        assert flat.pass_rate == pytest.approx(0.625)  # 5/8

    def test_flatten_preserves_score_fields(self):
        gs = GroupedScore(scores={
            "a": _make_score(1, 1),
        })
        flat = gs.flatten()
        assert flat.judge_model == "test"
        assert flat.scoring_method == "batch"

    def test_empty_scores(self):
        gs = GroupedScore(scores={})
        assert gs.per_group_pass_rates == {}
        assert gs.pass_rate == 0.0
        assert gs.micro_pass_rate == 0.0

    def test_single_group(self):
        gs = GroupedScore(scores={
            "only": _make_score(4, 1),
        })
        assert gs.pass_rate == pytest.approx(0.8)
        assert gs.micro_pass_rate == pytest.approx(0.8)

    def test_mean_score_default_equals_pass_rate(self):
        """mean_score equals pass_rate when primary_metric is 'pass' (default)."""
        gs = GroupedScore(scores={
            "coherence": _make_score(3, 1),  # 0.75
            "clarity": _make_score(2, 2),     # 0.50
        })
        assert gs.mean_score == pytest.approx(gs.pass_rate)

    def test_mean_score_weighted(self):
        """mean_score averages Score.primary_score (weighted) when primary_metric is 'weighted'."""
        s1 = _make_score(3, 1)  # pass_rate=0.75
        s1.weighted_score = 0.9
        s1.primary_metric = "weighted"
        s2 = _make_score(2, 2)  # pass_rate=0.50
        s2.weighted_score = 0.4
        s2.primary_metric = "weighted"
        gs = GroupedScore(scores={"a": s1, "b": s2})
        # mean_score should average weighted_scores: (0.9 + 0.4) / 2 = 0.65
        assert gs.mean_score == pytest.approx(0.65)
        # pass_rate still uses pass_rate: (0.75 + 0.50) / 2 = 0.625
        assert gs.pass_rate == pytest.approx(0.625)

    def test_mean_score_empty(self):
        gs = GroupedScore(scores={})
        assert gs.mean_score == 0.0

    def test_metadata_default(self):
        gs = GroupedScore(scores={"a": _make_score(1, 0)})
        assert gs.metadata == {}

    def test_metadata_custom(self):
        gs = GroupedScore(
            scores={"a": _make_score(1, 0)},
            metadata={"source": "checkeval"},
        )
        assert gs.metadata["source"] == "checkeval"
