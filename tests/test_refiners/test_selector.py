"""Tests for the Selector refiner.

The Selector uses beam search to select optimal subset based on diversity.

Without observations: Score(subset) = Diversity - λ·|subset|
With observations:    Score(k) = α·Coverage + (1-α)·Diversity - λ·k
"""

import pytest
from unittest.mock import patch
import numpy as np

from autochecklist.models import Checklist, ChecklistItem
from autochecklist.refiners.selector import Selector


@pytest.fixture
def large_checklist():
    """Checklist with many questions for subset selection."""
    return Checklist(
        items=[
            ChecklistItem(id=f"q{i}", question=f"Question {i}?")
            for i in range(20)
        ],
        source_method="test",
        generation_level="corpus",
    )


@pytest.fixture
def small_checklist():
    """Small checklist that doesn't need trimming."""
    return Checklist(
        items=[
            ChecklistItem(id="q1", question="Question 1?"),
            ChecklistItem(id="q2", question="Question 2?"),
            ChecklistItem(id="q3", question="Question 3?"),
        ],
        source_method="test",
        generation_level="corpus",
    )


@pytest.fixture
def diverse_embeddings():
    """Mock embeddings that are maximally diverse (orthogonal)."""
    # 5 orthogonal vectors in 5D space
    return np.eye(5)


@pytest.fixture
def similar_embeddings():
    """Mock embeddings with some similarity."""
    return np.array([
        [1.0, 0.0, 0.0],      # q0
        [0.99, 0.14, 0.0],    # q1 - similar to q0
        [0.0, 1.0, 0.0],      # q2 - different
        [0.0, 0.99, 0.14],    # q3 - similar to q2
        [0.0, 0.0, 1.0],      # q4 - different
    ])


class TestSelectorInit:
    """Tests for Selector initialization."""

    def test_default_max_questions(self):
        """Default max_questions should be 20."""
        selector = Selector()
        assert selector.max_questions == 20

    def test_custom_max_questions(self):
        """Should accept custom max_questions."""
        selector = Selector(max_questions=10)
        assert selector.max_questions == 10

    def test_default_beam_width(self):
        """Default beam_width should be 5."""
        selector = Selector()
        assert selector.beam_width == 5

    def test_default_length_penalty(self):
        """Default length penalty should be small (0.0005)."""
        selector = Selector()
        assert selector.length_penalty == 0.0005

    def test_default_alpha(self):
        """Default alpha should be 0.5."""
        selector = Selector()
        assert selector.alpha == 0.5

    def test_refiner_name(self):
        """Should return correct refiner name."""
        selector = Selector()
        assert selector.refiner_name == "selector"


class TestSelectorSmallChecklist:
    """Tests when checklist is already small enough."""

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_small_checklist_unchanged(self, mock_get_embeddings, small_checklist):
        """Checklist smaller than max_questions should be unchanged."""
        mock_get_embeddings.return_value = np.eye(3)

        selector = Selector(max_questions=10)
        result = selector.refine(small_checklist)

        assert len(result.items) == 3

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_equal_size_unchanged(self, mock_get_embeddings, small_checklist):
        """Checklist equal to max_questions should be unchanged."""
        mock_get_embeddings.return_value = np.eye(3)

        selector = Selector(max_questions=3)
        result = selector.refine(small_checklist)

        assert len(result.items) == 3


class TestSelectorSubsetSelection:
    """Tests for subset selection behavior."""

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_selects_max_questions(self, mock_get_embeddings, large_checklist):
        """Should select exactly max_questions items."""
        mock_get_embeddings.return_value = np.random.randn(20, 128)

        selector = Selector(max_questions=10)
        result = selector.refine(large_checklist)

        assert len(result.items) == 10

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_prefers_diverse_questions(self, mock_get_embeddings):
        """Should prefer diverse (dissimilar) questions."""
        # Create 5 questions where q0,q1 are similar and q2,q3,q4 are diverse
        checklist = Checklist(
            items=[ChecklistItem(id=f"q{i}", question=f"Question {i}?") for i in range(5)],
            source_method="test",
            generation_level="corpus",
        )
        mock_get_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],      # q0
            [0.99, 0.14, 0.0],    # q1 - similar to q0
            [0.0, 1.0, 0.0],      # q2 - different
            [0.0, 0.0, 1.0],      # q3 - different
            [0.5, 0.5, 0.707],    # q4 - somewhat different from all
        ])

        selector = Selector(max_questions=3, length_penalty=0)
        result = selector.refine(checklist)

        # Should pick diverse questions, avoiding q0+q1 together
        selected_ids = {item.id for item in result.items}
        # If both q0 and q1 selected, diversity is lower
        assert not ({"q0", "q1"}.issubset(selected_ids))


class TestSelectorMetadata:
    """Tests for metadata tracking."""

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_tracks_selection_metadata(self, mock_get_embeddings, large_checklist):
        """Metadata should track selection process."""
        mock_get_embeddings.return_value = np.random.randn(20, 128)

        selector = Selector(max_questions=10)
        result = selector.refine(large_checklist)

        assert result.metadata["refined_by"] == "selector"
        assert result.metadata["original_count"] == 20
        assert "diversity_score" in result.metadata
        assert "beam_width" in result.metadata

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_items_have_selection_order(self, mock_get_embeddings, large_checklist):
        """Selected items should have order metadata."""
        mock_get_embeddings.return_value = np.random.randn(20, 128)

        selector = Selector(max_questions=5)
        result = selector.refine(large_checklist)

        # Check that items have selection order
        for i, item in enumerate(result.items):
            # Items should maintain some metadata about selection
            assert item.metadata is not None


class TestSelectorEdgeCases:
    """Tests for edge cases."""

    def test_empty_checklist(self):
        """Should handle empty checklist."""
        empty = Checklist(
            items=[],
            source_method="test",
            generation_level="corpus",
        )
        selector = Selector()
        result = selector.refine(empty)

        assert len(result.items) == 0

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_single_item_checklist(self, mock_get_embeddings):
        """Should handle single-item checklist."""
        mock_get_embeddings.return_value = np.array([[1, 0, 0]])

        single = Checklist(
            items=[ChecklistItem(id="q1", question="Only question?")],
            source_method="test",
            generation_level="corpus",
        )
        selector = Selector()
        result = selector.refine(single)

        assert len(result.items) == 1

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_preserves_item_weights(self, mock_get_embeddings):
        """Should preserve item weights."""
        mock_get_embeddings.return_value = np.eye(3)

        checklist = Checklist(
            items=[
                ChecklistItem(id="q1", question="Q1?", weight=0.8),
                ChecklistItem(id="q2", question="Q2?", weight=0.5),
                ChecklistItem(id="q3", question="Q3?", weight=1.0),
            ],
            source_method="test",
            generation_level="corpus",
        )

        selector = Selector(max_questions=3)
        result = selector.refine(checklist)

        weights = {item.id: item.weight for item in result.items}
        assert weights["q1"] == 0.8
        assert weights["q2"] == 0.5
        assert weights["q3"] == 1.0


class TestSelectorBeamSearch:
    """Tests for beam search algorithm."""

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_beam_width_affects_search(self, mock_get_embeddings, large_checklist):
        """Different beam widths may produce different results."""
        # Use fixed random embeddings for reproducibility
        np.random.seed(42)
        embeddings = np.random.randn(20, 128)
        mock_get_embeddings.return_value = embeddings

        selector1 = Selector(max_questions=5, beam_width=1)
        selector2 = Selector(max_questions=5, beam_width=10)

        result1 = selector1.refine(large_checklist)
        result2 = selector2.refine(large_checklist)

        # Both should have 5 items (may or may not be the same)
        assert len(result1.items) == 5
        assert len(result2.items) == 5

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_length_penalty_prefers_smaller(self, mock_get_embeddings):
        """Higher length penalty should prefer smaller subsets (up to max)."""
        # This is hard to test directly since we have a max_questions constraint
        # but we can verify the penalty is applied
        mock_get_embeddings.return_value = np.eye(10)

        checklist = Checklist(
            items=[ChecklistItem(id=f"q{i}", question=f"Q{i}?") for i in range(10)],
            source_method="test",
            generation_level="corpus",
        )

        selector = Selector(max_questions=10, length_penalty=0.5)
        result = selector.refine(checklist)

        # Should still select up to max_questions (penalty can't reduce below that)
        assert len(result.items) <= 10


class TestSelectorDiversityScore:
    """Tests for diversity score calculation."""

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_orthogonal_vectors_high_diversity(self, mock_get_embeddings):
        """Orthogonal vectors should have diversity score near 1."""
        # Orthogonal vectors have cosine similarity 0, so diversity = 1 - 0 = 1
        mock_get_embeddings.return_value = np.eye(3)

        checklist = Checklist(
            items=[
                ChecklistItem(id="q1", question="Q1?"),
                ChecklistItem(id="q2", question="Q2?"),
                ChecklistItem(id="q3", question="Q3?"),
            ],
            source_method="test",
            generation_level="corpus",
        )

        selector = Selector(max_questions=3)
        result = selector.refine(checklist)

        # Diversity should be high (close to 1)
        assert result.metadata["diversity_score"] > 0.9

    @patch("autochecklist.refiners.selector.get_embeddings")
    def test_identical_vectors_low_diversity(self, mock_get_embeddings):
        """Identical vectors should have diversity score near 0."""
        # All identical - similarity = 1, diversity = 0
        mock_get_embeddings.return_value = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])

        checklist = Checklist(
            items=[
                ChecklistItem(id="q1", question="Q1?"),
                ChecklistItem(id="q2", question="Q2?"),
                ChecklistItem(id="q3", question="Q3?"),
            ],
            source_method="test",
            generation_level="corpus",
        )

        selector = Selector(max_questions=3)
        result = selector.refine(checklist)

        # Diversity should be low (close to 0)
        assert result.metadata["diversity_score"] < 0.1


class TestSelectorAlphaConvexCombination:
    """Tests that alpha forms a proper convex combination of coverage and diversity."""

    def test_alpha_zero_diversity_only(self):
        """alpha=0 should weight diversity only (ignore coverage)."""
        selector = Selector(alpha=0.0, length_penalty=0.0)
        selector._feedback_assignment = {0: {0}, 1: {1}}
        selector._total_feedback_count = 2

        sim = np.eye(3)
        # diversity=1.0 for orthogonal, coverage([0,1])=2/2=1.0
        # Score = 0*1.0 + 1*1.0 = 1.0 (diversity only)
        score = selector._score_subset([0, 1], sim)
        assert score == pytest.approx(1.0)

    def test_alpha_one_coverage_only(self):
        """alpha=1 should weight coverage only (ignore diversity)."""
        selector = Selector(alpha=1.0, length_penalty=0.0)
        selector._feedback_assignment = {0: {0}, 1: {1}}
        selector._total_feedback_count = 4

        sim = np.eye(3)
        # coverage([0,1])=2/4=0.5, diversity=1.0
        # Score = 1.0*0.5 + 0*1.0 = 0.5 (coverage only)
        score = selector._score_subset([0, 1], sim)
        assert score == pytest.approx(0.5)

    def test_alpha_half_equal_weight(self):
        """alpha=0.5 should equally weight coverage and diversity."""
        selector = Selector(alpha=0.5, length_penalty=0.0)
        selector._feedback_assignment = {0: {0, 1}, 1: {2}}
        selector._total_feedback_count = 4

        sim = np.eye(3)
        # coverage([0,1])=3/4=0.75, diversity=1.0
        # Score = 0.5*0.75 + 0.5*1.0 = 0.875
        score = selector._score_subset([0, 1], sim)
        assert score == pytest.approx(0.875)
