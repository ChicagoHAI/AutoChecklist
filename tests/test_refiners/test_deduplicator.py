"""Tests for the Deduplicator refiner.
"""

import pytest
from unittest.mock import patch
import numpy as np

from autochecklist.models import Checklist, ChecklistItem
from autochecklist.refiners.deduplicator import Deduplicator


@pytest.fixture
def unique_questions_checklist():
    """Checklist where all questions are semantically distinct."""
    return Checklist(
        items=[
            ChecklistItem(id="q1", question="Is the code well-documented?"),
            ChecklistItem(id="q2", question="Does the API handle errors gracefully?"),
            ChecklistItem(id="q3", question="Is the database schema normalized?"),
        ],
        source_method="test",
        generation_level="corpus",
    )


@pytest.fixture
def similar_questions_checklist():
    """Checklist with some semantically similar questions that should be merged."""
    return Checklist(
        items=[
            # These two are similar - should be merged
            ChecklistItem(id="q1", question="Is the code properly documented?"),
            ChecklistItem(id="q2", question="Does the code have adequate documentation?"),
            # This one is unique
            ChecklistItem(id="q3", question="Does the API handle errors gracefully?"),
            # These two are similar - should be merged
            ChecklistItem(id="q4", question="Is the response time acceptable?"),
            ChecklistItem(id="q5", question="Does the system respond quickly enough?"),
        ],
        source_method="test",
        generation_level="corpus",
    )


@pytest.fixture
def mock_embeddings_unique():
    """Mock embeddings that are all dissimilar (orthogonal vectors)."""
    # Create orthogonal unit vectors - cosine similarity = 0
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])


@pytest.fixture
def mock_embeddings_similar():
    """Mock embeddings with two pairs of similar vectors."""
    # q1 and q2 are similar (high cosine), q3 is unique, q4 and q5 are similar
    return np.array([
        [1.0, 0.0, 0.0],      # q1
        [0.99, 0.1, 0.0],     # q2 - similar to q1 (cosine ~0.995)
        [0.0, 1.0, 0.0],      # q3 - unique
        [0.0, 0.0, 1.0],      # q4
        [0.0, 0.1, 0.99],     # q5 - similar to q4 (cosine ~0.995)
    ])


class TestDeduplicatorInit:
    """Tests for Deduplicator initialization."""

    def test_default_threshold(self):
        """Default similarity threshold should be 0.85."""
        dedup = Deduplicator()
        assert dedup.similarity_threshold == 0.85

    def test_custom_threshold(self):
        """Should accept custom similarity threshold."""
        dedup = Deduplicator(similarity_threshold=0.9)
        assert dedup.similarity_threshold == 0.9

    def test_refiner_name(self):
        """Should return correct refiner name."""
        dedup = Deduplicator()
        assert dedup.refiner_name == "deduplicator"


class TestDeduplicatorNoMerge:
    """Tests when no questions need merging."""

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_all_unique_questions_unchanged(
        self, mock_get_embeddings, unique_questions_checklist, mock_embeddings_unique
    ):
        """When all questions are unique, checklist should be unchanged."""
        mock_get_embeddings.return_value = mock_embeddings_unique

        dedup = Deduplicator()
        result = dedup.refine(unique_questions_checklist)

        # Same number of items
        assert len(result.items) == len(unique_questions_checklist.items)
        # Same questions (order may vary)
        original_questions = {item.question for item in unique_questions_checklist.items}
        result_questions = {item.question for item in result.items}
        assert original_questions == result_questions

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_metadata_tracks_no_merges(
        self, mock_get_embeddings, unique_questions_checklist, mock_embeddings_unique
    ):
        """Metadata should indicate no questions were merged."""
        mock_get_embeddings.return_value = mock_embeddings_unique

        dedup = Deduplicator()
        result = dedup.refine(unique_questions_checklist)

        assert result.metadata["refined_by"] == "deduplicator"
        assert result.metadata["clusters_merged"] == 0
        assert result.metadata["original_count"] == 3


class TestDeduplicatorWithMerge:
    """Tests when questions need merging."""

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_similar_questions_merged(
        self, mock_get_embeddings, similar_questions_checklist, mock_embeddings_similar
    ):
        """Similar questions should be merged into single representative."""
        mock_get_embeddings.return_value = mock_embeddings_similar

        # Mock the LLM call for merging
        with patch.object(Deduplicator, "_call_model") as mock_call:
            # Return merged questions for each cluster
            mock_call.side_effect = [
                '{"question": "Is the code well-documented?"}',  # Merge of q1, q2
                '{"question": "Is the response time acceptable?"}',  # Merge of q4, q5
            ]

            dedup = Deduplicator(similarity_threshold=0.85)
            result = dedup.refine(similar_questions_checklist)

        # Should have 3 items: 2 merged + 1 unique
        assert len(result.items) == 3

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_metadata_tracks_merges(
        self, mock_get_embeddings, similar_questions_checklist, mock_embeddings_similar
    ):
        """Metadata should track how many clusters were merged."""
        mock_get_embeddings.return_value = mock_embeddings_similar

        with patch.object(Deduplicator, "_call_model") as mock_call:
            mock_call.side_effect = [
                '{"question": "Is the code well-documented?"}',
                '{"question": "Is the response time acceptable?"}',
            ]

            dedup = Deduplicator(similarity_threshold=0.85)
            result = dedup.refine(similar_questions_checklist)

        assert result.metadata["clusters_merged"] == 2
        assert result.metadata["original_count"] == 5

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_merged_items_track_source_ids(
        self, mock_get_embeddings, similar_questions_checklist, mock_embeddings_similar
    ):
        """Merged items should track which original items they came from."""
        mock_get_embeddings.return_value = mock_embeddings_similar

        with patch.object(Deduplicator, "_call_model") as mock_call:
            mock_call.side_effect = [
                '{"question": "Is the code well-documented?"}',
                '{"question": "Is the response time acceptable?"}',
            ]

            dedup = Deduplicator(similarity_threshold=0.85)
            result = dedup.refine(similar_questions_checklist)

        # Find merged items (those with merged_from metadata)
        merged_items = [item for item in result.items if item.metadata and "merged_from" in item.metadata]
        assert len(merged_items) == 2

        # Check that source IDs are tracked
        all_merged_sources = []
        for item in merged_items:
            all_merged_sources.extend(item.metadata["merged_from"])

        # Should include q1, q2, q4, q5 (the merged ones)
        assert set(all_merged_sources) == {"q1", "q2", "q4", "q5"}


class TestDeduplicatorThreshold:
    """Tests for threshold behavior."""

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_high_threshold_fewer_merges(
        self, mock_get_embeddings, similar_questions_checklist
    ):
        """Higher threshold should result in fewer merges."""
        # Embeddings where similarity is ~0.9 (below 0.95 threshold)
        mock_get_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.436, 0.0],  # cosine with q1 ~ 0.9
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.436, 0.9],  # cosine with q4 ~ 0.9
        ])

        # With threshold 0.95, nothing should merge
        dedup = Deduplicator(similarity_threshold=0.95)
        result = dedup.refine(similar_questions_checklist)

        assert len(result.items) == 5  # No merges
        assert result.metadata["clusters_merged"] == 0


class TestDeduplicatorEdgeCases:
    """Tests for edge cases."""

    def test_empty_checklist(self):
        """Should handle empty checklist gracefully."""
        empty = Checklist(
            items=[],
            source_method="test",
            generation_level="corpus",
        )
        dedup = Deduplicator()
        result = dedup.refine(empty)

        assert len(result.items) == 0
        assert result.metadata["original_count"] == 0

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_single_item_checklist(self, mock_get_embeddings):
        """Should handle single-item checklist."""
        mock_get_embeddings.return_value = np.array([[1, 0, 0]])

        single = Checklist(
            items=[ChecklistItem(id="q1", question="Is it good?")],
            source_method="test",
            generation_level="corpus",
        )
        dedup = Deduplicator()
        result = dedup.refine(single)

        assert len(result.items) == 1
        assert result.items[0].question == "Is it good?"

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_preserves_item_weights(
        self, mock_get_embeddings, mock_embeddings_unique
    ):
        """Should preserve item weights for non-merged items."""
        mock_get_embeddings.return_value = mock_embeddings_unique

        checklist = Checklist(
            items=[
                ChecklistItem(id="q1", question="Question 1?", weight=0.8),
                ChecklistItem(id="q2", question="Question 2?", weight=0.5),
                ChecklistItem(id="q3", question="Question 3?", weight=1.0),
            ],
            source_method="test",
            generation_level="corpus",
        )

        dedup = Deduplicator()
        result = dedup.refine(checklist)

        weights = {item.question: item.weight for item in result.items}
        assert weights["Question 1?"] == 0.8
        assert weights["Question 2?"] == 0.5
        assert weights["Question 3?"] == 1.0


class TestDeduplicatorGraphBuilding:
    """Tests for internal graph building logic."""

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_finds_connected_components(
        self, mock_get_embeddings
    ):
        """Should correctly identify clusters of similar questions."""
        # Create a chain: q1-q2-q3 (all connected), q4 isolated
        mock_get_embeddings.return_value = np.array([
            [1.0, 0.0, 0.0],        # q1
            [0.99, 0.14, 0.0],      # q2 - similar to q1
            [0.95, 0.31, 0.0],      # q3 - similar to q2 (and thus connected to q1)
            [0.0, 0.0, 1.0],        # q4 - isolated
        ])

        checklist = Checklist(
            items=[
                ChecklistItem(id="q1", question="Question A?"),
                ChecklistItem(id="q2", question="Question B?"),
                ChecklistItem(id="q3", question="Question C?"),
                ChecklistItem(id="q4", question="Unique question?"),
            ],
            source_method="test",
            generation_level="corpus",
        )

        with patch.object(Deduplicator, "_call_model") as mock_call:
            mock_call.return_value = '{"question": "Merged question ABC?"}'

            dedup = Deduplicator(similarity_threshold=0.85)
            result = dedup.refine(checklist)

        # Should have 2 items: 1 merged (q1-q2-q3) + 1 unique (q4)
        assert len(result.items) == 2
        assert result.metadata["clusters_merged"] == 1


class TestDeduplicatorLLMIntegration:
    """Tests for LLM integration (merge prompt)."""

    @patch("autochecklist.refiners.deduplicator.get_embeddings")
    def test_merge_prompt_includes_all_questions(
        self, mock_get_embeddings, similar_questions_checklist, mock_embeddings_similar
    ):
        """LLM merge call should include all questions in the cluster."""
        mock_get_embeddings.return_value = mock_embeddings_similar

        with patch.object(Deduplicator, "_call_model") as mock_call:
            mock_call.side_effect = [
                '{"question": "Merged 1"}',
                '{"question": "Merged 2"}',
            ]

            dedup = Deduplicator(similarity_threshold=0.85)
            dedup.refine(similar_questions_checklist)

        # Check that the prompt includes the similar questions
        calls = mock_call.call_args_list
        assert len(calls) == 2  # Two clusters to merge

        # First merge call should include q1 and q2 questions
        first_prompt = calls[0][0][0]  # First positional arg
        assert "documented" in first_prompt.lower() or "documentation" in first_prompt.lower()
