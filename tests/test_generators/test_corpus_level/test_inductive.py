"""Tests for InductiveGenerator generator and its refiners.

The InductiveGenerator generator creates checklists from feedback comments.

Pipeline:
1. Generate: LLM creates questions from feedback batches
2. Deduplicate: Merge semantically similar questions
3. Tag: Filter by applicability and specificity
4. (Optional) UnitTest: Validate enforceability
5. Select: Beam search for diverse subset
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from autochecklist.models import Checklist, ChecklistItem
from autochecklist.generators.corpus_level.inductive import InductiveGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_feedback():
    """Sample feedback comments for testing."""
    return [
        "The response was too long and rambling.",
        "Good use of examples to illustrate the point.",
        "The answer didn't directly address the question asked.",
        "Clear and well-organized structure.",
        "Missing important details about edge cases.",
        "Excellent grammar and professional tone.",
        "The response was confusing and hard to follow.",
        "Good explanation but lacked practical examples.",
    ]


@pytest.fixture
def large_feedback():
    """250 feedback items to trigger multiple batches at batch_size=100."""
    return [f"Feedback item {i}" for i in range(250)]


@pytest.fixture
def mock_generated_questions():
    """Mock LLM response for question generation."""
    return {
        "questions": [
            {
                "question": "Is the response concise and focused?",
                "source_feedback_indices": [0, 6],
            },
            {
                "question": "Does the response include relevant examples?",
                "source_feedback_indices": [1, 7],
            },
            {
                "question": "Does the response directly address the question asked?",
                "source_feedback_indices": [2],
            },
            {
                "question": "Is the response well-organized and easy to follow?",
                "source_feedback_indices": [3, 6],
            },
            {
                "question": "Does the response cover important edge cases?",
                "source_feedback_indices": [4],
            },
            {
                "question": "Is the grammar correct and tone professional?",
                "source_feedback_indices": [5],
            },
        ]
    }


@pytest.fixture
def sample_checklist():
    """Checklist with 5 items for selector/unit-tester tests."""
    return Checklist(
        items=[
            ChecklistItem(id="q-0", question="Is the response concise?"),
            ChecklistItem(id="q-1", question="Does it include examples?"),
            ChecklistItem(id="q-2", question="Does it address the question?"),
            ChecklistItem(id="q-3", question="Is it well-organized?"),
            ChecklistItem(id="q-4", question="Does it cover edge cases?"),
        ],
        source_method="feedback",
        generation_level="corpus",
    )


def _make_mock_questions(start_idx: int, count: int):
    """Build a mock LLM JSON response for a batch of feedback items."""
    questions = []
    for i in range(0, count, 2):
        questions.append({
            "question": f"Question covering [{start_idx + i}]?",
            "source_feedback_indices": [start_idx + i],
        })
    return json.dumps({"questions": questions})


# ===========================================================================
# 1. InductiveGenerator Init
# ===========================================================================

class TestInductiveGeneratorInit:
    """Tests for InductiveGenerator initialization."""

    def test_default_model(self):
        """Should have default model."""
        gen = InductiveGenerator()
        assert gen.model is not None

    def test_custom_model(self):
        """Should accept custom model."""
        gen = InductiveGenerator(model="openai/gpt-4o")
        assert gen.model == "openai/gpt-4o"

    def test_default_refinement_options(self):
        """Should have sensible default refinement options."""
        gen = InductiveGenerator()
        assert gen.dedup_threshold == 0.85
        assert gen.max_questions == 20

    def test_batch_size_parameter(self):
        """InductiveGenerator should accept batch_size parameter."""
        gen = InductiveGenerator(batch_size=50)
        assert gen.batch_size == 50

    def test_default_batch_size(self):
        """Default batch_size should be 100."""
        gen = InductiveGenerator()
        assert gen.batch_size == 100


# ===========================================================================
# 2. InductiveGenerator Generation
# ===========================================================================

class TestInductiveGeneratorGeneration:
    """Tests for question generation from feedback."""

    def test_generates_checklist_from_feedback(self, sample_feedback, mock_generated_questions):
        """Should generate checklist from feedback comments."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            # Mock refiners to pass through
            with patch("autochecklist.generators.corpus_level.inductive.Deduplicator") as mock_dedup:
                with patch("autochecklist.generators.corpus_level.inductive.Tagger") as mock_tag:
                    with patch("autochecklist.generators.corpus_level.inductive.Selector") as mock_sel:
                        # Set up mock refiners to return input unchanged
                        mock_dedup.return_value.refine.side_effect = lambda x: x
                        mock_tag.return_value.refine.side_effect = lambda x: x
                        mock_sel.return_value.refine.side_effect = lambda x: x

                        gen = InductiveGenerator()
                        checklist = gen.generate(observations=sample_feedback)

        assert isinstance(checklist, Checklist)
        assert len(checklist.items) == 6  # 6 questions from mock
        assert checklist.generation_level == "corpus"
        assert checklist.source_method == "feedback"

    def test_tracks_source_feedback_indices(self, sample_feedback, mock_generated_questions):
        """Generated items should track which feedback they address."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            with patch("autochecklist.generators.corpus_level.inductive.Deduplicator") as mock_dedup:
                with patch("autochecklist.generators.corpus_level.inductive.Tagger") as mock_tag:
                    with patch("autochecklist.generators.corpus_level.inductive.Selector") as mock_sel:
                        mock_dedup.return_value.refine.side_effect = lambda x: x
                        mock_tag.return_value.refine.side_effect = lambda x: x
                        mock_sel.return_value.refine.side_effect = lambda x: x

                        gen = InductiveGenerator()
                        checklist = gen.generate(observations=sample_feedback)

        # Check that items have source indices
        for item in checklist.items:
            assert "source_feedback_indices" in item.metadata


# ===========================================================================
# 3. Batched Generation
# ===========================================================================

class TestBatchedGeneration:
    """InductiveGenerator should split feedback into batches for LLM calls."""

    def test_batched_generation_makes_correct_number_of_calls(self, large_feedback):
        """250 items with batch_size=100 should make 3 LLM calls."""
        gen = InductiveGenerator(batch_size=100)

        # Mock the LLM client to return valid JSON for each batch
        mock_client = MagicMock()
        responses = []
        for batch_idx in range(3):
            start = batch_idx * 100
            batch_count = min(100, 250 - start)
            responses.append({
                "choices": [{"message": {"content": _make_mock_questions(start, batch_count)}}]
            })
        mock_client.chat_completion.side_effect = responses

        with patch.object(gen, "_get_or_create_client", return_value=mock_client):
            gen._generate_questions(large_feedback, domain="test")

        assert mock_client.chat_completion.call_count == 3

    def test_batched_generation_uses_global_indices(self, large_feedback):
        """Feedback indices in the prompt should be global, not batch-local."""
        gen = InductiveGenerator(batch_size=100)

        mock_client = MagicMock()
        prompts_received = []

        def capture_prompt(**kwargs):
            prompt = kwargs["messages"][-1]["content"]
            prompts_received.append(prompt)
            # Return minimal valid response
            return {"choices": [{"message": {"content": json.dumps({"questions": []})}}]}

        mock_client.chat_completion.side_effect = capture_prompt

        with patch.object(gen, "_get_or_create_client", return_value=mock_client):
            gen._generate_questions(large_feedback, domain="test")

        # Second batch (items 100-199) should have indices starting at [100]
        assert "[100]" in prompts_received[1]
        # Third batch (items 200-249) should have indices starting at [200]
        assert "[200]" in prompts_received[2]
        # First batch should start at [0]
        assert "[0]" in prompts_received[0]

    def test_batched_generation_pools_results(self, large_feedback):
        """Questions from all batches should be pooled into a single list."""
        gen = InductiveGenerator(batch_size=100)

        mock_client = MagicMock()
        # Each batch returns 2 questions
        responses = []
        for batch_idx in range(3):
            start = batch_idx * 100
            batch_count = min(100, 250 - start)
            responses.append({
                "choices": [{"message": {"content": _make_mock_questions(start, batch_count)}}]
            })
        mock_client.chat_completion.side_effect = responses

        with patch.object(gen, "_get_or_create_client", return_value=mock_client):
            questions = gen._generate_questions(large_feedback, domain="test")

        # 3 batches × 50 questions each (one per 2 items) = 50+50+25 = 125
        assert len(questions) == 125

    def test_small_feedback_single_batch(self, sample_feedback):
        """Feedback smaller than batch_size should make exactly 1 LLM call."""
        gen = InductiveGenerator(batch_size=100)

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {
            "choices": [{"message": {"content": _make_mock_questions(0, 8)}}]
        }

        with patch.object(gen, "_get_or_create_client", return_value=mock_client):
            gen._generate_questions(sample_feedback, domain="test")

        assert mock_client.chat_completion.call_count == 1


# ===========================================================================
# 4. Refinement Pipeline
# ===========================================================================

class TestInductiveGeneratorRefinement:
    """Tests for the refinement pipeline."""

    def test_applies_deduplication(self, sample_feedback, mock_generated_questions):
        """Should apply deduplication refiner."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            with patch("autochecklist.generators.corpus_level.inductive.Deduplicator") as mock_dedup:
                with patch("autochecklist.generators.corpus_level.inductive.Tagger") as mock_tag:
                    with patch("autochecklist.generators.corpus_level.inductive.Selector") as mock_sel:
                        mock_dedup_instance = MagicMock()
                        mock_dedup_instance.refine.side_effect = lambda x: x
                        mock_dedup.return_value = mock_dedup_instance

                        mock_tag.return_value.refine.side_effect = lambda x: x
                        mock_sel.return_value.refine.side_effect = lambda x: x

                        gen = InductiveGenerator(dedup_threshold=0.9)
                        gen.generate(observations=sample_feedback)

        # Verify deduplicator was called
        mock_dedup.assert_called_once()
        mock_dedup_instance.refine.assert_called_once()

    def test_applies_tagging(self, sample_feedback, mock_generated_questions):
        """Should apply tagging refiner."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            with patch("autochecklist.generators.corpus_level.inductive.Deduplicator") as mock_dedup:
                with patch("autochecklist.generators.corpus_level.inductive.Tagger") as mock_tag:
                    with patch("autochecklist.generators.corpus_level.inductive.Selector") as mock_sel:
                        mock_dedup.return_value.refine.side_effect = lambda x: x

                        mock_tag_instance = MagicMock()
                        mock_tag_instance.refine.side_effect = lambda x: x
                        mock_tag.return_value = mock_tag_instance

                        mock_sel.return_value.refine.side_effect = lambda x: x

                        gen = InductiveGenerator()
                        gen.generate(observations=sample_feedback)

        # Verify tagger was called
        mock_tag.assert_called_once()
        mock_tag_instance.refine.assert_called_once()

    def test_applies_selection(self, sample_feedback, mock_generated_questions):
        """Should apply selector refiner when items > max_questions."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            with patch("autochecklist.generators.corpus_level.inductive.Deduplicator") as mock_dedup:
                with patch("autochecklist.generators.corpus_level.inductive.Tagger") as mock_tag:
                    with patch("autochecklist.generators.corpus_level.inductive.Selector") as mock_sel:
                        mock_dedup.return_value.refine.side_effect = lambda x: x
                        mock_tag.return_value.refine.side_effect = lambda x: x

                        mock_sel_instance = MagicMock()
                        mock_sel_instance.refine.side_effect = lambda x: x
                        mock_sel.return_value = mock_sel_instance

                        # Set max_questions lower than generated count (6) to trigger selection
                        gen = InductiveGenerator(max_questions=3)
                        gen.generate(observations=sample_feedback)

        # Verify selector was called (since 6 > 3)
        mock_sel.assert_called_once()
        mock_sel_instance.refine.assert_called_once()

    def test_can_skip_refinement(self, sample_feedback, mock_generated_questions):
        """Should allow skipping refinement steps."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = InductiveGenerator()
            checklist = gen.generate(
                observations=sample_feedback,
                skip_dedup=True,
                skip_tagging=True,
                skip_selection=True,
            )

        # Should have raw questions (no refinement)
        assert len(checklist.items) == 6


# ===========================================================================
# 5. Coverage-Aware Selector
# ===========================================================================

class TestSelectorCoverage:
    """Selector should use feedback classification for coverage-aware selection."""

    def test_selector_accepts_feedback_parameter(self):
        """Selector should accept feedback list and classifier_model."""
        from autochecklist.refiners.selector import Selector

        feedback = ["item 1", "item 2"]
        sel = Selector(
            max_questions=3,
            observations=feedback,
            classifier_model="openai/gpt-4o-mini",
        )
        assert sel.observations == feedback
        assert sel.classifier_model == "openai/gpt-4o-mini"

    def test_selector_without_feedback_unchanged(self, sample_checklist):
        """Without feedback, Selector should use diversity-only scoring (backward compat)."""
        from autochecklist.refiners.selector import Selector

        sel = Selector(max_questions=3)
        # No feedback => no classification should happen
        assert sel.observations is None

        # Verify _score_subset still works without coverage
        with patch.object(sel, "_get_similarity_matrix") as mock_sim:
            import numpy as np
            mock_sim.return_value = np.eye(5)
            result = sel.refine(sample_checklist)

        # Should still produce a valid checklist
        assert isinstance(result, Checklist)

    def test_classify_feedback_builds_assignment(self, sample_checklist):
        """_classify_feedback should build question->feedback assignment map."""
        from autochecklist.refiners.selector import Selector

        feedback = ["Too verbose", "Needs examples", "Off topic"]
        sel = Selector(max_questions=5, observations=feedback, classifier_model="openai/gpt-4o-mini")

        # Mock _call_model to return question numbers that each feedback covers
        classify_responses = [
            "0, 3",  # feedback 0 covers questions 0 and 3
            "1",     # feedback 1 covers question 1
            "2",     # feedback 2 covers question 2
        ]

        with patch.object(sel, "_call_model", side_effect=classify_responses):
            assignment = sel._classify_feedback(sample_checklist)

        # assignment maps question index -> set of feedback indices
        assert 0 in assignment[0]  # q-0 covered by feedback 0
        assert 0 in assignment[3]  # q-3 covered by feedback 0
        assert 1 in assignment[1]  # q-1 covered by feedback 1
        assert 2 in assignment[2]  # q-2 covered by feedback 2

    def test_compute_coverage(self):
        """_compute_coverage should compute fraction of feedback covered."""
        from autochecklist.refiners.selector import Selector

        feedback = ["a", "b", "c", "d"]
        sel = Selector(max_questions=5, observations=feedback)

        # Simulate assignment: q0 covers {0,1}, q1 covers {2}, q2 covers {1,3}
        sel._feedback_assignment = {
            0: {0, 1},
            1: {2},
            2: {1, 3},
        }
        sel._total_feedback_count = 4

        # Selecting q0 and q1: covers {0,1,2} = 3/4 = 0.75
        assert sel._compute_coverage([0, 1]) == pytest.approx(0.75)

        # Selecting all: covers {0,1,2,3} = 4/4 = 1.0
        assert sel._compute_coverage([0, 1, 2]) == pytest.approx(1.0)

        # Empty selection
        assert sel._compute_coverage([]) == pytest.approx(0.0)

    def test_score_subset_includes_coverage(self):
        """_score_subset should include coverage term when feedback is provided."""
        from autochecklist.refiners.selector import Selector
        import numpy as np

        feedback = ["a", "b", "c", "d"]
        sel = Selector(
            max_questions=5,
            observations=feedback,
            alpha=0.5,
            length_penalty=0.0,
        )

        # Set up assignment
        sel._feedback_assignment = {
            0: {0, 1},
            1: {2},
            2: {1, 3},
        }
        sel._total_feedback_count = 4

        # Identity similarity matrix (all items perfectly diverse => diversity=1.0 for pairs)
        sim = np.eye(3)

        # Score with items [0, 1]: diversity=1.0, coverage=3/4=0.75
        score = sel._score_subset([0, 1], sim)
        # Score = 0.5*0.75 + 0.5*1.0 - 0.0 = 0.875
        assert score == pytest.approx(0.875)

    def test_alpha_parameter(self):
        """Selector should accept alpha for coverage-diversity trade-off."""
        from autochecklist.refiners.selector import Selector

        sel = Selector(alpha=0.7)
        assert sel.alpha == 0.7


# ===========================================================================
# 6. Auto-Scored Unit Testing
# ===========================================================================

class TestUnitTesterAutoScoring:
    """UnitTester should auto-score references when sample_scores not provided."""

    def test_raw_samples_triggers_auto_scoring(self, sample_checklist):
        """Providing raw_samples without sample_scores should trigger auto-scoring."""
        from autochecklist.refiners.unit_tester import UnitTester

        tester = UnitTester(enforceability_threshold=0.0)

        raw_samples = [
            {"id": "ref-0", "text": "A short, clear response."},
            {"id": "ref-1", "text": "Another good response with examples."},
        ]

        mock_auto = MagicMock(return_value={
            "ref-0": {"q-0": "Yes", "q-1": "Yes", "q-2": "Yes", "q-3": "Yes", "q-4": "Yes"},
            "ref-1": {"q-0": "Yes", "q-1": "Yes", "q-2": "Yes", "q-3": "Yes", "q-4": "Yes"},
        })
        tester._auto_score_samples = mock_auto
        tester._rewrite_sample = MagicMock(return_value="bad rewrite")
        tester._score_sample = MagicMock(return_value="No")

        tester.refine(sample_checklist, raw_samples=raw_samples)

        # _auto_score_samples should have been called since no sample_scores given
        mock_auto.assert_called_once_with(sample_checklist, raw_samples)

    def test_raw_samples_with_existing_scores_skips_auto(self, sample_checklist):
        """If sample_scores already provided, raw_samples auto-scoring should be skipped."""
        from autochecklist.refiners.unit_tester import UnitTester

        tester = UnitTester(enforceability_threshold=0.0)

        raw_samples = [
            {"id": "ref-0", "text": "A response."},
        ]
        # Pre-computed scores
        sample_scores = {
            "ref-0": {"q-0": "Yes", "q-1": "No", "q-2": "Yes", "q-3": "Yes", "q-4": "No"},
        }

        with patch.object(tester, "_rewrite_sample", return_value="bad"):
            with patch.object(tester, "_score_sample", return_value="No"):
                result = tester.refine(
                    sample_checklist,
                    samples=raw_samples,
                    sample_scores=sample_scores,
                )

        assert isinstance(result, Checklist)

    def test_no_raw_samples_no_samples_returns_empty(self, sample_checklist):
        """No raw_samples and no samples should return empty (existing behavior)."""
        from autochecklist.refiners.unit_tester import UnitTester

        tester = UnitTester()
        result = tester.refine(sample_checklist)

        assert len(result.items) == 0

    def test_auto_score_builds_sample_scores(self, sample_checklist):
        """_auto_score_samples should build a complete sample_scores dict."""
        from autochecklist.refiners.unit_tester import UnitTester

        tester = UnitTester()

        raw_samples = [
            {"id": "ref-0", "text": "Response A"},
            {"id": "ref-1", "text": "Response B"},
        ]

        # Mock to alternate Yes/No
        scores = iter(["Yes", "No", "Yes", "No", "Yes",
                       "No", "Yes", "No", "Yes", "No"])
        with patch.object(tester, "_score_sample", side_effect=scores):
            result = tester._auto_score_samples(sample_checklist, raw_samples)

        # Should have entries for both samples
        assert "ref-0" in result
        assert "ref-1" in result
        # Each sample should have scores for all 5 questions
        assert len(result["ref-0"]) == 5
        assert len(result["ref-1"]) == 5


# ===========================================================================
# 7. InductiveGenerator Wiring
# ===========================================================================

class TestInductiveGeneratorWiring:
    """InductiveGenerator should wire the improvements into its generate() method."""

    def test_generate_passes_feedback_to_selector(self, sample_feedback):
        """Selector should receive the feedback list for coverage scoring."""
        gen = InductiveGenerator(max_questions=3)

        mock_questions = [
            {"question": f"Q{i}?", "source_feedback_indices": [i]}
            for i in range(6)
        ]

        with patch.object(gen, "_generate_questions", return_value=mock_questions):
            with patch("autochecklist.generators.corpus_level.inductive.Deduplicator") as mock_dedup:
                with patch("autochecklist.generators.corpus_level.inductive.Tagger") as mock_tag:
                    with patch("autochecklist.generators.corpus_level.inductive.Selector") as mock_sel:
                        mock_dedup.return_value.refine.side_effect = lambda x: x
                        mock_tag.return_value.refine.side_effect = lambda x: x
                        mock_sel.return_value.refine.side_effect = lambda x: x

                        gen.generate(observations=sample_feedback)

        # Selector should have been called with observations=sample_feedback
        sel_kwargs = mock_sel.call_args[1]
        assert "observations" in sel_kwargs
        assert sel_kwargs["observations"] == sample_feedback

    def test_generate_with_references_enables_unit_testing(self, sample_feedback):
        """Passing references + skip_unit_testing=False should run UnitTester."""
        gen = InductiveGenerator(max_questions=100)  # high to skip selection

        mock_questions = [
            {"question": f"Q{i}?", "source_feedback_indices": [i]}
            for i in range(4)
        ]
        references = ["Good response 1", "Good response 2"]

        with patch.object(gen, "_generate_questions", return_value=mock_questions):
            with patch("autochecklist.generators.corpus_level.inductive.Deduplicator") as mock_dedup:
                with patch("autochecklist.generators.corpus_level.inductive.Tagger") as mock_tag:
                    with patch("autochecklist.generators.corpus_level.inductive.UnitTester") as mock_ut:
                        mock_dedup.return_value.refine.side_effect = lambda x, **kw: x
                        mock_tag.return_value.refine.side_effect = lambda x: x
                        mock_ut.return_value.refine.side_effect = lambda x, **kw: x

                        gen.generate(
                            observations=sample_feedback,
                            references=references,
                            skip_unit_testing=False,
                            skip_selection=True,
                        )

        # UnitTester should have been created and called
        mock_ut.assert_called_once()
        mock_ut.return_value.refine.assert_called_once()
        # raw_samples should be passed
        refine_kwargs = mock_ut.return_value.refine.call_args[1]
        assert "raw_samples" in refine_kwargs

    def test_generate_without_references_skips_unit_testing(self, sample_feedback):
        """No references should skip unit testing even if skip_unit_testing=False."""
        gen = InductiveGenerator(max_questions=100)

        mock_questions = [
            {"question": f"Q{i}?", "source_feedback_indices": [i]}
            for i in range(4)
        ]

        with patch.object(gen, "_generate_questions", return_value=mock_questions):
            with patch("autochecklist.generators.corpus_level.inductive.Deduplicator") as mock_dedup:
                with patch("autochecklist.generators.corpus_level.inductive.Tagger") as mock_tag:
                    mock_dedup.return_value.refine.side_effect = lambda x: x
                    mock_tag.return_value.refine.side_effect = lambda x: x

                    # No references, skip_unit_testing=False
                    checklist = gen.generate(
                        observations=sample_feedback,
                        skip_unit_testing=False,
                        skip_selection=True,
                    )

        # Should complete without error, no UnitTester import needed
        assert isinstance(checklist, Checklist)

    def test_skip_unit_testing_default_true(self, sample_feedback):
        """skip_unit_testing should default to True."""
        gen = InductiveGenerator(max_questions=100)

        mock_questions = [
            {"question": "Q?", "source_feedback_indices": [0]}
        ]

        with patch.object(gen, "_generate_questions", return_value=mock_questions):
            # Even with references, should skip unit testing by default
            checklist = gen.generate(
                observations=sample_feedback,
                references=["ref1"],
                skip_selection=True,
                skip_dedup=True,
                skip_tagging=True,
            )

        assert isinstance(checklist, Checklist)


# ===========================================================================
# 8. Edge Cases
# ===========================================================================

class TestInductiveGeneratorEdgeCases:
    """Tests for edge cases."""

    def test_empty_feedback(self):
        """Should handle empty feedback list."""
        gen = InductiveGenerator()
        checklist = gen.generate(observations=[])

        assert len(checklist.items) == 0

    def test_single_feedback(self):
        """Should handle single feedback item."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = [
                {"question": "Is it good?", "source_feedback_indices": [0]}
            ]

            gen = InductiveGenerator()
            checklist = gen.generate(
                observations=["Single feedback item"],
                skip_dedup=True,
                skip_tagging=True,
                skip_selection=True,
            )

        assert len(checklist.items) == 1

    def test_custom_domain(self, sample_feedback, mock_generated_questions):
        """Should accept custom domain for prompt."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = InductiveGenerator()
            gen.generate(
                observations=sample_feedback,
                domain="customer support responses",
                skip_dedup=True,
                skip_tagging=True,
                skip_selection=True,
            )

        # Domain should be passed to generation
        call_args = mock_gen.call_args
        assert "domain" in call_args[1] or len(call_args[0]) > 1


# ===========================================================================
# 9. Metadata
# ===========================================================================

class TestInductiveGeneratorMetadata:
    """Tests for metadata tracking."""

    def test_tracks_generation_metadata(self, sample_feedback, mock_generated_questions):
        """Should track generation metadata."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = InductiveGenerator()
            checklist = gen.generate(
                observations=sample_feedback,
                skip_dedup=True,
                skip_tagging=True,
                skip_selection=True,
            )

        assert "observation_count" in checklist.metadata
        assert checklist.metadata["observation_count"] == len(sample_feedback)


# ===========================================================================
# 10. Verbose Mode
# ===========================================================================

class TestInductiveGeneratorVerboseMode:
    """Tests for verbose mode output."""

    def test_verbose_mode_runs_without_error(self, sample_feedback, mock_generated_questions, capsys):
        """Verbose mode should print progress and not crash."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            with patch("autochecklist.generators.corpus_level.inductive.Deduplicator") as mock_dedup:
                with patch("autochecklist.generators.corpus_level.inductive.Tagger") as mock_tag:
                    # Set up mocks to add metadata
                    def dedup_refine(checklist):
                        checklist.metadata["clusters_merged"] = 2
                        return checklist

                    def tag_refine(checklist):
                        checklist.metadata["filtered_count"] = 1
                        return checklist

                    mock_dedup.return_value.refine.side_effect = dedup_refine
                    mock_tag.return_value.refine.side_effect = tag_refine

                    gen = InductiveGenerator()
                    gen.generate(
                        observations=sample_feedback,
                        verbose=True,
                        skip_selection=True,
                    )

        captured = capsys.readouterr()
        assert "[InductiveGenerator]" in captured.out
        assert "Generated" in captured.out
        assert "Final checklist" in captured.out

    def test_verbose_false_produces_no_output(self, sample_feedback, mock_generated_questions, capsys):
        """Non-verbose mode should produce no output."""
        with patch.object(InductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = InductiveGenerator()
            gen.generate(
                observations=sample_feedback,
                verbose=False,
                skip_dedup=True,
                skip_tagging=True,
                skip_selection=True,
            )

        captured = capsys.readouterr()
        assert captured.out == ""
