"""Tests for DeductiveGenerator generator.

DeductiveGenerator generates checklists from evaluation dimensions with sub-aspects.

Key features:
- Takes dimension definitions with optional sub-dimensions
- Supports three augmentation modes: seed, elaboration, diversification
- Generates binary yes/no questions organized by sub-aspect
"""

import pytest
from unittest.mock import patch, MagicMock

from autochecklist.models import Checklist, DeductiveInput


from autochecklist.generators.corpus_level.deductive import DeductiveGenerator, AugmentationMode


@pytest.fixture
def sample_dimensions():
    """Sample dimension definitions for testing."""
    return [
        DeductiveInput(
            name="coherence",
            definition="The response should maintain logical flow and consistency.",
            sub_dimensions=["Logical Flow", "Consistency", "Relevance"],
        ),
        DeductiveInput(
            name="clarity",
            definition="The response should be clear and easy to understand.",
            sub_dimensions=["Language Clarity", "Structure"],
        ),
    ]


@pytest.fixture
def single_dimension():
    """Single dimension for simpler tests."""
    return DeductiveInput(
        name="relevance",
        definition="The response should be directly relevant to the query.",
        sub_dimensions=["Topic Relevance", "Detail Appropriateness"],
    )


@pytest.fixture
def mock_generated_questions():
    """Mock LLM response for question generation."""
    return {
        "questions": [
            {
                "question": "Does the response maintain a clear logical flow?",
                "sub_aspect": "Logical Flow",
                "dimension": "coherence",
            },
            {
                "question": "Is the response internally consistent?",
                "sub_aspect": "Consistency",
                "dimension": "coherence",
            },
            {
                "question": "Is the response relevant to the topic?",
                "sub_aspect": "Relevance",
                "dimension": "coherence",
            },
            {
                "question": "Is the language clear and unambiguous?",
                "sub_aspect": "Language Clarity",
                "dimension": "clarity",
            },
            {
                "question": "Is the response well-structured?",
                "sub_aspect": "Structure",
                "dimension": "clarity",
            },
        ]
    }


@pytest.fixture
def mock_elaborated_questions():
    """Mock LLM response for elaboration augmentation."""
    return {
        "questions": [
            # Original + elaborated versions
            {
                "question": "Does the response maintain a clear logical flow?",
                "sub_aspect": "Logical Flow",
                "dimension": "coherence",
            },
            {
                "question": "Does the response transition smoothly between ideas?",
                "sub_aspect": "Logical Flow",
                "dimension": "coherence",
            },
            {
                "question": "Are the arguments presented in a coherent order?",
                "sub_aspect": "Logical Flow",
                "dimension": "coherence",
            },
            {
                "question": "Is the response internally consistent?",
                "sub_aspect": "Consistency",
                "dimension": "coherence",
            },
            {
                "question": "Does the response avoid contradicting itself?",
                "sub_aspect": "Consistency",
                "dimension": "coherence",
            },
        ]
    }


class TestDeductiveGeneratorInit:
    """Tests for DeductiveGenerator initialization."""

    def test_default_model(self):
        """Should have default model."""
        gen = DeductiveGenerator()
        assert gen.model is not None

    def test_custom_model(self):
        """Should accept custom model."""
        gen = DeductiveGenerator(model="openai/gpt-4o")
        assert gen.model == "openai/gpt-4o"

    def test_default_augmentation_mode(self):
        """Should default to seed mode (minimal questions)."""
        gen = DeductiveGenerator()
        assert gen.augmentation_mode == AugmentationMode.SEED

    def test_custom_augmentation_mode(self):
        """Should accept custom augmentation mode."""
        gen = DeductiveGenerator(augmentation_mode=AugmentationMode.ELABORATION)
        assert gen.augmentation_mode == AugmentationMode.ELABORATION

    def test_augmentation_mode_string(self):
        """Should accept augmentation mode as string."""
        gen = DeductiveGenerator(augmentation_mode="diversification")
        assert gen.augmentation_mode == AugmentationMode.DIVERSIFICATION


class TestDeductiveGeneratorGeneration:
    """Tests for checklist generation from dimensions."""

    def test_generates_checklist_from_dimensions(
        self, single_dimension, mock_generated_questions
    ):
        """Should generate checklist from dimension definitions."""
        # Use only coherence questions for single dimension test
        coherence_questions = [
            q for q in mock_generated_questions["questions"]
            if q["dimension"] == "coherence"
        ]

        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = coherence_questions

            gen = DeductiveGenerator()
            checklist = gen.generate(dimensions=[single_dimension])

        assert isinstance(checklist, Checklist)
        assert len(checklist.items) == 3  # 3 coherence questions
        assert checklist.generation_level == "corpus"
        assert checklist.source_method == "checkeval"

    def test_generates_from_single_dimension(
        self, single_dimension, mock_generated_questions
    ):
        """Should handle single dimension input."""
        questions = [
            q
            for q in mock_generated_questions["questions"]
            if q["dimension"] == "coherence"
        ][:2]

        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = questions

            gen = DeductiveGenerator()
            checklist = gen.generate(dimensions=[single_dimension])

        assert isinstance(checklist, Checklist)
        assert len(checklist.items) >= 1

    def test_tracks_dimension_metadata(
        self, sample_dimensions, mock_generated_questions
    ):
        """Generated items should track dimension and sub-aspect."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = DeductiveGenerator()
            checklist = gen.generate(dimensions=sample_dimensions)

        for item in checklist.items:
            assert "dimension" in item.metadata
            assert "sub_aspect" in item.metadata
            assert item.metadata["dimension"] in ["coherence", "clarity"]

    def test_category_set_to_dimension(
        self, sample_dimensions, mock_generated_questions
    ):
        """Item category should be set to dimension name."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = DeductiveGenerator()
            checklist = gen.generate(dimensions=sample_dimensions)

        for item in checklist.items:
            assert item.category in ["coherence", "clarity"]


class TestDeductiveGeneratorAugmentation:
    """Tests for augmentation modes."""

    def test_seed_mode_minimal_questions(
        self, sample_dimensions, mock_generated_questions
    ):
        """Seed mode should generate minimal questions."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = DeductiveGenerator(augmentation_mode=AugmentationMode.SEED)
            checklist = gen.generate(dimensions=sample_dimensions)

        # Seed mode: ~1-3 questions per sub-aspect
        assert len(checklist.items) <= 15  # 5 sub-aspects * 3 max

    def test_elaboration_mode_more_questions(
        self, sample_dimensions, mock_elaborated_questions
    ):
        """Elaboration mode should generate more detailed questions."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_elaborated_questions["questions"]

            gen = DeductiveGenerator(augmentation_mode=AugmentationMode.ELABORATION)
            checklist = gen.generate(dimensions=sample_dimensions)

        # Elaboration adds more detailed questions
        assert len(checklist.items) >= 5

    def test_diversification_mode_alternative_framings(self, single_dimension):
        """Diversification mode should generate alternative framings."""
        diversified_questions = [
            {
                "question": "Does the response stay relevant?",
                "sub_aspect": "Topic Relevance",
                "dimension": "relevance",
            },
            {
                "question": "Does the response avoid irrelevant tangents?",
                "sub_aspect": "Topic Relevance",
                "dimension": "relevance",
            },
        ]

        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = diversified_questions

            gen = DeductiveGenerator(augmentation_mode=AugmentationMode.DIVERSIFICATION)
            checklist = gen.generate(dimensions=[single_dimension])

        assert len(checklist.items) == 2

    def test_augmentation_mode_passed_to_generation(self, sample_dimensions):
        """Augmentation mode should be passed to question generation."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = []

            gen = DeductiveGenerator(augmentation_mode=AugmentationMode.ELABORATION)
            gen.generate(dimensions=sample_dimensions)

        # Verify mode was passed
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("mode") == AugmentationMode.ELABORATION


class TestDeductiveGeneratorSeedQuestions:
    """Tests for using pre-defined seed questions."""

    def test_accepts_seed_questions_dict(self, sample_dimensions):
        """Should accept pre-defined seed questions per sub-aspect."""
        seed_questions = {
            "coherence": {
                "Logical Flow": ["Does the response flow logically?"],
                "Consistency": ["Is the response consistent?"],
                "Relevance": ["Is the response relevant?"],
            },
            "clarity": {
                "Language Clarity": ["Is the language clear?"],
                "Structure": ["Is the structure good?"],
            },
        }

        gen = DeductiveGenerator()
        checklist = gen.generate(
            dimensions=sample_dimensions,
            seed_questions=seed_questions,
            augment=False,  # Don't augment to get exact count
        )

        assert isinstance(checklist, Checklist)
        assert len(checklist.items) == 5

    def test_seed_questions_skip_llm_generation(self, single_dimension):
        """When seed questions provided for all dimensions, should skip LLM generation."""
        # Provide seed questions that cover the single dimension
        seed_questions = {
            "relevance": {
                "Topic Relevance": ["Is the response relevant to the topic?"],
                "Detail Appropriateness": ["Are the details appropriate?"],
            },
        }

        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            gen = DeductiveGenerator()
            gen.generate(
                dimensions=[single_dimension],
                seed_questions=seed_questions,
                augment=False,  # Don't augment
            )

        # LLM should not be called when seed questions cover all dimensions and no augmentation
        mock_gen.assert_not_called()

    def test_augment_from_seed_questions(self, single_dimension):
        """Should be able to augment from seed questions."""
        seed_questions = {
            "relevance": {
                "Topic Relevance": ["Is the response relevant?"],
            },
        }

        augmented = [
            {
                "question": "Is the response relevant?",
                "sub_aspect": "Topic Relevance",
                "dimension": "relevance",
            },
            {
                "question": "Does the response address the main topic?",
                "sub_aspect": "Topic Relevance",
                "dimension": "relevance",
            },
        ]

        with patch.object(DeductiveGenerator, "_augment_questions") as mock_aug:
            mock_aug.return_value = augmented

            gen = DeductiveGenerator(augmentation_mode=AugmentationMode.ELABORATION)
            checklist = gen.generate(
                dimensions=[single_dimension],
                seed_questions=seed_questions,
                augment=True,
            )

        mock_aug.assert_called_once()
        assert len(checklist.items) == 2


class TestDeductiveGeneratorTaskType:
    """Tests for different task types (summarization vs dialog)."""

    def test_accepts_task_type(self, sample_dimensions):
        """Should accept task type for context-specific generation."""
        gen = DeductiveGenerator(task_type="summarization")
        assert gen.task_type == "summarization"

    def test_task_type_affects_prompt(self, sample_dimensions):
        """Task type should be used in generation prompt."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = []

            gen = DeductiveGenerator(task_type="dialog")
            gen.generate(dimensions=sample_dimensions)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("task_type") == "dialog"


class TestDeductiveGeneratorEdgeCases:
    """Tests for edge cases."""

    def test_empty_dimensions(self):
        """Should handle empty dimensions list."""
        gen = DeductiveGenerator()
        checklist = gen.generate(dimensions=[])

        assert len(checklist.items) == 0

    def test_dimension_without_sub_dimensions(self):
        """Should handle dimension without sub-dimensions."""
        dimension = DeductiveInput(
            name="quality",
            definition="The response should be high quality.",
            sub_dimensions=None,
        )

        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = [
                {"question": "Is the response high quality?", "dimension": "quality"}
            ]

            gen = DeductiveGenerator()
            checklist = gen.generate(dimensions=[dimension])

        assert len(checklist.items) == 1

    def test_max_questions_parameter(self, sample_dimensions, mock_generated_questions):
        """Should respect max_questions limit."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = DeductiveGenerator()
            checklist = gen.generate(dimensions=sample_dimensions, max_questions=3)

        assert len(checklist.items) <= 3


class TestDeductiveGeneratorMetadata:
    """Tests for metadata tracking."""

    def test_tracks_generation_metadata(
        self, sample_dimensions, mock_generated_questions
    ):
        """Should track generation metadata."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = DeductiveGenerator()
            checklist = gen.generate(dimensions=sample_dimensions)

        assert "dimension_count" in checklist.metadata
        assert checklist.metadata["dimension_count"] == 2
        assert "augmentation_mode" in checklist.metadata

    def test_tracks_dimensions_in_metadata(
        self, sample_dimensions, mock_generated_questions
    ):
        """Should store dimension names in metadata."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = DeductiveGenerator()
            checklist = gen.generate(dimensions=sample_dimensions)

        assert "dimensions" in checklist.metadata
        assert "coherence" in checklist.metadata["dimensions"]
        assert "clarity" in checklist.metadata["dimensions"]


class TestDeductiveGeneratorQuestionFormat:
    """Tests for question format validation."""

    def test_questions_are_yes_no_format(
        self, sample_dimensions, mock_generated_questions
    ):
        """Generated questions should be in yes/no format."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = DeductiveGenerator()
            checklist = gen.generate(dimensions=sample_dimensions)

        for item in checklist.items:
            # Questions should end with ?
            assert item.question.endswith("?")
            # Questions should be answerable with yes/no (starts with common patterns)
            question_lower = item.question.lower()
            assert any(
                question_lower.startswith(word)
                for word in ["does", "is", "are", "can", "has", "will", "should", "do"]
            )


class TestDeductiveGeneratorFiltering:
    """Tests for DeductiveGenerator-specific filtering (Alignment + Dimension Consistency + Deduplication)."""

    def test_apply_filtering_parameter(self, sample_dimensions, mock_generated_questions):
        """Should apply DeductiveGenerator filtering and Deduplicator when apply_filtering=True."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            with patch.object(DeductiveGenerator, "_filter_questions") as mock_filter:
                with patch("autochecklist.generators.corpus_level.deductive.Deduplicator") as mock_dedup:
                    # Set up filter to return all items with stats
                    def filter_side_effect(items, dim_lookup, verbose):
                        return items, {"alignment_filtered": 0, "consistency_filtered": 0}

                    mock_filter.side_effect = filter_side_effect

                    mock_dedup_instance = MagicMock()
                    mock_dedup_instance.refine.side_effect = lambda x: x
                    mock_dedup.return_value = mock_dedup_instance

                    gen = DeductiveGenerator()
                    gen.generate(
                        dimensions=sample_dimensions,
                        apply_filtering=True,
                    )

        # Verify filtering and deduplication were called
        mock_filter.assert_called_once()
        mock_dedup.assert_called_once()
        mock_dedup_instance.refine.assert_called_once()

    def test_filtering_disabled_by_default(self, sample_dimensions, mock_generated_questions):
        """Filtering should be disabled by default."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            with patch.object(DeductiveGenerator, "_filter_questions") as mock_filter:
                with patch("autochecklist.generators.corpus_level.deductive.Deduplicator") as mock_dedup:
                    gen = DeductiveGenerator()
                    gen.generate(dimensions=sample_dimensions)

        # Filtering should NOT be called when filtering is disabled
        mock_filter.assert_not_called()
        mock_dedup.assert_not_called()

    def test_filtering_skipped_for_empty_checklist(self, sample_dimensions):
        """Filtering should be skipped if checklist is empty."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = []

            with patch.object(DeductiveGenerator, "_filter_questions") as mock_filter:
                gen = DeductiveGenerator()
                gen.generate(
                    dimensions=sample_dimensions,
                    apply_filtering=True,
                )

        # Filter should not be called for empty checklist
        mock_filter.assert_not_called()


class TestDeductiveGeneratorVerboseMode:
    """Tests for verbose mode output."""

    def test_verbose_mode_runs_without_error(
        self, sample_dimensions, mock_generated_questions, capsys
    ):
        """Verbose mode should print progress and not crash."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            with patch.object(DeductiveGenerator, "_filter_questions") as mock_filter:
                with patch("autochecklist.generators.corpus_level.deductive.Deduplicator") as mock_dedup:
                    # Set up filter to return items with stats
                    def filter_side_effect(items, dim_lookup, verbose):
                        return items, {"alignment_filtered": 1, "consistency_filtered": 0}

                    mock_filter.side_effect = filter_side_effect

                    def dedup_refine(checklist):
                        checklist.metadata["clusters_merged"] = 2
                        return checklist

                    mock_dedup.return_value.refine.side_effect = dedup_refine

                    gen = DeductiveGenerator()
                    gen.generate(
                        dimensions=sample_dimensions,
                        apply_filtering=True,
                        verbose=True,
                    )

        captured = capsys.readouterr()
        assert "[DeductiveGenerator]" in captured.out
        assert "Generated" in captured.out
        assert "Final checklist" in captured.out

    def test_verbose_with_filtering_shows_stages(
        self, sample_dimensions, mock_generated_questions, capsys
    ):
        """Verbose mode with filtering should show Filtering and Deduplication stages."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            with patch.object(DeductiveGenerator, "_filter_questions") as mock_filter:
                with patch("autochecklist.generators.corpus_level.deductive.Deduplicator") as mock_dedup:
                    def filter_side_effect(items, dim_lookup, verbose):
                        return items, {"alignment_filtered": 0, "consistency_filtered": 0}

                    mock_filter.side_effect = filter_side_effect
                    mock_dedup.return_value.refine.side_effect = lambda x: x

                    gen = DeductiveGenerator()
                    gen.generate(
                        dimensions=sample_dimensions,
                        apply_filtering=True,
                        verbose=True,
                    )

        captured = capsys.readouterr()
        assert "Filtering:" in captured.out
        assert "Deduplication:" in captured.out

    def test_verbose_false_produces_no_output(
        self, sample_dimensions, mock_generated_questions, capsys
    ):
        """Non-verbose mode should produce no output."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_generated_questions["questions"]

            gen = DeductiveGenerator()
            gen.generate(
                dimensions=sample_dimensions,
                verbose=False,
            )

        captured = capsys.readouterr()
        assert captured.out == ""


# --- Fixtures for COMBINED mode tests ---

@pytest.fixture
def mock_seed_questions():
    """Mock seed questions returned by _generate_questions in SEED mode."""
    return [
        {
            "question": "Does the response maintain a clear logical flow?",
            "sub_aspect": "Logical Flow",
            "dimension": "coherence",
        },
        {
            "question": "Is the response internally consistent?",
            "sub_aspect": "Consistency",
            "dimension": "coherence",
        },
    ]


@pytest.fixture
def mock_elaborated_from_seeds():
    """Mock elaborated questions returned by _augment_questions with ELABORATION mode."""
    return [
        {
            "question": "Does the response maintain a clear logical flow?",
            "sub_aspect": "Logical Flow",
            "dimension": "coherence",
        },
        {
            "question": "Does the response transition smoothly between ideas?",
            "sub_aspect": "Logical Flow",
            "dimension": "coherence",
        },
        {
            "question": "Is the response internally consistent?",
            "sub_aspect": "Consistency",
            "dimension": "coherence",
        },
        {
            "question": "Does the response avoid contradicting itself?",
            "sub_aspect": "Consistency",
            "dimension": "coherence",
        },
    ]


@pytest.fixture
def mock_diversified_from_seeds():
    """Mock diversified questions returned by _augment_questions with DIVERSIFICATION mode."""
    return [
        {
            "question": "Does the response maintain a clear logical flow?",
            "sub_aspect": "Logical Flow",
            "dimension": "coherence",
        },
        {
            "question": "Does the response present arguments in a coherent order?",
            "sub_aspect": "Logical Flow",
            "dimension": "coherence",
        },
        {
            "question": "Is the response internally consistent?",
            "sub_aspect": "Consistency",
            "dimension": "coherence",
        },
        {
            "question": "Is the response free from self-contradictions?",
            "sub_aspect": "Consistency",
            "dimension": "coherence",
        },
    ]


class TestCombinedAugmentationMode:
    """Tests for COMBINED augmentation mode (CheckEval paper faithfulness).

    The CheckEval paper runs both elaboration AND diversification independently
    from the same seeds, then merges all three pools before filtering.
    """

    def test_combined_mode_enum_exists(self):
        """COMBINED should be a valid AugmentationMode."""
        assert hasattr(AugmentationMode, "COMBINED")
        assert AugmentationMode.COMBINED.value == "combined"

    def test_combined_mode_init(self):
        """Should accept combined as augmentation mode."""
        gen = DeductiveGenerator(augmentation_mode="combined")
        assert gen.augmentation_mode == AugmentationMode.COMBINED

    def test_combined_mode_init_enum(self):
        """Should accept AugmentationMode.COMBINED."""
        gen = DeductiveGenerator(augmentation_mode=AugmentationMode.COMBINED)
        assert gen.augmentation_mode == AugmentationMode.COMBINED

    def test_combined_calls_seed_then_both_augmentations(
        self, single_dimension, mock_seed_questions,
        mock_elaborated_from_seeds, mock_diversified_from_seeds,
    ):
        """COMBINED should generate seeds, then call augmentation twice (elaboration + diversification)."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_seed_questions

            with patch.object(DeductiveGenerator, "_augment_questions") as mock_aug:
                mock_aug.side_effect = [
                    mock_elaborated_from_seeds,
                    mock_diversified_from_seeds,
                ]

                gen = DeductiveGenerator(augmentation_mode=AugmentationMode.COMBINED)
                gen.generate(dimensions=[single_dimension])

        # Seed generation should use SEED mode
        mock_gen.assert_called_once()
        gen_kwargs = mock_gen.call_args[1]
        assert gen_kwargs["mode"] == AugmentationMode.SEED

        # Augmentation should be called twice: once for elaboration, once for diversification
        assert mock_aug.call_count == 2
        aug_modes = [call[1]["mode"] for call in mock_aug.call_args_list]
        assert AugmentationMode.ELABORATION in aug_modes
        assert AugmentationMode.DIVERSIFICATION in aug_modes

    def test_combined_merges_all_pools(
        self, single_dimension, mock_seed_questions,
        mock_elaborated_from_seeds, mock_diversified_from_seeds,
    ):
        """COMBINED should merge seed + elaborated + diversified questions."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_seed_questions

            with patch.object(DeductiveGenerator, "_augment_questions") as mock_aug:
                mock_aug.side_effect = [
                    mock_elaborated_from_seeds,
                    mock_diversified_from_seeds,
                ]

                gen = DeductiveGenerator(augmentation_mode=AugmentationMode.COMBINED)
                checklist = gen.generate(dimensions=[single_dimension])

        # Total unique questions: seeds(2) + elaborated_new(2) + diversified_new(2) = 6
        # (the duplicate seed questions in elaborated/diversified are deduped)
        questions = [item.question for item in checklist.items]
        assert len(questions) == len(set(questions)), "Should have no duplicate questions"
        assert len(checklist.items) == 6

    def test_combined_deduplicates_across_pools(
        self, single_dimension, mock_seed_questions,
    ):
        """COMBINED should remove duplicate questions across the three pools."""
        # Elaborated returns same questions as seeds (all duplicates)
        elaborated_dupes = list(mock_seed_questions)
        # Diversified returns one new + one duplicate
        diversified_mixed = [
            mock_seed_questions[0],  # duplicate
            {
                "question": "A brand new question?",
                "sub_aspect": "Logical Flow",
                "dimension": "coherence",
            },
        ]

        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_seed_questions

            with patch.object(DeductiveGenerator, "_augment_questions") as mock_aug:
                mock_aug.side_effect = [elaborated_dupes, diversified_mixed]

                gen = DeductiveGenerator(augmentation_mode=AugmentationMode.COMBINED)
                checklist = gen.generate(dimensions=[single_dimension])

        # Seeds: 2, elaborated: 0 new, diversified: 1 new = 3 unique
        questions = [item.question for item in checklist.items]
        assert len(questions) == len(set(questions))
        assert len(checklist.items) == 3

    def test_combined_tracks_augmentation_source_metadata(
        self, single_dimension, mock_seed_questions,
        mock_elaborated_from_seeds, mock_diversified_from_seeds,
    ):
        """Each item should have augmentation_source in metadata."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_seed_questions

            with patch.object(DeductiveGenerator, "_augment_questions") as mock_aug:
                mock_aug.side_effect = [
                    mock_elaborated_from_seeds,
                    mock_diversified_from_seeds,
                ]

                gen = DeductiveGenerator(augmentation_mode=AugmentationMode.COMBINED)
                checklist = gen.generate(dimensions=[single_dimension])

        sources = [item.metadata.get("augmentation_source") for item in checklist.items]
        assert "seed" in sources
        assert "elaboration" in sources
        assert "diversification" in sources

    def test_combined_metadata_in_checklist(
        self, single_dimension, mock_seed_questions,
        mock_elaborated_from_seeds, mock_diversified_from_seeds,
    ):
        """Checklist metadata should record combined augmentation mode."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_seed_questions

            with patch.object(DeductiveGenerator, "_augment_questions") as mock_aug:
                mock_aug.side_effect = [
                    mock_elaborated_from_seeds,
                    mock_diversified_from_seeds,
                ]

                gen = DeductiveGenerator(augmentation_mode=AugmentationMode.COMBINED)
                checklist = gen.generate(dimensions=[single_dimension])

        assert checklist.metadata["augmentation_mode"] == "combined"

    def test_combined_with_seed_questions_and_augment(self, single_dimension):
        """COMBINED should work with user-provided seed questions + augment=True."""
        seed_questions = {
            "relevance": {
                "Topic Relevance": ["Is the response relevant?"],
                "Detail Appropriateness": ["Are the details appropriate?"],
            },
        }

        elaborated = [
            {"question": "Is the response relevant?", "sub_aspect": "Topic Relevance", "dimension": "relevance"},
            {"question": "Does the response address the main topic directly?", "sub_aspect": "Topic Relevance", "dimension": "relevance"},
            {"question": "Are the details appropriate?", "sub_aspect": "Detail Appropriateness", "dimension": "relevance"},
        ]
        diversified = [
            {"question": "Is the response relevant?", "sub_aspect": "Topic Relevance", "dimension": "relevance"},
            {"question": "Does the response stay on topic without digressions?", "sub_aspect": "Topic Relevance", "dimension": "relevance"},
            {"question": "Are the details appropriate?", "sub_aspect": "Detail Appropriateness", "dimension": "relevance"},
        ]

        with patch.object(DeductiveGenerator, "_augment_questions") as mock_aug:
            mock_aug.side_effect = [elaborated, diversified]

            gen = DeductiveGenerator(augmentation_mode=AugmentationMode.COMBINED)
            checklist = gen.generate(
                dimensions=[single_dimension],
                seed_questions=seed_questions,
                augment=True,
            )

        # Should call augment twice (elaboration + diversification)
        assert mock_aug.call_count == 2
        # Seeds(2) + elaborated_new(1) + diversified_new(1) = 4 unique
        assert len(checklist.items) == 4

    def test_combined_with_multiple_dimensions(
        self, sample_dimensions, mock_seed_questions,
    ):
        """COMBINED should work across multiple dimensions."""
        clarity_seeds = [
            {"question": "Is the language clear?", "sub_aspect": "Language Clarity", "dimension": "clarity"},
        ]

        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            # Return different seeds for each dimension
            mock_gen.side_effect = [mock_seed_questions, clarity_seeds]

            with patch.object(DeductiveGenerator, "_augment_questions") as mock_aug:
                # Return augmented for each (2 dims x 2 modes = 4 calls)
                mock_aug.side_effect = [
                    # coherence elaboration
                    mock_seed_questions + [{"question": "Elaborated coherence?", "sub_aspect": "Logical Flow", "dimension": "coherence"}],
                    # coherence diversification
                    mock_seed_questions + [{"question": "Diversified coherence?", "sub_aspect": "Logical Flow", "dimension": "coherence"}],
                    # clarity elaboration
                    clarity_seeds + [{"question": "Elaborated clarity?", "sub_aspect": "Language Clarity", "dimension": "clarity"}],
                    # clarity diversification
                    clarity_seeds + [{"question": "Diversified clarity?", "sub_aspect": "Language Clarity", "dimension": "clarity"}],
                ]

                gen = DeductiveGenerator(augmentation_mode=AugmentationMode.COMBINED)
                checklist = gen.generate(dimensions=sample_dimensions)

        # _generate_questions called once per dimension (SEED)
        assert mock_gen.call_count == 2
        # _augment_questions called 2x per dimension (elab + div)
        assert mock_aug.call_count == 4
        # All items present (deduped)
        questions = [item.question for item in checklist.items]
        assert len(questions) == len(set(questions))

    def test_combined_with_apply_filtering(
        self, single_dimension, mock_seed_questions,
        mock_elaborated_from_seeds, mock_diversified_from_seeds,
    ):
        """COMBINED should work with apply_filtering=True."""
        with patch.object(DeductiveGenerator, "_generate_questions") as mock_gen:
            mock_gen.return_value = mock_seed_questions

            with patch.object(DeductiveGenerator, "_augment_questions") as mock_aug:
                mock_aug.side_effect = [
                    mock_elaborated_from_seeds,
                    mock_diversified_from_seeds,
                ]

                with patch.object(DeductiveGenerator, "_filter_questions") as mock_filter:
                    with patch("autochecklist.generators.corpus_level.deductive.Deduplicator") as mock_dedup:
                        mock_filter.side_effect = lambda items, dim_lookup, verbose: (
                            items, {"alignment_filtered": 0, "consistency_filtered": 0}
                        )
                        mock_dedup.return_value.refine.side_effect = lambda x: x

                        gen = DeductiveGenerator(augmentation_mode=AugmentationMode.COMBINED)
                        checklist = gen.generate(
                            dimensions=[single_dimension],
                            apply_filtering=True,
                        )

        mock_filter.assert_called_once()
        assert len(checklist.items) > 0
