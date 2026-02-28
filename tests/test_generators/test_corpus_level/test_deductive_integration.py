"""Integration tests for DeductiveGenerator generator (real API calls).

These tests make actual API calls to verify end-to-end functionality.
Run with: pytest tests/test_generators/test_corpus_level/test_checkeval_integration.py -v
"""

import os
import pytest

from autochecklist import DeductiveGenerator, AugmentationMode, DeductiveInput


# Skip if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


@pytest.fixture
def coherence_dimension():
    """A single coherence dimension for testing."""
    return DeductiveInput(
        name="coherence",
        definition="The response should maintain logical flow and consistency throughout.",
        sub_dimensions=["Logical Flow", "Consistency"],
    )


@pytest.fixture
def multiple_dimensions():
    """Multiple dimensions for testing."""
    return [
        DeductiveInput(
            name="clarity",
            definition="The response should be clear and easy to understand.",
            sub_dimensions=["Language Clarity", "Structure"],
        ),
        DeductiveInput(
            name="relevance",
            definition="The response should be directly relevant to the query.",
            sub_dimensions=["Topic Relevance"],
        ),
    ]


class TestDeductiveGeneratorIntegration:
    """Integration tests for DeductiveGenerator with real API calls."""

    def test_generate_from_single_dimension(self, coherence_dimension):
        """Should generate checklist with correct metadata from a single dimension."""
        gen = DeductiveGenerator(model="openai/gpt-4o-mini")
        checklist = gen.generate(dimensions=[coherence_dimension])

        assert len(checklist.items) >= 1
        assert checklist.source_method == "checkeval"
        assert checklist.generation_level == "corpus"
        for item in checklist.items:
            assert "dimension" in item.metadata

    def test_generate_from_multiple_dimensions(self, multiple_dimensions):
        """Should generate checklist with dimension metadata from multiple dimensions."""
        gen = DeductiveGenerator(model="openai/gpt-4o-mini")
        checklist = gen.generate(dimensions=multiple_dimensions)

        assert len(checklist.items) >= 1
        dimensions_found = set(item.metadata.get("dimension") for item in checklist.items)
        assert len(dimensions_found) >= 1

    def test_seed_questions_used_directly(self, coherence_dimension):
        """Should use seed questions directly without LLM call."""
        seed_questions = {
            "coherence": {
                "Logical Flow": [
                    "Does the response maintain a clear logical flow?",
                    "Are the ideas connected smoothly?",
                ],
                "Consistency": [
                    "Is the response internally consistent?",
                ],
            },
        }

        gen = DeductiveGenerator(model="openai/gpt-4o-mini")
        checklist = gen.generate(
            dimensions=[coherence_dimension],
            seed_questions=seed_questions,
            augment=False,
        )

        # Should have exactly the seed questions
        assert len(checklist.items) == 3
        questions = [item.question for item in checklist.items]
        assert "Does the response maintain a clear logical flow?" in questions

    def test_augment_from_seed_questions(self, coherence_dimension):
        """Should run augmentation without error when augment=True."""
        seed_questions = {
            "coherence": {
                "Logical Flow": [
                    "Does the response flow logically?",
                ],
            },
        }

        gen = DeductiveGenerator(
            model="openai/gpt-4o-mini",
            augmentation_mode=AugmentationMode.ELABORATION,
        )
        checklist = gen.generate(
            dimensions=[coherence_dimension],
            seed_questions=seed_questions,
            augment=True,
        )

        assert len(checklist.items) >= 1

    def test_task_type_in_metadata(self, coherence_dimension):
        """Task type should be tracked in metadata."""
        gen = DeductiveGenerator(
            model="openai/gpt-4o-mini",
            task_type="summarization",
        )
        checklist = gen.generate(dimensions=[coherence_dimension])

        assert len(checklist.items) >= 1
        assert checklist.metadata["task_type"] == "summarization"

    def test_max_questions_limits_output(self, multiple_dimensions):
        """max_questions should limit the number of items."""
        gen = DeductiveGenerator(model="openai/gpt-4o-mini")
        checklist = gen.generate(
            dimensions=multiple_dimensions,
            max_questions=5,
        )

        assert len(checklist.items) <= 5

    def test_metadata_tracks_generation_info(self, multiple_dimensions):
        """Should track generation metadata."""
        gen = DeductiveGenerator(
            model="openai/gpt-4o-mini",
            augmentation_mode=AugmentationMode.SEED,
            task_type="general",
        )
        checklist = gen.generate(dimensions=multiple_dimensions)

        assert "dimension_count" in checklist.metadata
        assert checklist.metadata["dimension_count"] == 2
        assert "dimensions" in checklist.metadata
        assert "augmentation_mode" in checklist.metadata
        assert checklist.metadata["augmentation_mode"] == "seed"
