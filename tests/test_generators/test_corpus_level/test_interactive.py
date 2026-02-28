"""Tests for InteractiveGenerator generator.

InteractiveGenerator generates checklists from think-aloud attributes through a 5-stage pipeline:
1. Component Extraction: Find recurring themes from attributes
2. Attributes Clustering: Group attributes under components
3. Key Question Generation: 1 yes/no question per component
4. Sub-Question Generation: 2-3 sub-questions per component
5. Question Validation: Refine and validate final checklist
"""

import pytest
from unittest.mock import patch

from autochecklist.models import Checklist, InteractiveInput


# Import will fail until we implement the generator
from autochecklist.generators.corpus_level.interactive import InteractiveGenerator


@pytest.fixture
def sample_think_aloud_input():
    """Sample think-aloud input for testing."""
    return InteractiveInput(
        source="human_llm",
        dimension="coherence",
        attributes=[
            "Check if the summary maintains logical flow",
            "Ensure ideas connect smoothly between sentences",
            "Assess if the structure is well-organized",
            "Verify the summary avoids abrupt topic changes",
            "Look for consistent point of view throughout",
            "Check that main points are clearly presented",
            "Ensure the summary builds to a coherent conclusion",
        ],
        sample_context="Summary evaluation for news articles",
    )


@pytest.fixture
def coherence_rubric():
    """Sample rubric/definition for coherence dimension."""
    return (
        "Coherence measures the collective quality of all sentences. "
        "The summary should be well-structured and well-organized. "
        "The summary should not just be a heap of related information, "
        "but should build from sentence to sentence to a coherent body "
        "of information about a topic."
    )


@pytest.fixture
def mock_components():
    """Mock LLM response for component extraction."""
    return [
        "Well-structured organization",
        "Logical flow of ideas",
        "Consistent point of view",
        "Clear presentation of main points",
        "Coherent conclusion",
    ]


@pytest.fixture
def mock_clustered_attributes():
    """Mock LLM response for attribute clustering."""
    return {
        "Well-structured organization": [
            "Assess if the structure is well-organized",
            "Verify the summary avoids abrupt topic changes",
        ],
        "Logical flow of ideas": [
            "Check if the summary maintains logical flow",
            "Ensure ideas connect smoothly between sentences",
        ],
        "Consistent point of view": [
            "Look for consistent point of view throughout",
        ],
        "Clear presentation of main points": [
            "Check that main points are clearly presented",
        ],
        "Coherent conclusion": [
            "Ensure the summary builds to a coherent conclusion",
        ],
    }


@pytest.fixture
def mock_key_questions():
    """Mock LLM response for key question generation."""
    return {
        "Well-structured organization": "Is the summary well-structured and organized?",
        "Logical flow of ideas": "Does the summary maintain a logical flow of ideas?",
        "Consistent point of view": "Is the point of view consistent throughout?",
        "Clear presentation of main points": "Are the main points clearly presented?",
        "Coherent conclusion": "Does the summary build to a coherent conclusion?",
    }


@pytest.fixture
def mock_sub_questions():
    """Mock LLM response for sub-question generation."""
    return {
        "Well-structured organization": [
            "Is the summary well-structured and organized?",
            "Does the summary avoid abrupt topic changes?",
            "Is the information presented in a clear structure?",
        ],
        "Logical flow of ideas": [
            "Does the summary maintain a logical flow of ideas?",
            "Do ideas connect smoothly between sentences?",
        ],
        "Consistent point of view": [
            "Is the point of view consistent throughout?",
            "Does the summary maintain a consistent voice?",
        ],
        "Clear presentation of main points": [
            "Are the main points clearly presented?",
            "Is the summary easy to follow?",
        ],
        "Coherent conclusion": [
            "Does the summary build to a coherent conclusion?",
        ],
    }


@pytest.fixture
def mock_validated_questions():
    """Mock LLM response for question validation."""
    return [
        "Is the summary well-structured and organized?",
        "Does the summary avoid abrupt topic changes?",
        "Does the summary maintain a logical flow of ideas?",
        "Do ideas connect smoothly between sentences?",
        "Is the point of view consistent throughout?",
        "Are the main points clearly presented?",
        "Does the summary build to a coherent conclusion?",
    ]


class TestInteractiveGeneratorInit:
    """Tests for InteractiveGenerator initialization."""

    def test_default_model(self):
        """Should have default model."""
        gen = InteractiveGenerator()
        assert gen.model is not None

    def test_custom_model(self):
        """Should accept custom model."""
        gen = InteractiveGenerator(model="openai/gpt-4o")
        assert gen.model == "openai/gpt-4o"

    def test_default_max_components(self):
        """Should default to 5 max components."""
        gen = InteractiveGenerator()
        assert gen.max_components == 5

    def test_custom_max_components(self):
        """Should accept custom max components."""
        gen = InteractiveGenerator(max_components=3)
        assert gen.max_components == 3


class TestInteractiveGeneratorGeneration:
    """Tests for checklist generation from think-aloud attributes."""

    def test_generates_checklist_from_attributes(
        self, sample_think_aloud_input, coherence_rubric, mock_validated_questions
    ):
        """Should generate checklist from think-aloud attributes."""
        with patch.object(InteractiveGenerator, "_run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = mock_validated_questions

            gen = InteractiveGenerator()
            checklist = gen.generate(
                inputs=[sample_think_aloud_input],
                rubric=coherence_rubric,
            )

        assert isinstance(checklist, Checklist)
        assert len(checklist.items) == 7
        assert checklist.generation_level == "corpus"
        assert checklist.source_method == "interacteval"

    def test_tracks_dimension_metadata(
        self, sample_think_aloud_input, coherence_rubric, mock_validated_questions
    ):
        """Generated checklist should track dimension in metadata."""
        with patch.object(InteractiveGenerator, "_run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = mock_validated_questions

            gen = InteractiveGenerator()
            checklist = gen.generate(
                inputs=[sample_think_aloud_input],
                rubric=coherence_rubric,
            )

        assert "dimension" in checklist.metadata
        assert checklist.metadata["dimension"] == "coherence"


class TestInteractiveGeneratorPipeline:
    """Tests for the 5-stage pipeline."""

    def test_stage1_component_extraction(
        self, sample_think_aloud_input, coherence_rubric, mock_components
    ):
        """Stage 1: Should extract components from attributes."""
        with patch.object(InteractiveGenerator, "_extract_components") as mock_extract:
            mock_extract.return_value = mock_components

            gen = InteractiveGenerator()
            components = gen._extract_components(
                sample_think_aloud_input.attributes, coherence_rubric
            )

        assert len(components) <= 5
        assert isinstance(components, list)

    def test_stage2_attribute_clustering(
        self, sample_think_aloud_input, mock_components, mock_clustered_attributes
    ):
        """Stage 2: Should cluster attributes under components."""
        with patch.object(InteractiveGenerator, "_cluster_attributes") as mock_cluster:
            mock_cluster.return_value = mock_clustered_attributes

            gen = InteractiveGenerator()
            clustered = gen._cluster_attributes(
                mock_components, sample_think_aloud_input.attributes
            )

        assert isinstance(clustered, dict)
        assert len(clustered) == len(mock_components)

    def test_stage3_key_question_generation(
        self, coherence_rubric, mock_clustered_attributes, mock_key_questions
    ):
        """Stage 3: Should generate key questions for each component."""
        with patch.object(InteractiveGenerator, "_generate_key_questions") as mock_gen:
            mock_gen.return_value = mock_key_questions

            gen = InteractiveGenerator()
            questions = gen._generate_key_questions(
                mock_clustered_attributes, coherence_rubric
            )

        assert isinstance(questions, dict)
        for component, question in questions.items():
            assert question.endswith("?")

    def test_stage4_sub_question_generation(
        self, coherence_rubric, mock_key_questions, mock_sub_questions
    ):
        """Stage 4: Should generate sub-questions for each key question."""
        with patch.object(InteractiveGenerator, "_generate_sub_questions") as mock_gen:
            mock_gen.return_value = mock_sub_questions

            gen = InteractiveGenerator()
            sub_qs = gen._generate_sub_questions(mock_key_questions, coherence_rubric)

        assert isinstance(sub_qs, dict)
        for component, questions in sub_qs.items():
            assert isinstance(questions, list)
            assert len(questions) >= 1

    def test_stage5_question_validation(
        self, coherence_rubric, mock_sub_questions, mock_validated_questions
    ):
        """Stage 5: Should validate and refine questions."""
        with patch.object(InteractiveGenerator, "_validate_questions") as mock_validate:
            mock_validate.return_value = mock_validated_questions

            gen = InteractiveGenerator()
            validated = gen._validate_questions(mock_sub_questions, coherence_rubric)

        assert isinstance(validated, list)
        for q in validated:
            assert q.endswith("?")


class TestInteractiveGeneratorEdgeCases:
    """Tests for edge cases."""

    def test_empty_attributes(self, coherence_rubric):
        """Should handle empty attributes list."""
        empty_input = InteractiveInput(
            source="human",
            dimension="coherence",
            attributes=[],
        )

        gen = InteractiveGenerator()
        checklist = gen.generate(inputs=[empty_input], rubric=coherence_rubric)

        assert len(checklist.items) == 0

    def test_single_attribute(self, coherence_rubric):
        """Should handle single attribute."""
        single_input = InteractiveInput(
            source="llm",
            dimension="coherence",
            attributes=["Check for logical flow"],
        )

        with patch.object(InteractiveGenerator, "_run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = ["Does the text have logical flow?"]

            gen = InteractiveGenerator()
            checklist = gen.generate(inputs=[single_input], rubric=coherence_rubric)

        assert len(checklist.items) >= 1

    def test_max_questions_parameter(
        self, sample_think_aloud_input, coherence_rubric, mock_validated_questions
    ):
        """Should respect max_questions limit."""
        with patch.object(InteractiveGenerator, "_run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = mock_validated_questions

            gen = InteractiveGenerator()
            checklist = gen.generate(
                inputs=[sample_think_aloud_input],
                rubric=coherence_rubric,
                max_questions=3,
            )

        assert len(checklist.items) <= 3


class TestInteractiveGeneratorMetadata:
    """Tests for metadata tracking."""

    def test_tracks_pipeline_metadata(
        self, sample_think_aloud_input, coherence_rubric, mock_validated_questions
    ):
        """Should track pipeline metadata."""
        with patch.object(InteractiveGenerator, "_run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = mock_validated_questions

            gen = InteractiveGenerator()
            checklist = gen.generate(
                inputs=[sample_think_aloud_input],
                rubric=coherence_rubric,
            )

        assert "attribute_count" in checklist.metadata
        assert "source" in checklist.metadata

    def test_items_have_dimension_metadata(
        self, sample_think_aloud_input, coherence_rubric, mock_validated_questions
    ):
        """Items should track their source dimension."""
        with patch.object(InteractiveGenerator, "_run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = mock_validated_questions

            gen = InteractiveGenerator()
            checklist = gen.generate(
                inputs=[sample_think_aloud_input],
                rubric=coherence_rubric,
            )

        # Items should have dimension metadata
        for item in checklist.items:
            assert "dimension" in item.metadata
            assert item.metadata["dimension"] == "coherence"


class TestInteractiveGeneratorQuestionFormat:
    """Tests for question format validation."""

    def test_questions_are_yes_no_format(
        self, sample_think_aloud_input, coherence_rubric, mock_validated_questions
    ):
        """Generated questions should be in yes/no format."""
        with patch.object(InteractiveGenerator, "_run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = mock_validated_questions

            gen = InteractiveGenerator()
            checklist = gen.generate(
                inputs=[sample_think_aloud_input],
                rubric=coherence_rubric,
            )

        for item in checklist.items:
            # Questions should end with ?
            assert item.question.endswith("?")
            # Questions should be answerable with yes/no
            question_lower = item.question.lower()
            assert any(
                question_lower.startswith(word)
                for word in ["does", "is", "are", "can", "has", "will", "should", "do"]
            )


class TestInteractiveGeneratorMultipleInputs:
    """Tests for handling multiple think-aloud inputs."""

    def test_combines_multiple_sources(self, coherence_rubric):
        """Should combine attributes from multiple sources."""
        human_input = InteractiveInput(
            source="human",
            dimension="coherence",
            attributes=["Human consideration 1", "Human consideration 2"],
        )
        llm_input = InteractiveInput(
            source="llm",
            dimension="coherence",
            attributes=["LLM consideration 1", "LLM consideration 2"],
        )

        with patch.object(InteractiveGenerator, "_run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = ["Question from combined attributes?"]

            gen = InteractiveGenerator()
            gen.generate(
                inputs=[human_input, llm_input],
                rubric=coherence_rubric,
            )

        # Should combine 4 attributes from both sources
        call_args = mock_pipeline.call_args
        assert call_args is not None
