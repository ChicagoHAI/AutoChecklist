"""Integration tests for InteractiveGenerator generator (real API calls).

These tests make actual API calls to verify end-to-end functionality.
Run with: pytest tests/test_generators/test_corpus_level/test_interacteval_integration.py -v
"""

import os
import pytest

from autochecklist import InteractiveGenerator, InteractiveInput


# Skip if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


@pytest.fixture
def coherence_attributes():
    """Think-aloud attributes for coherence evaluation."""
    return InteractiveInput(
        source="human_llm",
        dimension="coherence",
        attributes=[
            "Check if the summary maintains logical flow between sentences",
            "Ensure ideas connect smoothly without abrupt transitions",
            "Assess if the structure is well-organized with clear sections",
            "Verify the summary avoids jumping between unrelated topics",
            "Look for consistent point of view throughout the text",
            "Check that main points are clearly and concisely presented",
            "Ensure the summary builds to a coherent conclusion",
            "Verify there is a clear beginning, middle, and end",
            "Check for effective use of transition words",
            "Assess if the summary maintains a consistent narrative voice",
        ],
        sample_context="Summary evaluation for news articles",
    )


@pytest.fixture
def coherence_rubric():
    """Rubric for coherence dimension."""
    return (
        "Coherence measures the collective quality of all sentences. "
        "The summary should be well-structured and well-organized. "
        "The summary should not just be a heap of related information, "
        "but should build from sentence to sentence to a coherent body "
        "of information about a topic."
    )


class TestInteractiveGeneratorIntegration:
    """Integration tests for InteractiveGenerator with real API calls."""

    def test_generate_from_attributes(self, coherence_attributes, coherence_rubric):
        """Should generate checklist with correct metadata from attributes."""
        gen = InteractiveGenerator(model="openai/gpt-4o-mini")
        checklist = gen.generate(
            inputs=[coherence_attributes],
            rubric=coherence_rubric,
        )

        assert len(checklist.items) >= 1
        assert checklist.source_method == "interacteval"
        assert checklist.generation_level == "corpus"

    def test_pipeline_stages_produce_valid_output(
        self, coherence_attributes, coherence_rubric
    ):
        """Each pipeline stage should produce valid output."""
        gen = InteractiveGenerator(model="openai/gpt-4o-mini", max_components=3)

        # Stage 1: Component extraction
        components = gen._extract_components(
            coherence_attributes.attributes,
            coherence_rubric,
            "coherence",
        )
        assert len(components) >= 1
        assert len(components) <= 3

        # Stage 2: Attribute clustering
        clustered = gen._cluster_attributes(
            components, coherence_attributes.attributes
        )
        assert isinstance(clustered, dict)

        # Stage 3: Key question generation
        key_questions = gen._generate_key_questions(clustered, coherence_rubric, "coherence")
        assert isinstance(key_questions, dict)

        # Stage 4: Sub-question generation
        sub_questions = gen._generate_sub_questions(
            key_questions, coherence_rubric, "coherence"
        )
        assert isinstance(sub_questions, dict)

        # Stage 5: Question validation
        validated = gen._validate_questions(sub_questions, coherence_rubric, "coherence")
        assert isinstance(validated, list)

    def test_max_questions_limits_output(self, coherence_attributes, coherence_rubric):
        """max_questions should limit the number of items."""
        gen = InteractiveGenerator(model="openai/gpt-4o-mini")
        checklist = gen.generate(
            inputs=[coherence_attributes],
            rubric=coherence_rubric,
            max_questions=5,
        )

        assert len(checklist.items) <= 5

    def test_metadata_tracks_generation_info(
        self, coherence_attributes, coherence_rubric
    ):
        """Should track generation metadata."""
        gen = InteractiveGenerator(model="openai/gpt-4o-mini")
        checklist = gen.generate(
            inputs=[coherence_attributes],
            rubric=coherence_rubric,
        )

        assert "dimension" in checklist.metadata
        assert checklist.metadata["dimension"] == "coherence"
        assert "attribute_count" in checklist.metadata
        assert checklist.metadata["attribute_count"] == 10
        assert "source" in checklist.metadata

    def test_combines_multiple_inputs(self, coherence_rubric):
        """Should combine attributes from multiple sources."""
        human_input = InteractiveInput(
            source="human",
            dimension="coherence",
            attributes=[
                "Check for logical flow",
                "Ensure smooth transitions",
            ],
        )
        llm_input = InteractiveInput(
            source="llm",
            dimension="coherence",
            attributes=[
                "Verify consistent structure",
                "Assess narrative coherence",
            ],
        )

        gen = InteractiveGenerator(model="openai/gpt-4o-mini")
        checklist = gen.generate(
            inputs=[human_input, llm_input],
            rubric=coherence_rubric,
        )

        assert checklist.metadata["attribute_count"] == 4
        assert checklist.metadata["source"] == "human_llm"

    def test_empty_attributes_returns_empty_checklist(self, coherence_rubric):
        """Empty attributes should return empty checklist."""
        empty_input = InteractiveInput(
            source="human",
            dimension="coherence",
            attributes=[],
        )

        gen = InteractiveGenerator(model="openai/gpt-4o-mini")
        checklist = gen.generate(
            inputs=[empty_input],
            rubric=coherence_rubric,
        )

        assert len(checklist.items) == 0
