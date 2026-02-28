"""Tests for DirectGenerator and custom prompt pipeline usage.

Integration tests use real LLM calls via OpenRouter.
Requires OPENROUTER_API_KEY environment variable.
"""

import os
from pathlib import Path

import pytest

from autochecklist import Checklist, ChecklistItem, pipeline, register_custom_generator, register_custom_scorer
from autochecklist.generators.instance_level.direct import DirectGenerator

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "custom_templates"


# --- Unit Tests (no API calls) ---


class TestDirectGeneratorUnit:
    """Unit tests for DirectGenerator that don't require API calls."""

    def test_loads_template_from_file(self):
        """DirectGenerator reads .md file and wraps it in a PromptTemplate."""
        gen = DirectGenerator(
            custom_prompt=FIXTURES_DIR / "generator.md",
        )
        assert gen.method_name == "custom"

    def test_custom_method_name(self):
        """DirectGenerator accepts a custom method_name."""
        gen = DirectGenerator(
            custom_prompt=FIXTURES_DIR / "generator.md",
            method_name="my_eval",
        )
        assert gen.method_name == "my_eval"

    def test_missing_file_raises(self):
        """DirectGenerator raises FileNotFoundError for missing .md file (Path)."""
        with pytest.raises(FileNotFoundError):
            DirectGenerator(custom_prompt=Path("/nonexistent/path.md"))

    def test_missing_placeholder_raises(self, tmp_path):
        """Template without {input} or {instruction} raises ValueError on generate()."""
        bad_template = tmp_path / "bad.md"
        bad_template.write_text("Generate questions about: {topic}")

        gen = DirectGenerator(custom_prompt=bad_template)
        with pytest.raises(ValueError, match="Missing template variables"):
            gen.generate(input="Write a haiku")


# --- Integration Tests (real API calls) ---

pytestmark_integration = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


@pytestmark_integration
class TestDirectGeneratorIntegration:
    """Integration tests for DirectGenerator with real API calls."""

    @pytest.fixture
    def custom_gen(self):
        return DirectGenerator(
            custom_prompt=FIXTURES_DIR / "generator.md",
            model="openai/gpt-4o-mini",
            temperature=0.3,
        )

    def test_generate_produces_valid_checklist(self, custom_gen):
        """DirectGenerator produces a Checklist with valid question items."""
        checklist = custom_gen.generate(input="Write a haiku about autumn.")

        assert isinstance(checklist, Checklist)
        assert checklist.source_method == "custom"
        assert checklist.generation_level == "instance"
        assert len(checklist.items) >= 1
        for item in checklist.items:
            assert isinstance(item, ChecklistItem)


@pytestmark_integration
class TestPipelineWithCustomPrompts:
    """Integration tests for pipeline() with custom generator/scorer prompts."""

    def test_pipeline_with_custom_generator(self):
        """pipeline() with registered custom generator runs full pipeline."""
        register_custom_generator("test_custom_gen", str(FIXTURES_DIR / "generator.md"))
        pipe = pipeline(
            "test_custom_gen",
            generator_model="openai/gpt-4o-mini",
            scorer_model="openai/gpt-4o-mini",
        )
        result = pipe(
            "Write a Python function that reverses a string.",
            target="def reverse(s): return s[::-1]",
        )

        assert result.checklist is not None
        assert len(result.checklist.items) >= 2
        assert result.score is not None
        assert 0.0 <= result.pass_rate <= 1.0

    def test_pipeline_with_custom_scorer(self):
        """pipeline() with registered custom scorer uses custom scoring prompt."""
        register_custom_scorer("test_custom_scorer", str(FIXTURES_DIR / "scorer.md"))
        pipe = pipeline(
            "tick",
            generator_model="openai/gpt-4o-mini",
            scorer="test_custom_scorer",
            scorer_model="openai/gpt-4o-mini",
        )
        result = pipe(
            "Write a greeting.",
            target="Hello! How are you doing today?",
        )

        assert result.score is not None
        assert 0.0 <= result.pass_rate <= 1.0
