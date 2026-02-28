"""Tests for DirectGenerator."""

import os
import pytest

from autochecklist.models import ChecklistResponse, WeightedChecklistResponse

requires_api = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)


class TestDirectGeneratorConfig:
    def test_tick_preset_loads(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator

        gen = DirectGenerator(method_name="tick")
        assert gen.method_name == "tick"
        assert gen.max_items == 8
        assert gen._format_name == "checklist"
        assert gen.temperature == 0.7

    def test_rocketeval_preset_loads(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator

        gen = DirectGenerator(method_name="rocketeval")
        assert gen.method_name == "rocketeval"
        assert gen._response_schema == ChecklistResponse
        assert gen.max_items == 10

    def test_rlcf_direct_preset_loads(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator

        gen = DirectGenerator(method_name="rlcf_direct")
        assert gen._response_schema == WeightedChecklistResponse
        assert gen._format_name == "weighted_checklist"
        assert gen.temperature == 0.0

    def test_unknown_method_without_path_raises(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator

        with pytest.raises(ValueError, match="Unknown method"):
            DirectGenerator(method_name="nonexistent")

    def test_custom_prompt_path(self, tmp_path):
        prompt_file = tmp_path / "test.md"
        prompt_file.write_text("Generate questions for: {input}")

        from autochecklist.generators.instance_level.direct import DirectGenerator

        gen = DirectGenerator(
            method_name="my_custom", custom_prompt=prompt_file
        )
        assert gen.method_name == "my_custom"

    def test_temperature_override(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator

        gen = DirectGenerator(method_name="tick", temperature=0.5)
        assert gen.temperature == 0.5

    def test_parse_structured_valid_json(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator

        gen = DirectGenerator(method_name="tick")
        raw = '{"questions": [{"question": "Is it a haiku?"}, {"question": "Has 3 lines?"}]}'
        items = gen._parse_structured(raw)
        assert len(items) == 2
        assert items[0].question == "Is it a haiku?"
        assert items[1].question == "Has 3 lines?"

    def test_parse_structured_weighted(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator

        gen = DirectGenerator(method_name="rlcf_direct")
        raw = '{"questions": [{"question": "Is it correct?", "weight": 90}, {"question": "Is it clear?", "weight": 60}]}'
        items = gen._parse_structured(raw)
        assert len(items) == 2
        assert items[0].weight == 90
        assert items[1].weight == 60

    def test_parse_structured_max_items(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator

        gen = DirectGenerator(method_name="tick")
        gen.max_items = 1
        raw = '{"questions": [{"question": "Q1?"}, {"question": "Q2?"}]}'
        items = gen._parse_structured(raw)
        assert len(items) == 1


class TestContrastiveGeneratorConfig:
    def test_rlcf_candidate_preset_loads(self):
        from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator

        gen = ContrastiveGenerator(
            method_name="rlcf_candidate",
            candidate_models=["openai/gpt-4o-mini"],
        )
        assert gen.method_name == "rlcf_candidate"
        assert gen._response_schema == WeightedChecklistResponse

    def test_rlcf_candidates_only_preset_loads(self):
        from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator

        gen = ContrastiveGenerator(
            method_name="rlcf_candidates_only",
            candidate_models=["openai/gpt-4o-mini"],
        )
        assert gen.method_name == "rlcf_candidates_only"

    def test_format_candidates(self):
        from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator

        gen = ContrastiveGenerator(
            method_name="rlcf_candidate",
            candidate_models=["openai/gpt-4o-mini"],
        )
        formatted = gen._format_candidates(["Response A", "Response B"])
        assert "### Candidate 1" in formatted
        assert "### Candidate 2" in formatted
        assert "Response A" in formatted
        assert "Response B" in formatted

    def test_no_candidates_or_models_raises(self):
        from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator

        gen = ContrastiveGenerator(method_name="rlcf_candidate")
        with pytest.raises(ValueError, match="requires 'candidates'"):
            gen.generate(input="Write a haiku.")
