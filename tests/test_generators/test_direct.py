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


class TestCategorizedParsing:
    def test_parse_structured_with_category(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator
        from autochecklist.models import CategorizedChecklistResponse

        gen = DirectGenerator(method_name="tick")
        gen._response_schema = CategorizedChecklistResponse
        raw = '{"questions": [{"question": "Does it follow instructions?", "category": "hard_rule"}, {"question": "Is reasoning clear?", "category": "principle"}]}'
        items = gen._parse_structured(raw)
        assert len(items) == 2
        assert items[0].category == "hard_rule"
        assert items[1].category == "principle"

    def test_parse_structured_without_category_unchanged(self):
        from autochecklist.generators.instance_level.direct import DirectGenerator

        gen = DirectGenerator(method_name="tick")
        raw = '{"questions": [{"question": "Is it a haiku?"}]}'
        items = gen._parse_structured(raw)
        assert len(items) == 1
        assert items[0].category is None


class TestCustomResponseSchema:
    def test_custom_schema_with_items_list_str(self):
        """Custom schema with 'items: List[str]' parses into ChecklistItems."""
        from pydantic import BaseModel
        from typing import List
        from autochecklist.generators.instance_level.direct import DirectGenerator

        class CustomResponse(BaseModel):
            items: List[str]

        gen = DirectGenerator(
            method_name="custom",
            custom_prompt="Generate criteria for: {input}",
            response_schema=CustomResponse,
        )
        assert gen._format_name is None

        raw = '{"items": ["Does the response address the topic?", "Is the tone appropriate?"]}'
        items = gen._parse_structured(raw)
        assert len(items) == 2
        assert items[0].question == "Does the response address the topic?"
        assert items[1].question == "Is the tone appropriate?"
        assert items[0].weight == 100.0
        assert items[0].category is None

    def test_nested_schema_extra_fields_in_metadata(self):
        """Nested item model with non-str fields preserved in metadata."""
        from pydantic import BaseModel
        from autochecklist.generators.instance_level.direct import DirectGenerator

        class ActionItem(BaseModel):
            item: str
            sources: list[int]

        class ActionItemsResponse(BaseModel):
            questions: list[ActionItem]

        gen = DirectGenerator(
            method_name="custom",
            custom_prompt="Generate criteria for: {input}",
            response_schema=ActionItemsResponse,
        )
        raw = '{"questions": [{"item": "Is it cited?", "sources": [1, 3]}]}'
        items = gen._parse_structured(raw)
        assert len(items) == 1
        assert items[0].question == "Is it cited?"
        assert items[0].metadata == {"sources": [1, 3]}


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


class TestCandidateRouting:
    """Test that candidates are routed to correct template placeholders."""

    def _make_generator(self, template_text):
        from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator
        gen = ContrastiveGenerator(
            method_name="custom",
            custom_prompt=template_text,
            generate_candidates=False,
        )
        return gen

    def test_dict_candidates_inject_chosen_rejected(self):
        gen = self._make_generator("Input: {input}\nChosen: {chosen}\nRejected: {rejected}\n{format_instructions}")
        from unittest.mock import MagicMock
        gen._call_model = MagicMock(return_value='{"questions": [{"question": "Q1?"}]}')
        gen.generate(input="Write a poem", candidates={"chosen": "Good poem", "rejected": "Bad poem"})
        prompt = gen._call_model.call_args[0][0]
        assert "Chosen: Good poem" in prompt
        assert "Rejected: Bad poem" in prompt

    def test_list_candidates_inject_responses(self):
        gen = self._make_generator("Input: {input}\nResponses:\n{responses}\n{format_instructions}")
        from unittest.mock import MagicMock
        gen._call_model = MagicMock(return_value='{"questions": [{"question": "Q1?"}]}')
        gen.generate(input="Write a poem", candidates=["Best poem", "Mid poem", "Worst poem"])
        prompt = gen._call_model.call_args[0][0]
        assert "Best poem" in prompt
        assert "Worst poem" in prompt

    def test_list_candidates_inject_candidates_rlcf(self):
        gen = self._make_generator("Input: {input}\n{candidates}\n{format_instructions}")
        from unittest.mock import MagicMock
        gen._call_model = MagicMock(return_value='{"questions": [{"question": "Q1?"}]}')
        gen.generate(input="Write a poem", candidates=["Resp A", "Resp B"])
        prompt = gen._call_model.call_args[0][0]
        assert "### Candidate 1" in prompt
        assert "Resp A" in prompt

    def test_dict_with_candidates_placeholder_raises(self):
        gen = self._make_generator("Input: {input}\n{candidates}\n{format_instructions}")
        with pytest.raises(ValueError):
            gen.generate(input="Write a poem", candidates={"chosen": "A", "rejected": "B"})

    def test_list_with_chosen_placeholder_raises(self):
        gen = self._make_generator("Input: {input}\n{chosen}\n{rejected}\n{format_instructions}")
        with pytest.raises(ValueError):
            gen.generate(input="Write a poem", candidates=["A", "B"])

    def test_dict_missing_keys_raises(self):
        gen = self._make_generator("Input: {input}\n{chosen}\n{rejected}\n{format_instructions}")
        with pytest.raises(ValueError):
            gen.generate(input="Write a poem", candidates={"chosen": "A"})

    def test_context_kwarg_injected(self):
        gen = self._make_generator("Input: {input}\nCtx: {context}\n{chosen}\n{rejected}\n{format_instructions}")
        from unittest.mock import MagicMock
        gen._call_model = MagicMock(return_value='{"questions": [{"question": "Q1?"}]}')
        gen.generate(input="Write a poem", candidates={"chosen": "A", "rejected": "B"}, context="Poetry contest")
        prompt = gen._call_model.call_args[0][0]
        assert "Ctx: Poetry contest" in prompt

    def test_context_defaults_empty(self):
        gen = self._make_generator("Input: {input}\nCtx: {context}\n{chosen}\n{rejected}\n{format_instructions}")
        from unittest.mock import MagicMock
        gen._call_model = MagicMock(return_value='{"questions": [{"question": "Q1?"}]}')
        gen.generate(input="Write a poem", candidates={"chosen": "A", "rejected": "B"})
        prompt = gen._call_model.call_args[0][0]
        assert "Ctx: \n" in prompt

    def test_metadata_count_dict(self):
        gen = self._make_generator("Input: {input}\n{chosen}\n{rejected}\n{format_instructions}")
        from unittest.mock import MagicMock
        gen._call_model = MagicMock(return_value='{"questions": [{"question": "Q1?"}]}')
        checklist = gen.generate(input="X", candidates={"chosen": "A", "rejected": "B"})
        assert checklist.metadata["num_candidates"] == 2

    def test_metadata_count_list(self):
        gen = self._make_generator("Input: {input}\n{candidates}\n{format_instructions}")
        from unittest.mock import MagicMock
        gen._call_model = MagicMock(return_value='{"questions": [{"question": "Q1?"}]}')
        checklist = gen.generate(input="X", candidates=["A", "B", "C"])
        assert checklist.metadata["num_candidates"] == 3


class TestOpenRubricsPresets:
    def test_openrubrics_pairwise_preset_loads(self):
        from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator
        from autochecklist.models import CategorizedChecklistResponse

        gen = ContrastiveGenerator(method_name="openrubrics_pairwise")
        assert gen.method_name == "openrubrics_pairwise"
        assert gen._response_schema == CategorizedChecklistResponse
        assert gen._format_name == "categorized_checklist"
        assert gen.temperature == 0.0
        assert gen.generate_candidates is False
        assert gen.max_items == 15

    def test_openrubrics_listwise_preset_loads(self):
        from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator
        from autochecklist.models import CategorizedChecklistResponse

        gen = ContrastiveGenerator(method_name="openrubrics_listwise")
        assert gen.method_name == "openrubrics_listwise"
        assert gen._response_schema == CategorizedChecklistResponse
        assert gen.generate_candidates is False
        assert gen.max_items == 20

    def test_openrubrics_pairwise_has_correct_placeholders(self):
        from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator

        gen = ContrastiveGenerator(method_name="openrubrics_pairwise")
        placeholders = gen._template._placeholders
        assert "input" in placeholders
        assert "chosen" in placeholders
        assert "rejected" in placeholders
        assert "context" in placeholders
        assert "format_instructions" in placeholders

    def test_openrubrics_listwise_has_correct_placeholders(self):
        from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator

        gen = ContrastiveGenerator(method_name="openrubrics_listwise")
        placeholders = gen._template._placeholders
        assert "input" in placeholders
        assert "responses" in placeholders
        assert "context" in placeholders
        assert "format_instructions" in placeholders
