"""Integration tests for OpenRubrics CRG presets (requires OPENROUTER_API_KEY)."""

import os
import pytest

from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator

requires_api = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)

SAMPLE_INPUT = "Explain the concept of recursion in programming."

SAMPLE_CHOSEN = (
    "Recursion is when a function calls itself to solve a smaller instance of the "
    "same problem. It requires a base case to stop the recursion and a recursive "
    "case that breaks the problem down. For example, calculating factorial: "
    "factorial(n) = n * factorial(n-1), with factorial(0) = 1 as the base case."
)

SAMPLE_REJECTED = (
    "Recursion is a programming thing where code repeats. It's like a loop but "
    "different. You just call the function again and it works."
)


@requires_api
class TestOpenRubricsPairwiseIntegration:
    def test_generates_categorized_items(self):
        gen = ContrastiveGenerator(
            method_name="openrubrics_pairwise",
            model="openai/gpt-4o-mini",
        )

        checklist = gen.generate(
            input=SAMPLE_INPUT,
            candidates={"chosen": SAMPLE_CHOSEN, "rejected": SAMPLE_REJECTED},
        )

        assert len(checklist.items) >= 1
        for item in checklist.items:
            assert item.category in ("hard_rule", "principle")
            assert "[Hard Rule]" not in item.question
            assert "[Principle]" not in item.question

        grouped = checklist.by_category()
        for key in grouped:
            assert key in ("hard_rule", "principle")

    def test_metadata_populated(self):
        gen = ContrastiveGenerator(
            method_name="openrubrics_pairwise",
            model="openai/gpt-4o-mini",
        )

        checklist = gen.generate(
            input=SAMPLE_INPUT,
            candidates={"chosen": SAMPLE_CHOSEN, "rejected": SAMPLE_REJECTED},
        )

        assert checklist.metadata["num_candidates"] == 2
        assert "raw_response" in checklist.metadata


@requires_api
class TestOpenRubricsListwiseIntegration:
    def test_generates_categorized_items(self):
        gen = ContrastiveGenerator(
            method_name="openrubrics_listwise",
            model="openai/gpt-4o-mini",
        )

        checklist = gen.generate(
            input=SAMPLE_INPUT,
            candidates=[SAMPLE_CHOSEN, SAMPLE_REJECTED],
        )

        assert len(checklist.items) >= 1
        for item in checklist.items:
            assert item.category in ("hard_rule", "principle")
