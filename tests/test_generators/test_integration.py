"""Integration tests for all pipeline presets via pipeline() factory."""

import os
import pytest

from autochecklist import pipeline

requires_api = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)


@requires_api
class TestAllPipelinePresets:
    """Test that all pipeline presets generate valid checklists."""

    def test_tick(self):
        pipe = pipeline("tick", generator_model="openai/gpt-4o-mini")
        checklist = pipe.generate(input="Write a haiku about autumn.")
        assert len(checklist.items) >= 1
        assert checklist.source_method == "tick"

    def test_rocketeval(self):
        pipe = pipeline("rocketeval", generator_model="openai/gpt-4o-mini")
        checklist = pipe.generate(
            input="What is the capital of France?",
            reference="The capital of France is Paris.",
        )
        assert len(checklist.items) >= 1
        assert checklist.source_method == "rocketeval"

    def test_rlcf_direct(self):
        pipe = pipeline("rlcf_direct", generator_model="openai/gpt-4o-mini")
        checklist = pipe.generate(
            input="Write a haiku about autumn.",
            reference="Leaves fall gently down / Whisper autumn farewell song / Nature quilt unfolds",
        )
        assert len(checklist.items) >= 1
        assert checklist.source_method == "rlcf_direct"


@requires_api
class TestRegistryDiscovery:
    """Test that all generators are discoverable via registry."""

    def test_get_generator_instantiates(self):
        from autochecklist import get_generator

        gen_cls = get_generator("tick")
        gen = gen_cls(model="openai/gpt-4o-mini")
        assert gen.method_name == "tick"
        assert gen.generation_level == "instance"
