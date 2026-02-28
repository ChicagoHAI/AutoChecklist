"""Tests for reasoning_effort parameter threading through base classes and pipeline."""

from unittest.mock import MagicMock

from autochecklist.generators.base import ChecklistGenerator
from autochecklist.models import Checklist


# Minimal concrete subclass for testing base-class plumbing
class _StubGenerator(ChecklistGenerator):
    @property
    def generation_level(self):
        return "instance"

    @property
    def method_name(self):
        return "stub"

    def generate(self, **kwargs):
        return Checklist(items=[], source_method="stub", generation_level="instance")


class TestCallModelPassesReasoningEffort:
    """Verify _call_model threads reasoning_effort to the client."""

    def test_included_when_set(self):
        gen = _StubGenerator(model="openai/o3-mini", reasoning_effort="low")
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        gen._client = mock_client

        gen._call_model("hello")

        call_kwargs = mock_client.chat_completion.call_args
        assert call_kwargs.kwargs.get("reasoning_effort") == "low" or \
            call_kwargs[1].get("reasoning_effort") == "low", \
            f"reasoning_effort not found in call kwargs: {call_kwargs}"

    def test_absent_when_none(self):
        gen = _StubGenerator(model="openai/gpt-4o-mini")  # no reasoning_effort
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        gen._client = mock_client

        gen._call_model("hello")

        call_kwargs = mock_client.chat_completion.call_args
        # Should not be present at all (not even as None)
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "reasoning_effort" not in all_kwargs, \
            f"reasoning_effort should not be in kwargs: {all_kwargs}"


class TestPipelineThreadsAndOverrides:
    """Verify pipeline threads reasoning_effort to components with override support."""

    def test_shared_and_override(self):
        from autochecklist.pipeline import pipeline

        pipe = pipeline(
            "tick",
            generator_model="openai/o3-mini",
            scorer_model="openai/gpt-4o-mini",
            reasoning_effort="low",
            generator_kwargs={"reasoning_effort": "high"},
        )

        assert pipe.generator.reasoning_effort == "high", \
            "generator should use overridden reasoning_effort"
        assert pipe.scorer.reasoning_effort == "low", \
            "scorer should use shared reasoning_effort"
