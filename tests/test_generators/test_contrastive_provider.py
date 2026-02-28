"""Tests for ContrastiveGenerator separate candidate provider support."""

import sys
from unittest.mock import patch, MagicMock


from autochecklist.generators.instance_level.contrastive import ContrastiveGenerator

# autochecklist.pipeline (module) is shadowed by the pipeline() function in __init__.py,
# so we reference the module explicitly for patching
_pipeline_module = sys.modules["autochecklist.pipeline"]


class TestGetCandidateClient:
    """Tests for _get_candidate_client() method."""

    @patch("autochecklist.generators.instance_level.contrastive.get_client")
    def test_returns_separate_client_when_candidate_params_set(self, mock_get_client):
        """When candidate_provider is set, _get_candidate_client creates a new client."""
        mock_candidate_client = MagicMock(name="candidate_client")
        mock_get_client.return_value = mock_candidate_client

        gen = ContrastiveGenerator(
            method_name="rlcf_candidate",
            model="openai/gpt-4o-mini",
            candidate_models=["openai/gpt-3.5-turbo"],
            candidate_provider="vllm",
            candidate_base_url="http://localhost:8000/v1",
        )

        client = gen._get_candidate_client()

        assert client is mock_candidate_client
        mock_get_client.assert_called_once_with(
            provider="vllm",
            base_url="http://localhost:8000/v1",
            api_key=None,
            model="openai/gpt-4o-mini",
            api_format=None,
        )

    @patch("autochecklist.generators.instance_level.contrastive.get_client")
    def test_candidate_api_key_and_format_threaded(self, mock_get_client):
        """All candidate_* params are threaded through to get_client."""
        mock_get_client.return_value = MagicMock()

        gen = ContrastiveGenerator(
            method_name="rlcf_candidate",
            model="openai/gpt-4o-mini",
            candidate_models=["openai/gpt-3.5-turbo"],
            candidate_provider="openai",
            candidate_base_url="https://custom.api.com/v1",
            candidate_api_key="sk-candidate-key",
            candidate_api_format="responses",
        )

        gen._get_candidate_client()

        mock_get_client.assert_called_once_with(
            provider="openai",
            base_url="https://custom.api.com/v1",
            api_key="sk-candidate-key",
            model="openai/gpt-4o-mini",
            api_format="responses",
        )

    def test_falls_back_to_main_client_when_no_candidate_params(self):
        """When no candidate_* params are set, falls back to _get_or_create_client."""
        gen = ContrastiveGenerator(
            method_name="rlcf_candidate",
            model="openai/gpt-4o-mini",
            candidate_models=["openai/gpt-3.5-turbo"],
        )

        # Mock the base class method
        mock_main_client = MagicMock(name="main_client")
        gen._get_or_create_client = MagicMock(return_value=mock_main_client)

        client = gen._get_candidate_client()

        assert client is mock_main_client
        gen._get_or_create_client.assert_called_once()

    @patch("autochecklist.generators.instance_level.contrastive.get_client")
    def test_only_candidate_provider_triggers_separate_client(self, mock_get_client):
        """Even just candidate_provider alone (without base_url) triggers a separate client."""
        mock_get_client.return_value = MagicMock()

        gen = ContrastiveGenerator(
            method_name="rlcf_candidate",
            model="openai/gpt-4o-mini",
            candidate_models=["openai/gpt-3.5-turbo"],
            candidate_provider="openai",
        )

        gen._get_candidate_client()

        # Should call get_client, not _get_or_create_client
        mock_get_client.assert_called_once()


class TestGenerateCandidatesUsesClientMethod:
    """Tests that _generate_candidates uses _get_candidate_client."""

    def test_generate_candidates_calls_candidate_client(self):
        """_generate_candidates should use _get_candidate_client, not _get_or_create_client."""
        gen = ContrastiveGenerator(
            method_name="rlcf_candidate",
            model="openai/gpt-4o-mini",
            candidate_models=["model-a", "model-b"],
        )

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {
            "choices": [{"message": {"content": "candidate response"}}]
        }
        gen._get_candidate_client = MagicMock(return_value=mock_client)
        gen._get_or_create_client = MagicMock(
            side_effect=AssertionError("Should not call _get_or_create_client for candidates")
        )

        candidates = gen._generate_candidates("Write a haiku")

        gen._get_candidate_client.assert_called_once()
        assert len(candidates) == 2
        assert all(c == "candidate response" for c in candidates)


class TestPipelineThreadsCandidateParams:
    """Tests that pipeline() factory threads candidate_* params via generator_kwargs."""

    @patch.object(_pipeline_module, "get_generator")
    def test_pipeline_passes_candidate_params_via_generator_kwargs(
        self, mock_get_generator
    ):
        """pipeline() should pass generator_kwargs to the generator constructor."""
        from autochecklist.pipeline import pipeline
        from autochecklist.generators.instance_level.contrastive import (
            ContrastiveGenerator,
        )

        # Create a mock factory class that is a ContrastiveGenerator subclass
        class MockContrastiveFactory(ContrastiveGenerator):
            def __init__(self, **kwargs):
                self._init_kwargs = kwargs
                # Don't call super().__init__ — just capture kwargs
                self._method_name = "rlcf_candidate"

            @property
            def generation_level(self):
                return "instance"

            @property
            def method_name(self):
                return "rlcf_candidate"

        mock_get_generator.return_value = MockContrastiveFactory

        pipe = pipeline(
            "rlcf_candidate",
            generator_model="openai/gpt-4o-mini",
            generator_kwargs={
                "candidate_provider": "vllm",
                "candidate_base_url": "http://localhost:8000/v1",
                "candidate_api_key": "sk-test",
                "candidate_api_format": "chat",
            },
        )

        init_kwargs = pipe.generator._init_kwargs
        assert init_kwargs["candidate_provider"] == "vllm"
        assert init_kwargs["candidate_base_url"] == "http://localhost:8000/v1"
        assert init_kwargs["candidate_api_key"] == "sk-test"
        assert init_kwargs["candidate_api_format"] == "chat"

    @patch.object(_pipeline_module, "get_generator")
    def test_pipeline_generator_kwargs_not_passed_without_them(
        self, mock_get_generator
    ):
        """pipeline() without generator_kwargs doesn't inject candidate params."""
        from autochecklist.pipeline import pipeline
        from autochecklist.generators.instance_level.direct import (
            DirectGenerator,
        )

        class MockPromptFactory(DirectGenerator):
            def __init__(self, **kwargs):
                self._init_kwargs = kwargs
                self._method_name = "tick"

            @property
            def generation_level(self):
                return "instance"

            @property
            def method_name(self):
                return "tick"

        mock_get_generator.return_value = MockPromptFactory

        pipe = pipeline(
            "tick",
            generator_model="openai/gpt-4o-mini",
        )

        init_kwargs = pipe.generator._init_kwargs
        assert "candidate_provider" not in init_kwargs
        assert "candidate_base_url" not in init_kwargs
