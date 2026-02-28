"""Tests for vLLM inference client (server or offline).

Uses the vllm_offline_client fixture which auto-detects a running vLLM server
(fast) or falls back to offline mode with a small model (slow).
Start a server with: bash scripts/vllm_server.sh
"""
import pytest

from autochecklist.providers.vllm_offline import VLLMOfflineClient

# Mark all tests in this module
pytestmark = pytest.mark.vllm_offline


def _is_offline(client) -> bool:
    """Check if the client is an offline VLLMOfflineClient."""
    return isinstance(client, VLLMOfflineClient)


class TestVLLMClient:
    """Test vLLM client core functionality (works with server or offline)."""

    def test_chat_completion_returns_openai_format(self, vllm_offline_client):
        """Response dict should have choices[0].message.content structure."""
        messages = [
            {"role": "user", "content": "Say hello in exactly one word."}
        ]
        response = vllm_offline_client.chat_completion(
            model=vllm_offline_client._test_model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
        )
        assert "choices" in response
        assert len(response["choices"]) > 0
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    def test_chat_completion_with_system_message(self, vllm_offline_client):
        """Should handle system + user messages via chat template."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always respond in one word."},
            {"role": "user", "content": "What color is the sky?"}
        ]
        response = vllm_offline_client.chat_completion(
            model=vllm_offline_client._test_model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
        )
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    def test_supports_logprobs_always_true(self, vllm_offline_client):
        """vLLM should always report logprobs support."""
        assert vllm_offline_client.supports_logprobs("any-model") is True

    def test_get_logprobs_returns_yes_no_probs(self, vllm_offline_client):
        """get_logprobs() should return yes/no probability dict."""
        messages = [
            {"role": "system", "content": "Answer only Yes or No."},
            {"role": "user", "content": "Is the Earth round? Answer Yes or No."}
        ]
        result = vllm_offline_client.get_logprobs(
            model=vllm_offline_client._test_model,
            messages=messages,
        )
        assert "yes" in result
        assert "no" in result
        assert isinstance(result["yes"], float)
        assert isinstance(result["no"], float)
        assert result["yes"] > 0 or result["no"] > 0

    def test_batch_completions_returns_correct_count(self, vllm_offline_client):
        """Batch should return same number of responses as requests."""
        model = vllm_offline_client._test_model
        requests = [
            {"model": model, "messages": [{"role": "user", "content": "Say A"}], "max_tokens": 16},
            {"model": model, "messages": [{"role": "user", "content": "Say B"}], "max_tokens": 16},
            {"model": model, "messages": [{"role": "user", "content": "Say C"}], "max_tokens": 16},
        ]
        responses = vllm_offline_client.batch_completions(requests)
        assert len(responses) == 3
        for resp in responses:
            assert "choices" in resp
            content = resp["choices"][0]["message"]["content"]
            assert isinstance(content, str)
            assert len(content) > 0

    def test_context_manager_does_not_unload_model(self, vllm_offline_client):
        """Context manager should be a no-op — model stays loaded (offline only)."""
        if not _is_offline(vllm_offline_client):
            pytest.skip("context manager no-op only applies to offline client")
        model = vllm_offline_client._test_model
        with vllm_offline_client as client:
            response1 = client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": "Say test"}],
                max_tokens=16,
            )
        response2 = vllm_offline_client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": "Say test again"}],
            max_tokens=16,
        )
        assert response1["choices"][0]["message"]["content"]
        assert response2["choices"][0]["message"]["content"]


class TestVLLMChatTemplate:
    """Test chat template application (offline only — server handles templates internally)."""

    def test_apply_chat_template_produces_string(self, vllm_offline_client):
        if not _is_offline(vllm_offline_client):
            pytest.skip("chat template tests only apply to offline client")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        prompt = vllm_offline_client._apply_chat_template(messages)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Hello" in prompt

    def test_apply_chat_template_includes_generation_prompt(self, vllm_offline_client):
        if not _is_offline(vllm_offline_client):
            pytest.skip("chat template tests only apply to offline client")
        messages = [{"role": "user", "content": "Hi"}]
        prompt = vllm_offline_client._apply_chat_template(messages)
        assert isinstance(prompt, str)
        assert len(prompt) > len("Hi")


class TestVLLMOfflineInit:
    """Test VLLMOfflineClient initialization."""

    def test_init_without_vllm_raises_import_error(self, monkeypatch):
        """Should raise clear ImportError if vLLM is not installed."""
        import sys
        monkeypatch.setitem(sys.modules, "vllm", None)
        with pytest.raises(ImportError, match="vllm"):
            VLLMOfflineClient(model="any-model")

    def test_progress_callback_in_batch(self, vllm_offline_client):
        if not _is_offline(vllm_offline_client):
            pytest.skip("progress callback only supported in offline client")
        progress_calls = []
        model = vllm_offline_client._test_model
        requests = [
            {"model": model, "messages": [{"role": "user", "content": "A"}], "max_tokens": 8},
            {"model": model, "messages": [{"role": "user", "content": "B"}], "max_tokens": 8},
        ]
        vllm_offline_client.batch_completions(
            requests, progress_callback=lambda n: progress_calls.append(n)
        )
        assert len(progress_calls) == 2
        assert progress_calls == [1, 2]


class TestVLLMEndToEnd:
    """End-to-end test with DirectGenerator."""

    def test_tick_generator_with_vllm_client(self, vllm_offline_client):
        """DirectGenerator should produce a valid Checklist using vLLM."""
        from autochecklist.generators.instance_level.direct import DirectGenerator
        from autochecklist.models import Checklist

        gen = DirectGenerator(
            method_name="tick",
            client=vllm_offline_client,
            model=vllm_offline_client._test_model,
        )
        checklist = gen.generate(input="Write a short poem about nature")

        assert isinstance(checklist, Checklist)
        assert len(checklist.items) > 0
        for item in checklist.items:
            assert item.question.strip().endswith("?")
