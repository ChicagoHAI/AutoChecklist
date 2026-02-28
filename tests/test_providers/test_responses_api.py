"""Tests for Responses API support.

The Responses API is OpenAI's newer API format. It returns output items
instead of choices. Our client normalizes both formats to a standard dict.
"""
import os
import pytest


class TestResponseNormalizationUnit:
    """Unit tests for response format normalization (no API needed)."""

    def test_chat_completions_passthrough(self):
        """Chat Completions format should pass through with minimal changes."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai", api_key="test", api_format="chat")
        raw = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = client._normalize_response(raw)
        assert result["choices"][0]["message"]["content"] == "Hello"
        client.close()

    def test_responses_api_text_extraction(self):
        """Should extract text from Responses API output items."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai", api_key="test", api_format="responses")
        raw = {
            "id": "resp_abc",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "The answer is 42"},
                    ],
                }
            ],
            "model": "gpt-4o-mini",
        }
        result = client._normalize_response(raw)
        assert result["choices"][0]["message"]["content"] == "The answer is 42"
        client.close()

    def test_responses_api_logprobs_extraction(self):
        """Should extract logprobs from Responses API output_text items."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai", api_key="test", api_format="responses")
        raw = {
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Yes",
                    "logprobs": [{
                        "token": "Yes",
                        "logprob": -0.05,
                        "top_logprobs": [
                            {"token": "Yes", "logprob": -0.05},
                            {"token": "No", "logprob": -3.1},
                        ],
                    }],
                }],
            }],
            "model": "gpt-4o-mini",
        }
        result = client._normalize_response(raw)
        logprobs = result["choices"][0]["logprobs"]["content"]
        assert logprobs[0]["top_logprobs"][0]["token"] == "Yes"
        assert logprobs[0]["top_logprobs"][1]["token"] == "No"
        client.close()

    def test_responses_api_multiple_output_texts(self):
        """Should concatenate multiple output_text items."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai", api_key="test", api_format="responses")
        raw = {
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Part 1. "},
                    {"type": "output_text", "text": "Part 2."},
                ],
            }],
            "model": "gpt-4o-mini",
        }
        result = client._normalize_response(raw)
        content = result["choices"][0]["message"]["content"]
        assert "Part 1" in content
        assert "Part 2" in content
        client.close()

    def test_responses_api_ignores_non_message_output(self):
        """Should skip output items that aren't messages (e.g., reasoning)."""
        from autochecklist.providers.http_client import LLMHTTPClient
        client = LLMHTTPClient(provider="openai", api_key="test", api_format="responses")
        raw = {
            "output": [
                {"type": "reasoning", "content": "thinking..."},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Final answer"}],
                },
            ],
            "model": "gpt-4o-mini",
        }
        result = client._normalize_response(raw)
        assert result["choices"][0]["message"]["content"] == "Final answer"
        client.close()


class TestResponsesAPIIntegration:
    """Integration tests requiring an API key."""

    pytestmark = pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )

    def test_real_responses_api_call(self):
        """Make a real Responses API call to OpenAI."""
        from autochecklist.providers.http_client import LLMHTTPClient
        with LLMHTTPClient(provider="openai", api_format="responses") as client:
            response = client.chat_completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello in one word"}],
                temperature=0.0,
                max_tokens=16,
            )
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    def test_real_responses_api_logprobs(self):
        """Verify logprobs work through Responses API."""
        from autochecklist.providers.http_client import LLMHTTPClient
        with LLMHTTPClient(provider="openai", api_format="responses") as client:
            result = client.get_logprobs(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer Yes or No only."},
                    {"role": "user", "content": "Is 2+2=4?"},
                ],
            )
        assert "yes" in result
        assert "no" in result
        assert result["yes"] > 0  # Should confidently say yes
