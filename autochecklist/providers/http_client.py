"""Provider-agnostic HTTP client for OpenAI-compatible APIs.

Supports OpenRouter, OpenAI, and vLLM server mode.
"""

import asyncio
import json
import math
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .base import get_provider_config


def _raise_with_detail(response: httpx.Response) -> None:
    """Like response.raise_for_status() but includes the provider's error message."""
    if response.is_success:
        return
    # Try to extract the provider's error message from the JSON body
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error", {})
            if isinstance(err, dict):
                detail = err.get("message", "")
            elif isinstance(err, str):
                detail = err
    except Exception:
        pass
    if detail:
        raise httpx.HTTPStatusError(
            f"{response.status_code}: {detail}",
            request=response.request,
            response=response,
        )
    response.raise_for_status()


def _is_retryable(exc: BaseException) -> bool:
    """Only retry on server errors (5xx) and timeouts, not client errors (4xx)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    # Retry on connection/timeout errors
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout))

# Model prefixes for reasoning models that have API parameter restrictions:
# - Require max_completion_tokens instead of max_tokens
# - Only support temperature=1 (reject other values)
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model with restricted API parameters."""
    # Strip common provider prefixes (e.g. "openai/gpt-5-mini")
    name = model.rsplit("/", 1)[-1] if "/" in model else model
    return any(name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _normalize_model_name(model: str, provider: str) -> str:
    """Normalize model name for the target provider.

    OpenRouter uses 'org/model' format (e.g. 'openai/gpt-5-mini').
    OpenAI expects bare model names (e.g. 'gpt-5-mini').
    vLLM uses HuggingFace IDs which legitimately contain slashes.
    """
    if provider == "openai" and "/" in model:
        # Strip org prefix (e.g. "openai/gpt-5-mini" → "gpt-5-mini")
        return model.split("/", 1)[1]
    return model


class LLMHTTPClient:
    """HTTP client for OpenAI-compatible LLM APIs.

    Works with OpenRouter, OpenAI, and vLLM server mode. Supports both
    Chat Completions and Responses API formats.
    """

    # Class-level cache for model capabilities, keyed by (provider, base_url)
    _models_cache: Dict[tuple, List[Dict[str, Any]]] = {}
    _models_cache_time: Dict[tuple, float] = {}
    _CACHE_TTL = 3600  # 1 hour

    def __init__(
        self,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        api_format: str = "chat",
    ):
        config = get_provider_config(provider, base_url=base_url)
        self.provider = provider
        self.api_format = api_format

        # Resolve base URL
        self.base_url = base_url or config.base_url or ""

        # Resolve API key: explicit > env var > config
        self.api_key = self._resolve_api_key(api_key, config)

        self.timeout = timeout
        self._provider_config = config

        # Build headers
        headers = {
            "Content-Type": "application/json",
            **config.default_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )

    def _resolve_api_key(
        self,
        explicit_key: Optional[str],
        config: Any,
    ) -> Optional[str]:
        """Resolve API key from explicit param, env var, or global config."""
        if explicit_key:
            return explicit_key

        # Try provider-specific env var
        if config.api_key_env_var:
            env_key = os.getenv(config.api_key_env_var)
            if env_key:
                return env_key

        # Try global config for backward compat
        if self.provider == "openrouter":
            from ..config import get_config
            cfg = get_config()
            if cfg.openrouter_api_key:
                return cfg.openrouter_api_key

        if config.requires_api_key:
            env_var = config.api_key_env_var or "API_KEY"
            raise ValueError(
                f"{self.provider.title()} API key required. "
                f"Set {env_var} env var or pass api_key parameter."
            )

        return None

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "LLMHTTPClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable),
    )
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a chat completion request.

        For api_format="responses", translates to/from the Responses API format.
        Always returns normalized OpenAI Chat Completions format.
        """
        model = _normalize_model_name(model, self.provider)

        if self.api_format == "responses":
            return self._chat_completion_responses(
                model, messages, temperature, max_tokens, **kwargs
            )

        # Reasoning models require max_completion_tokens and only support temperature=1
        is_reasoning = _is_reasoning_model(model)
        token_key = "max_completion_tokens" if is_reasoning else "max_tokens"
        body = {
            "model": model,
            "messages": messages,
            token_key: max_tokens,
            **kwargs,
        }
        if not is_reasoning:
            body["temperature"] = temperature
        response = self._client.post("/chat/completions", json=body)
        _raise_with_detail(response)
        raw = response.json()
        return self._normalize_response(raw)

    def _chat_completion_responses(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a Responses API request and normalize to Chat Completions format."""
        # Build Responses API request body
        body: Dict[str, Any] = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
        }
        if not _is_reasoning_model(model):
            body["temperature"] = temperature

        # Handle response_format for Responses API
        # Chat Completions uses: {"response_format": {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}}
        # Responses API uses:    {"text": {"format": {"type": "json_schema", "name": ..., "schema": ...}}}
        response_format = kwargs.pop("response_format", None)
        if response_format is not None:
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and "json_schema" in response_format
            ):
                # Flatten: unwrap json_schema wrapper for Responses API
                inner = response_format["json_schema"]
                body["text"] = {"format": {
                    "type": "json_schema",
                    **inner,
                }}
            else:
                body["text"] = {"format": response_format}

        # Map Chat Completions kwargs to Responses API equivalents
        if kwargs.get("logprobs"):
            if "text" not in body:
                body["text"] = {"format": {"type": "text"}}
            body["include"] = ["message.output_text.logprobs"]
            # Responses API uses top_logprobs at request level
            if "top_logprobs" in kwargs:
                body["top_logprobs"] = kwargs.pop("top_logprobs")
            kwargs.pop("logprobs", None)

        # Pass through remaining kwargs
        for k, v in kwargs.items():
            if k not in ("max_tokens",):
                body[k] = v

        response = self._client.post("/responses", json=body)
        _raise_with_detail(response)
        raw = response.json()
        return self._normalize_response(raw)

    def _normalize_response(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize response to Chat Completions format.

        Handles both Chat Completions (passthrough) and Responses API
        (translation) formats.
        """
        # Chat Completions format — already has "choices"
        if "choices" in raw:
            return raw

        # Responses API format — has "output"
        if "output" in raw:
            return self._normalize_responses_api(raw)

        # Unknown format — return as-is
        return raw

    def _normalize_responses_api(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Translate Responses API format to Chat Completions format."""
        # Find message outputs
        text_parts = []
        all_logprobs = []

        for item in raw.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text_parts.append(content.get("text", ""))
                    if "logprobs" in content and content["logprobs"]:
                        all_logprobs.extend(content["logprobs"])

        combined_text = "".join(text_parts)

        choice: Dict[str, Any] = {
            "message": {
                "role": "assistant",
                "content": combined_text,
            },
            "finish_reason": "stop",
        }

        # Include logprobs if present
        if all_logprobs:
            choice["logprobs"] = {"content": all_logprobs}

        result: Dict[str, Any] = {
            "choices": [choice],
            "model": raw.get("model", ""),
        }

        if "usage" in raw:
            result["usage"] = raw["usage"]

        return result

    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream chat completion. Yields content chunks."""
        model = _normalize_model_name(model, self.provider)
        with self._client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                **kwargs,
            },
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                            if content := chunk["choices"][0]["delta"].get("content"):
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

    def get_logprobs(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Get Yes/No log probabilities for normalized scoring.

        Returns dict with "yes" and "no" probability values.
        """
        response = self.chat_completion(
            model=model,
            messages=messages,
            logprobs=True,
            top_logprobs=5,
            max_tokens=512,
            **kwargs,
        )

        try:
            logprobs_data = response["choices"][0].get("logprobs")
            if logprobs_data is None:
                return {"yes": 0.0, "no": 0.0}

            logprobs = logprobs_data["content"][0]["top_logprobs"]
            probs = {
                lp["token"].lower().strip(): math.exp(lp["logprob"])
                for lp in logprobs
            }
            return {
                "yes": probs.get("yes", 0.0),
                "no": probs.get("no", 0.0),
            }
        except (KeyError, IndexError, TypeError):
            return {"yes": 0.0, "no": 0.0}

    def _get_models(self) -> List[Dict[str, Any]]:
        """Get models list (OpenRouter only), using cache if fresh."""
        cache_key = (self.provider, self.base_url)
        now = time.time()

        cached = LLMHTTPClient._models_cache.get(cache_key)
        cached_time = LLMHTTPClient._models_cache_time.get(cache_key, 0)

        if cached is not None and now - cached_time < self._CACHE_TTL:
            return cached

        try:
            response = self._client.get("/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            LLMHTTPClient._models_cache[cache_key] = models
            LLMHTTPClient._models_cache_time[cache_key] = now
            return models
        except Exception:
            return []

    def supports_logprobs(self, model: str) -> bool:
        """Check if a model supports logprobs.

        vLLM and OpenAI always support logprobs.
        OpenRouter queries the /models endpoint.
        """
        if self.provider in ("vllm", "openai"):
            return True

        # OpenRouter: query models endpoint
        models = self._get_models()
        for m in models:
            if m.get("id") == model:
                supported = m.get("supported_parameters", [])
                return "logprobs" in supported
        return False

    async def chat_completion_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async chat completion request."""
        headers = {
            "Content-Type": "application/json",
            **self._provider_config.default_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        model = _normalize_model_name(model, self.provider)

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        ) as client:
            is_reasoning = _is_reasoning_model(model)
            token_key = "max_completion_tokens" if is_reasoning else "max_tokens"
            body = {
                "model": model,
                "messages": messages,
                token_key: max_tokens,
                **kwargs,
            }
            if not is_reasoning:
                body["temperature"] = temperature
            response = await client.post("/chat/completions", json=body)
            _raise_with_detail(response)
            return response.json()

    async def _batch_completions_async(
        self,
        requests: List[Dict[str, Any]],
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple requests concurrently (async implementation)."""
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Optional[Dict[str, Any]]] = [None] * len(requests)
        completed = 0

        async def limited_request(idx: int, req: Dict[str, Any]):
            nonlocal completed
            async with semaphore:
                try:
                    result = await self.chat_completion_async(**req)
                    results[idx] = result
                except Exception as e:
                    results[idx] = {"error": str(e)}
                finally:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed)

        await asyncio.gather(*[
            limited_request(i, req) for i, req in enumerate(requests)
        ])

        return results  # type: ignore

    def batch_completions(
        self,
        requests: List[Dict[str, Any]],
        concurrency: int = 5,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple requests concurrently (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._batch_completions_async(requests, concurrency, progress_callback)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._batch_completions_async(requests, concurrency, progress_callback)
                )
        except RuntimeError:
            return asyncio.run(
                self._batch_completions_async(requests, concurrency, progress_callback)
            )
