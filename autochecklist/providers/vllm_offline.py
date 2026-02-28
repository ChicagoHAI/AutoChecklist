"""vLLM offline inference client.

Wraps vllm.LLM for direct Python inference without an HTTP server.
Satisfies the LLMClient protocol so it can be used anywhere the
HTTP client is used.
"""

import math
from typing import Any, Callable, Dict, List, Optional


class VLLMOfflineClient:
    """Offline inference client using vLLM's Python API.

    Loads a model once at __init__ and reuses it for all calls.
    The model parameter in method signatures is ignored — the model
    is fixed at construction time.

    Context manager is a no-op: the model stays loaded. This is critical
    because existing code does `with Client() as client:` in tight loops.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        **vllm_kwargs: Any,
    ):
        try:
            from vllm import LLM, SamplingParams
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "vllm is required for offline inference. "
                "Install with: pip install vllm"
            )

        self._SamplingParams = SamplingParams
        self._model_name = model
        self._llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **vllm_kwargs,
        )
        self._tokenizer = self._llm.get_tokenizer()

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a prompt string using the model's template."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback for models without chat templates
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate completion, returning OpenAI-format dict.

        The model parameter is ignored — uses the model loaded at init.
        """
        prompt = self._apply_chat_template(messages)

        logprobs_count = kwargs.pop("top_logprobs", None)
        request_logprobs = kwargs.pop("logprobs", False)
        kwargs.pop("reasoning_effort", None)  # Not supported by vLLM

        # Handle response_format → guided decoding
        response_format = kwargs.pop("response_format", None)
        guided_params = self._build_guided_params(response_format)

        sampling_params = self._SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs_count if request_logprobs else None,
            guided_decoding=guided_params,
        )

        outputs = self._llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # Build OpenAI-format response
        response: Dict[str, Any] = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": output.text,
                },
                "finish_reason": output.finish_reason,
            }],
            "model": self._model_name,
            "usage": {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(output.token_ids),
            },
        }

        # Include logprobs if requested
        if request_logprobs and output.logprobs:
            response["choices"][0]["logprobs"] = self._format_logprobs(
                output.logprobs, logprobs_count or 5
            )

        return response

    @staticmethod
    def _build_guided_params(response_format: Optional[Dict[str, Any]]) -> Any:
        """Convert OpenAI-style response_format to vLLM GuidedDecodingParams.

        Returns None when no response_format is provided.
        """
        if response_format is None:
            return None
        from vllm import GuidedDecodingParams

        json_schema = response_format
        if isinstance(response_format, dict) and "json_schema" in response_format:
            json_schema = response_format["json_schema"].get("schema", response_format)
        return GuidedDecodingParams(json=json_schema)

    def _format_logprobs(
        self,
        vllm_logprobs: List,
        top_n: int,
    ) -> Dict[str, Any]:
        """Convert vLLM logprobs to OpenAI format.

        vLLM returns: List[Dict[int, Logprob]] per token
        OpenAI format: {"content": [{"token": str, "logprob": float, "top_logprobs": [...]}]}
        """
        content = []
        for token_logprobs in vllm_logprobs:
            if token_logprobs is None:
                continue

            sorted_lps = sorted(
                token_logprobs.values(),
                key=lambda lp: lp.logprob,
                reverse=True,
            )[:top_n]

            top_logprobs_list = []
            for lp in sorted_lps:
                token_str = (
                    lp.decoded_token
                    if hasattr(lp, "decoded_token") and lp.decoded_token
                    else str(lp.rank if hasattr(lp, "rank") else "")
                )
                top_logprobs_list.append({
                    "token": token_str,
                    "logprob": lp.logprob,
                })

            if top_logprobs_list:
                content.append({
                    "token": top_logprobs_list[0]["token"],
                    "logprob": top_logprobs_list[0]["logprob"],
                    "top_logprobs": top_logprobs_list,
                })

        return {"content": content}

    def get_logprobs(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Get Yes/No log probabilities."""
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

    def supports_logprobs(self, model: str) -> bool:
        """vLLM always supports logprobs."""
        return True

    def batch_completions(
        self,
        requests: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Process batch using vLLM's native batching."""
        prompts = []
        sampling_params_list = []

        for req in requests:
            msgs = req["messages"]
            prompt = self._apply_chat_template(msgs)
            prompts.append(prompt)
            guided_params = self._build_guided_params(
                req.get("response_format")
            )
            sampling_params_list.append(self._SamplingParams(
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens", 2048),
                guided_decoding=guided_params,
            ))

        all_outputs = self._llm.generate(prompts, sampling_params_list)

        results = []
        for i, output in enumerate(all_outputs):
            results.append({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": output.outputs[0].text,
                    },
                }],
                "model": self._model_name,
            })
            if progress_callback:
                progress_callback(i + 1)

        return results

    def close(self) -> None:
        """No-op — model stays loaded until garbage collection."""
        pass

    def __enter__(self) -> "VLLMOfflineClient":
        return self

    def __exit__(self, *args: Any) -> None:
        # Do NOT unload model on context exit
        pass
