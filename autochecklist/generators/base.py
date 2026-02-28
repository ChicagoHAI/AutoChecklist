"""Base classes for checklist generators."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

from ..config import get_config
from ..models import Checklist

logger = logging.getLogger(__name__)


class ChecklistGenerator(ABC):
    """Base class for all checklist generators."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
        api_format: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        config = get_config()
        self.model = model or config.generator_model.model_id
        self.temperature = temperature if temperature is not None else config.generator_model.temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self._client = client
        self._provider = provider or config.generator_model.provider or "openrouter"
        self._base_url = base_url
        self._api_format = api_format or "chat"
        self.reasoning_effort = reasoning_effort

    @property
    @abstractmethod
    def generation_level(self) -> str:
        """Return 'instance' or 'corpus'."""
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the method name (e.g., 'tick', 'rlcf')."""
        pass

    @abstractmethod
    def generate(self, **kwargs: Any) -> Checklist:
        """Generate a checklist."""
        pass

    def generate_stream(self, **kwargs: Any) -> Iterator[str]:
        """Stream checklist generation (for UI).

        Default implementation just yields the final result.
        Override for true streaming support.
        """
        checklist = self.generate(**kwargs)
        yield checklist.to_text()

    def _get_or_create_client(self) -> Any:
        """Get injected client or create one from provider settings."""
        if self._client is not None:
            return self._client
        from ..providers.factory import get_client
        return get_client(
            provider=self._provider,
            api_key=self.api_key,
            base_url=self._base_url,
            model=self.model,
            api_format=self._api_format,
        )

    def _call_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Call the LLM and return the response text.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system prompt.
            response_format: Optional OpenAI-compatible response_format dict
                for structured JSON output. When provided, the call is attempted
                with ``response_format`` first; if the provider does not support
                it, the call is retried without it (fallback to schema-in-prompt).

        Returns:
            The model's response text.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        client = self._get_or_create_client()

        try:
            response = client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
        except (ValueError, KeyError, TypeError) as e:
            # Schema/parsing errors from response_format — fall back to schema-in-prompt
            if response_format is not None:
                logger.warning(
                    "Structured output failed (%s), retrying without "
                    "response_format (fallback to schema-in-prompt).",
                    e,
                )
                fallback_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
                response = client.chat_completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **fallback_kwargs,
                )
            else:
                raise
        except Exception as e:
            # For HTTP errors (auth, rate limit, server), only fallback on 400 (bad schema)
            import httpx

            if (
                response_format is not None
                and isinstance(e, httpx.HTTPStatusError)
                and e.response.status_code == 400
            ):
                logger.warning(
                    "Structured output failed (%s), retrying without "
                    "response_format (fallback to schema-in-prompt).",
                    e,
                )
                fallback_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
                response = client.chat_completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **fallback_kwargs,
                )
            else:
                raise

        return response["choices"][0]["message"]["content"]


class InstanceChecklistGenerator(ChecklistGenerator):
    """Base for instance-level generators (one checklist per input)."""

    @property
    def generation_level(self) -> str:
        return "instance"

    @abstractmethod
    def generate(
        self,
        input: str,
        target: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> Checklist:
        """Generate checklist for an input.

        Args:
            input: The input/query/instruction text
            target: Optional target response to evaluate
            reference: Optional reference target for comparison
            **kwargs: Method-specific arguments

        Returns:
            Generated Checklist
        """
        pass


class CorpusChecklistGenerator(ChecklistGenerator):
    """Base for corpus-level generators (one checklist for entire dataset)."""

    @property
    def generation_level(self) -> str:
        return "corpus"

    @abstractmethod
    def generate(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Checklist:
        """Generate checklist from corpus inputs.

        Args:
            inputs: List of input items (feedback, dimensions, think-aloud, etc.)
            **kwargs: Method-specific arguments

        Returns:
            Generated Checklist
        """
        pass
