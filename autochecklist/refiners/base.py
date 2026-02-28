"""Base class for checklist refiners."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..config import get_config
from ..models import Checklist, ChecklistItem


class ChecklistRefiner(ABC):
    """Base class for all checklist refiners.

    Refiners take a checklist and improve it through various operations:
    - Deduplication (merge similar questions)
    - Filtering (remove low-quality questions)
    - Selection (choose optimal subset)
    - Testing (validate discriminativeness)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
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
        self.api_key = api_key
        self._client = client
        self._provider = provider or config.generator_model.provider or "openrouter"
        self._base_url = base_url
        self._api_format = api_format or "chat"
        self.reasoning_effort = reasoning_effort

    @property
    @abstractmethod
    def refiner_name(self) -> str:
        """Return the refiner name (e.g., 'deduplicator', 'tagger')."""
        pass

    @abstractmethod
    def refine(self, checklist: Checklist, **kwargs: Any) -> Checklist:
        """Refine the checklist.

        Args:
            checklist: Input checklist to refine
            **kwargs: Refiner-specific arguments

        Returns:
            Refined checklist
        """
        pass

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
        response_format: Optional[Dict] = None,
    ) -> str:
        """Call the LLM and return the response text."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_or_create_client()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2048,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = client.chat_completion(**kwargs)
        return response["choices"][0]["message"]["content"]

    def _create_refined_checklist(
        self,
        original: Checklist,
        items: List[ChecklistItem],
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Checklist:
        """Create a new checklist with refined items.

        Args:
            original: Original checklist to base metadata on
            items: New refined items
            metadata_updates: Additional metadata to add

        Returns:
            New Checklist instance
        """
        metadata = dict(original.metadata) if original.metadata else {}
        metadata["refined_by"] = self.refiner_name
        metadata["original_count"] = len(original.items)
        if metadata_updates:
            metadata.update(metadata_updates)

        return Checklist(
            items=items,
            source_method=original.source_method,
            generation_level=original.generation_level,
            input=original.input,
            metadata=metadata,
        )
