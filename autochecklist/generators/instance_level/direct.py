"""DirectGenerator: Unified instance-level checklist generator."""

import json
from pathlib import Path
from typing import Any, Optional, Union

from ...models import (
    Checklist,
    ChecklistItem,
    ChecklistResponse,
    to_response_format,
)
from ...prompts import load_template, load_format, PromptTemplate
from ...utils.parsing import extract_json
from ..base import InstanceChecklistGenerator


class DirectGenerator(InstanceChecklistGenerator):
    """Generate checklists using a prompt template + structured JSON output.

    Can be configured via pipeline presets (built-in methods) or custom prompts.

    Args:
        method_name: Pipeline preset name (e.g., "tick") or custom name.
            If a known preset, loads config from PIPELINE_PRESETS.
        custom_prompt: Custom prompt template. Pass a Path to load from file,
            or a str for raw prompt text. Overrides preset template.
        response_schema: Pydantic model for JSON validation. Default: ChecklistResponse.
        format_name: Format prompt file name (e.g., "checklist"). Default from preset.
        max_items: Maximum checklist items to return.
        min_items: Minimum expected items.
        **kwargs: Passed to InstanceChecklistGenerator (model, temperature, etc.)
    """

    def __init__(
        self,
        method_name: str = "custom",
        custom_prompt: Optional[Union[str, Path]] = None,
        response_schema: Optional[type] = None,
        format_name: Optional[str] = None,
        max_items: int = 10,
        min_items: int = 2,
        **kwargs: Any,
    ):
        # Load preset defaults if this is a known method
        from .pipeline_presets import PIPELINE_PRESETS

        preset = PIPELINE_PRESETS.get(method_name, {})

        # Apply preset defaults, allowing kwargs to override
        if "temperature" not in kwargs and "temperature" in preset:
            kwargs["temperature"] = preset["temperature"]

        super().__init__(**kwargs)

        self._method_name = method_name
        self.max_items = preset.get("max_items", max_items)
        self.min_items = preset.get("min_items", min_items)
        self._response_schema = response_schema or preset.get(
            "response_schema", ChecklistResponse
        )
        self._format_name = format_name or preset.get("format_name", "checklist")

        # Load template
        if custom_prompt is not None:
            if isinstance(custom_prompt, Path):
                template_text = custom_prompt.read_text(encoding="utf-8")
            else:
                template_text = custom_prompt
        elif preset:
            template_text = load_template(
                preset["template_dir"], preset["template_name"]
            )
        else:
            raise ValueError(
                f"Unknown method '{method_name}' and no custom_prompt provided"
            )

        self._template = PromptTemplate(template_text)

    @property
    def method_name(self) -> str:
        return self._method_name

    @property
    def prompt_text(self) -> str:
        """The raw prompt template text."""
        return self._template.template

    def generate(
        self,
        input: str,
        target: Optional[str] = None,
        reference: Optional[str] = None,
        history: str = "",
        **kwargs: Any,
    ) -> Checklist:
        """Generate checklist from input using template + structured output.

        Automatically detects which placeholders the template needs and passes
        only those. This allows the same class to handle TICK (input only),
        RocketEval (input + reference + history), RLCF-direct
        (input + reference), etc.
        """
        # Build format kwargs — only pass placeholders that exist in template
        format_kwargs: dict[str, str] = {"input": input}
        if "target" in self._template._placeholders and target is not None:
            format_kwargs["target"] = target
        if "reference" in self._template._placeholders:
            if reference is None:
                raise ValueError(
                    f"{self._method_name} requires a reference target."
                )
            format_kwargs["reference"] = reference
        if "history" in self._template._placeholders:
            format_kwargs["history"] = history

        # Load format instructions
        format_text = load_format(self._format_name)

        # Inject format inline if template has {format_instructions} placeholder,
        # otherwise append after the prompt (default).
        if "format_instructions" in self._template._placeholders:
            format_kwargs["format_instructions"] = format_text
            full_prompt = self._template.format(**format_kwargs)
        else:
            prompt = self._template.format(**format_kwargs)
            full_prompt = prompt + "\n\n" + format_text

        # Call model with structured output
        response_format = to_response_format(
            self._response_schema, self._method_name
        )
        raw = self._call_model(full_prompt, response_format=response_format)

        # Parse structured response
        items = self._parse_structured(raw)

        return Checklist(
            items=items,
            source_method=self.method_name,
            generation_level=self.generation_level,
            input=input,
            metadata={"raw_response": raw},
        )

    def _parse_structured(self, raw: str) -> list[ChecklistItem]:
        """Parse JSON response using Pydantic schema.

        Primary path: json.loads() succeeds (structured output).
        Fallback path: extract_json() extracts JSON from raw text.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = extract_json(raw)
        validated = self._response_schema.model_validate(data)

        items = []
        for q in validated.questions[: self.max_items]:
            items.append(
                ChecklistItem(
                    question=q.question,
                    weight=getattr(q, "weight", 100.0),
                )
            )
        return items
