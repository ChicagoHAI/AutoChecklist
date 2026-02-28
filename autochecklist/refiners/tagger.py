"""Tagger refiner for filtering checklist items based on quality criteria.

Filters questions based on two criteria:
1. Generally applicable - question can be answered (Yes/No) for any input
2. Section specific - question evaluates a single focused aspect

Uses zero-shot chain-of-thought prompting for classification.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..models import Checklist, ChecklistItem
from ..prompts import load_template, PromptTemplate
from .base import ChecklistRefiner


class Tagger(ChecklistRefiner):
    """Refiner that filters checklist items based on applicability and specificity.

    Uses LLM (default: gpt-5-mini) with zero-shot CoT to classify each question:
    - Generally applicable: Can be answered Yes/No for any input (no N/A scenarios)
    - Section specific: Evaluates single aspect without cross-references
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        custom_prompt: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        # Use gpt-5-mini as default (instead of o3-mini from paper)
        super().__init__(
            model=model or "openai/gpt-5-mini",
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )
        # Load tagging prompt template
        if custom_prompt is not None:
            if isinstance(custom_prompt, Path):
                template_str = custom_prompt.read_text(encoding="utf-8")
            else:
                template_str = custom_prompt
        else:
            template_str = load_template("generators/feedback", "tag")
        self._tag_template = PromptTemplate(template_str)

    @property
    def refiner_name(self) -> str:
        return "tagger"

    def refine(self, checklist: Checklist, **kwargs: Any) -> Checklist:
        """Filter checklist items based on tagging criteria.

        Args:
            checklist: Input checklist to filter

        Returns:
            Checklist with only items that pass both criteria
        """
        if len(checklist.items) == 0:
            return self._create_refined_checklist(
                checklist,
                [],
                metadata_updates={
                    "filtered_count": 0,
                    "filtered_items": [],
                },
            )

        passing_items: List[ChecklistItem] = []
        filtered_items: List[Dict[str, Any]] = []

        for item in checklist.items:
            tag_result = self._tag_question(item)

            if tag_result["generally_applicable"] and tag_result["section_specific"]:
                # Question passes - add tagging metadata
                item_with_metadata = ChecklistItem(
                    id=item.id,
                    question=item.question,
                    weight=item.weight,
                    category=item.category,
                    metadata={
                        **(item.metadata or {}),
                        "generally_applicable": True,
                        "section_specific": True,
                        "tag_reasoning": tag_result.get("reasoning", ""),
                    },
                )
                passing_items.append(item_with_metadata)
            else:
                # Question filtered out - track reason
                filtered_items.append({
                    "id": item.id,
                    "question": item.question,
                    "generally_applicable": tag_result["generally_applicable"],
                    "section_specific": tag_result["section_specific"],
                    "reasoning": tag_result.get("reasoning", ""),
                })

        return self._create_refined_checklist(
            checklist,
            passing_items,
            metadata_updates={
                "filtered_count": len(filtered_items),
                "filtered_items": filtered_items,
            },
        )

    def _tag_question(self, item: ChecklistItem) -> Dict[str, Any]:
        """Tag a single question for applicability and specificity.

        Args:
            item: Checklist item to tag

        Returns:
            Dict with keys: generally_applicable, section_specific, reasoning
        """
        prompt = self._tag_template.format(question=item.question)

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "tag_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "generally_applicable": {"type": "boolean"},
                        "section_specific": {"type": "boolean"},
                    },
                    "required": ["reasoning", "generally_applicable", "section_specific"],
                    "additionalProperties": False,
                },
            },
        }

        try:
            response = self._call_model(prompt, response_format=response_format)
            result = json.loads(response)
            return {
                "generally_applicable": result.get("generally_applicable", False),
                "section_specific": result.get("section_specific", False),
                "reasoning": result.get("reasoning", ""),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            # Malformed response - default to filtering out
            return {
                "generally_applicable": False,
                "section_specific": False,
                "reasoning": "Failed to parse LLM response",
            }
