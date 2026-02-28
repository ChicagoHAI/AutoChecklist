"""DeductiveGenerator - dimension-based checklist generation, abstracted from CheckEval.

DeductiveGenerator generates binary yes/no evaluation questions from dimension definitions.
It supports four augmentation modes:
- seed: Minimal set of questions (1-3 per sub-dimension)
- elaboration: Expanded questions with more detail
- diversification: Alternative framings of questions
- combined: Both elaboration AND diversification from seeds

Reference: https://arxiv.org/abs/2403.18771 (CheckEval paper)
"""

from enum import Enum
from typing import Dict, List, Optional, Any
import json
import uuid

from ...models import Checklist, ChecklistItem, DeductiveInput
from ...prompts import load_template, PromptTemplate
from ...refiners import Deduplicator
from ..base import CorpusChecklistGenerator


class AugmentationMode(str, Enum):
    """Augmentation modes for question generation."""

    SEED = "seed"  # Minimal questions (1-3 per sub-dimension)
    ELABORATION = "elaboration"  # Expanded with more detail
    DIVERSIFICATION = "diversification"  # Alternative framings
    COMBINED = "combined"  # Both elaboration + diversification from seeds (paper-faithful)


# Augmentation instructions for each mode
AUGMENTATION_INSTRUCTIONS = {
    AugmentationMode.SEED: """
For SEED mode, keep only the most essential questions:
- 1-3 questions per sub-aspect
- Focus on the core criteria
- Remove redundant or overlapping questions
""",
    AugmentationMode.ELABORATION: """
For ELABORATION mode, expand each question with more specific checks:
- Add 3-5 detailed questions per sub-aspect
- Break down broad criteria into specific verifiable points
- Add questions that check for common failure modes
- Include questions about specific aspects like transitions, consistency, structure
""",
    AugmentationMode.DIVERSIFICATION: """
For DIVERSIFICATION mode, create alternative framings:
- Rephrase questions using different words
- Add both positive and negative framings (e.g., "Does it have X?" and "Does it avoid not-X?")
- Include questions from different perspectives
- Add questions that check the same criterion in different ways
""",
}


class DeductiveGenerator(CorpusChecklistGenerator):
    """Generate checklists from evaluation dimension definitions.

    CheckEval creates binary yes/no evaluation questions organized by
    dimension and sub-dimension. Questions can be provided as seeds
    or generated from dimension definitions.

    Args:
        model: OpenRouter model ID for generation
        augmentation_mode: One of seed, elaboration, or diversification
        task_type: Type of task being evaluated (e.g., "summarization", "dialog")

    Example:
        >>> checkeval = DeductiveGenerator(model="openai/gpt-4o-mini")
        >>> dimensions = [
        ...     DeductiveInput(
        ...         name="coherence",
        ...         definition="The response should maintain logical flow.",
        ...         sub_dimensions=["Logical Flow", "Consistency"]
        ...     )
        ... ]
        >>> checklist = checkeval.generate(dimensions=dimensions)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        augmentation_mode: str | AugmentationMode = AugmentationMode.SEED,
        task_type: str = "general",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, **kwargs)

        # Handle string augmentation mode
        if isinstance(augmentation_mode, str):
            augmentation_mode = AugmentationMode(augmentation_mode.lower())
        self.augmentation_mode = augmentation_mode

        self.task_type = task_type

        # Load prompt templates
        self._generate_template = PromptTemplate(load_template("generators/checkeval", "generate"))
        self._augment_template = PromptTemplate(load_template("generators/checkeval", "augment"))
        self._filter_template = PromptTemplate(load_template("generators/checkeval", "filter"))

    @property
    def method_name(self) -> str:
        """Return the method name for this generator."""
        return "checkeval"

    def generate(
        self,
        dimensions: List[DeductiveInput],
        seed_questions: Optional[Dict[str, Dict[str, List[str]]]] = None,
        augment: bool = True,
        max_questions: Optional[int] = None,
        apply_filtering: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Checklist:
        """Generate a checklist from dimension definitions.

        Args:
            dimensions: List of DeductiveInput with name, definition, sub_dimensions
            seed_questions: Optional pre-defined questions by dimension/sub-aspect
                Format: {dimension: {sub_aspect: [questions]}}
            augment: Whether to augment seed questions (default True)
            max_questions: Maximum number of questions to include
            apply_filtering: Whether to apply Tagger and Deduplicator refiners (default False)
            verbose: Print progress at each stage (default False)
            **kwargs: Additional parameters passed to generation

        Returns:
            Checklist with generated questions
        """
        if not dimensions:
            return Checklist(
                id=str(uuid.uuid4()),
                items=[],
                source_method="checkeval",
                generation_level="corpus",
                metadata={
                    "dimension_count": 0,
                    "dimensions": [],
                    "augmentation_mode": self.augmentation_mode.value,
                },
            )

        all_questions = []

        for dimension in dimensions:
            # Check if seed questions provided for this dimension
            dim_seed_questions = (
                seed_questions.get(dimension.name) if seed_questions else None
            )

            if dim_seed_questions and not augment:
                # Use seed questions directly without augmentation
                questions = self._convert_seed_to_questions(
                    dim_seed_questions, dimension.name
                )
            elif self.augmentation_mode == AugmentationMode.COMBINED:
                # Paper-faithful: generate seeds, then run both augmentation
                # modes independently, then merge all three pools
                if dim_seed_questions:
                    seeds = self._convert_seed_to_questions(
                        dim_seed_questions, dimension.name
                    )
                else:
                    seeds = self._generate_questions(
                        dimension,
                        mode=AugmentationMode.SEED,
                        task_type=self.task_type,
                    )

                # Tag seeds
                for q in seeds:
                    q["augmentation_source"] = "seed"

                # Run elaboration independently from seeds
                elaborated = self._augment_questions(
                    seeds, dimension,
                    mode=AugmentationMode.ELABORATION,
                    task_type=self.task_type,
                )
                for q in elaborated:
                    q.setdefault("augmentation_source", "elaboration")

                # Run diversification independently from seeds
                diversified = self._augment_questions(
                    seeds, dimension,
                    mode=AugmentationMode.DIVERSIFICATION,
                    task_type=self.task_type,
                )
                for q in diversified:
                    q.setdefault("augmentation_source", "diversification")

                # Merge all three pools, deduplicating by question text
                questions = self._merge_and_deduplicate(seeds, elaborated, diversified)
            elif dim_seed_questions and augment:
                # Augment from seed questions
                seed_list = self._convert_seed_to_questions(
                    dim_seed_questions, dimension.name
                )
                questions = self._augment_questions(
                    seed_list,
                    dimension,
                    mode=self.augmentation_mode,
                    task_type=self.task_type,
                )
            else:
                # Generate questions from dimension definition
                questions = self._generate_questions(
                    dimension,
                    mode=self.augmentation_mode,
                    task_type=self.task_type,
                )

            all_questions.extend(questions)

        # Apply max_questions limit if specified
        if max_questions and len(all_questions) > max_questions:
            all_questions = all_questions[:max_questions]

        # Convert to ChecklistItems
        items = []
        for q in all_questions:
            meta = {
                "dimension": q.get("dimension"),
                "sub_aspect": q.get("sub_aspect"),
            }
            if "augmentation_source" in q:
                meta["augmentation_source"] = q["augmentation_source"]
            item = ChecklistItem(
                id=str(uuid.uuid4()),
                question=q["question"],
                category=q.get("dimension"),
                metadata=meta,
            )
            items.append(item)

        checklist = Checklist(
            id=str(uuid.uuid4()),
            items=items,
            source_method="checkeval",
            generation_level="corpus",
            metadata={
                "dimension_count": len(dimensions),
                "dimensions": [d.name for d in dimensions],
                "augmentation_mode": self.augmentation_mode.value,
                "task_type": self.task_type,
            },
        )

        if verbose:
            print(f"[DeductiveGenerator] Generated {len(checklist.items)} questions from {len(dimensions)} dimensions")

        # Apply filtering if enabled (CheckEval paper §4.2)
        if apply_filtering and len(checklist.items) > 0:
            # Build dimension lookup for consistency checks
            dim_lookup = {d.name: d.definition for d in dimensions}

            # Stage 1 & 2: Alignment + Dimension Consistency (CheckEval-specific)
            before_count = len(checklist.items)
            filtered_items, filter_stats = self._filter_questions(
                checklist.items, dim_lookup, verbose
            )

            # Update checklist with filtered items
            checklist = Checklist(
                id=checklist.id,
                items=filtered_items,
                source_method=checklist.source_method,
                generation_level=checklist.generation_level,
                input=checklist.input,
                metadata={
                    **checklist.metadata,
                    "alignment_filtered": filter_stats["alignment_filtered"],
                    "consistency_filtered": filter_stats["consistency_filtered"],
                },
            )

            if verbose:
                after_count = len(checklist.items)
                print(
                    f"[DeductiveGenerator] Filtering: {before_count} → {after_count} questions "
                    f"({filter_stats['alignment_filtered']} alignment, "
                    f"{filter_stats['consistency_filtered']} consistency)"
                )

            # Stage 3: Redundancy Removal (via Deduplicator)
            if len(checklist.items) > 1:
                before_count = len(checklist.items)
                dedup = Deduplicator(
                    similarity_threshold=0.85,
                    model=self.model,
                    api_key=self.api_key,
                )
                checklist = dedup.refine(checklist)
                if verbose:
                    after_count = len(checklist.items)
                    merged = checklist.metadata.get("clusters_merged", 0)
                    print(f"[DeductiveGenerator] Deduplication: {before_count} → {after_count} questions ({merged} clusters merged)")

        if verbose:
            print(f"[DeductiveGenerator] Final checklist: {len(checklist.items)} questions")

        return checklist

    def generate_grouped(
        self,
        dimensions: List[DeductiveInput],
        **kwargs: Any,
    ) -> Dict[str, "Checklist"]:
        """Generate a checklist and split it by dimension category.

        Convenience wrapper around ``generate().by_category()``.

        Args:
            dimensions: List of DeductiveInput with name, definition, sub_dimensions
            **kwargs: Additional arguments passed to generate()

        Returns:
            Dict mapping dimension name to sub-Checklist
        """
        checklist = self.generate(dimensions=dimensions, **kwargs)
        return checklist.by_category()

    def _generate_questions(
        self,
        dimension: DeductiveInput,
        mode: AugmentationMode,
        task_type: str,
    ) -> List[Dict[str, Any]]:
        """Generate questions from a dimension definition.

        Args:
            dimension: Dimension with name, definition, sub_dimensions
            mode: Augmentation mode (affects how many questions)
            task_type: Type of task being evaluated

        Returns:
            List of question dicts with question, sub_aspect, dimension
        """
        # Determine questions per sub-dimension based on mode
        questions_per_sub = {
            AugmentationMode.SEED: 2,
            AugmentationMode.ELABORATION: 5,
            AugmentationMode.DIVERSIFICATION: 4,
            AugmentationMode.COMBINED: 2,  # Seeds only; augmentation happens separately
        }.get(mode, 2)

        # Format sub-dimensions
        sub_dims = dimension.sub_dimensions or ["General"]
        sub_dims_text = "\n".join(f"- {sd}" for sd in sub_dims)

        # Render prompt
        prompt = self._generate_template.format(
            task_type=task_type,
            dimension_name=dimension.name,
            definition=dimension.definition,
            sub_dimensions=sub_dims_text,
            questions_per_sub=questions_per_sub,
        )

        # Call LLM (request JSON output for reliable parsing with reasoning models)
        client = self._get_or_create_client()
        response = client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        # Parse response
        content = response["choices"][0]["message"]["content"]
        questions = self._parse_questions_response(content)

        # Ensure dimension is set
        for q in questions:
            q["dimension"] = dimension.name
            if "sub_aspect" not in q:
                q["sub_aspect"] = "General"

        return questions

    def _augment_questions(
        self,
        existing_questions: List[Dict[str, Any]],
        dimension: DeductiveInput,
        mode: AugmentationMode,
        task_type: str,
    ) -> List[Dict[str, Any]]:
        """Augment existing questions using the specified mode.

        Args:
            existing_questions: List of existing question dicts
            dimension: Dimension definition
            mode: Augmentation mode
            task_type: Type of task being evaluated

        Returns:
            List of augmented question dicts
        """
        # Format existing questions
        existing_text = "\n".join(
            f"- [{q.get('sub_aspect', 'General')}] {q['question']}"
            for q in existing_questions
        )

        # Get augmentation instructions
        instructions = AUGMENTATION_INSTRUCTIONS.get(mode, "")

        # Render prompt
        prompt = self._augment_template.format(
            task_type=task_type,
            dimension_name=dimension.name,
            definition=dimension.definition,
            existing_questions=existing_text,
            augmentation_mode=mode.value,
            augmentation_instructions=instructions,
        )

        # Call LLM (request JSON output for reliable parsing with reasoning models)
        client = self._get_or_create_client()
        response = client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        # Parse response
        content = response["choices"][0]["message"]["content"]
        questions = self._parse_questions_response(content)

        # Ensure dimension is set
        for q in questions:
            q["dimension"] = dimension.name
            if "sub_aspect" not in q:
                q["sub_aspect"] = "General"

        return questions

    def _convert_seed_to_questions(
        self,
        seed_dict: Dict[str, List[str]],
        dimension_name: str,
    ) -> List[Dict[str, Any]]:
        """Convert seed question dict to list of question dicts.

        Args:
            seed_dict: {sub_aspect: [questions]}
            dimension_name: Name of the dimension

        Returns:
            List of question dicts
        """
        questions = []
        for sub_aspect, question_list in seed_dict.items():
            for q in question_list:
                questions.append(
                    {
                        "question": q,
                        "sub_aspect": sub_aspect,
                        "dimension": dimension_name,
                    }
                )
        return questions

    @staticmethod
    def _merge_and_deduplicate(
        *pools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge multiple question pools, deduplicating by question text.

        Questions from earlier pools take priority (seeds first, then
        elaborated, then diversified). This preserves the augmentation_source
        tag from the first occurrence.

        Args:
            *pools: Variable number of question lists to merge

        Returns:
            Deduplicated list of question dicts
        """
        seen = set()
        merged = []
        for pool in pools:
            for q in pool:
                text = q["question"].strip()
                if text not in seen:
                    seen.add(text)
                    merged.append(q)
        return merged

    def _parse_questions_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract questions.

        Args:
            content: Raw LLM response

        Returns:
            List of question dicts
        """
        # Try to find JSON in the response
        try:
            # Look for JSON block
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "{" in content:
                # Find JSON object
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                return []

            data = json.loads(json_str)

            if isinstance(data, dict) and "questions" in data:
                return data["questions"]
            elif isinstance(data, list):
                return data
            else:
                return []

        except (json.JSONDecodeError, ValueError):
            return []

    def _filter_questions(
        self,
        items: List[ChecklistItem],
        dim_lookup: Dict[str, str],
        verbose: bool = False,
    ) -> tuple[List[ChecklistItem], Dict[str, int]]:
        """Filter questions using CheckEval-specific criteria.

        Implements the CheckEval paper's filtering stages:
        1. Alignment: "YES" should indicate higher quality
        2. Dimension Consistency: Question matches its dimension definition

        Args:
            items: List of checklist items to filter
            dim_lookup: Dict mapping dimension name to definition
            verbose: Print per-item filtering details

        Returns:
            Tuple of (filtered items, stats dict)
        """
        filtered_items = []
        stats = {"alignment_filtered": 0, "consistency_filtered": 0}

        client = self._get_or_create_client()
        for item in items:
            # Get dimension info from item metadata
            dimension_name = item.metadata.get("dimension", "unknown") if item.metadata else "unknown"
            dimension_def = dim_lookup.get(dimension_name, "General evaluation criterion")

            # Build filter prompt
            prompt = self._filter_template.format(
                dimension_name=dimension_name,
                dimension_definition=dimension_def,
                question=item.question,
            )

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "filter_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string"},
                            "alignment_pass": {"type": "boolean"},
                            "dimension_consistent": {"type": "boolean"},
                        },
                        "required": ["reasoning", "alignment_pass", "dimension_consistent"],
                        "additionalProperties": False,
                    },
                },
            }

            try:
                response = client.chat_completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format=response_format,
                )
                result = json.loads(response["choices"][0]["message"]["content"])

                alignment_pass = result.get("alignment_pass", False)
                dimension_consistent = result.get("dimension_consistent", False)

                if alignment_pass and dimension_consistent:
                    # Question passes both checks
                    filtered_items.append(item)
                else:
                    # Track why it was filtered
                    if not alignment_pass:
                        stats["alignment_filtered"] += 1
                    if not dimension_consistent:
                        stats["consistency_filtered"] += 1

            except (json.JSONDecodeError, KeyError, TypeError):
                # On parse error, be conservative and keep the question
                filtered_items.append(item)

        return filtered_items, stats
