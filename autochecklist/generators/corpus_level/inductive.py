"""InductiveGenerator: creates checklists from observations (e.g., feedback comments).

Implements the feedback-based checklist generation pipeline from:
"From Feedback to Checklists" (EMNLP 2025 Industry Track)

Pipeline:
1. Generate: LLM creates questions from observation batches
2. Deduplicate: Merge semantically similar questions
3. Tag: Filter by applicability and specificity
4. (Optional) UnitTest: Validate enforceability
5. Select: Beam search for diverse subset
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...models import Checklist, ChecklistItem
from ...prompts import load_template, PromptTemplate
from ...refiners import Deduplicator, Tagger, Selector, UnitTester
from ..base import CorpusChecklistGenerator


class InductiveGenerator(CorpusChecklistGenerator):
    """Generator that induces checklists from observations.

    Takes a collection of evaluative observations (e.g., reviewer feedback,
    user complaints, quality notes, strengths/weaknesses) and generates a
    comprehensive yes/no checklist that addresses them.

    The pipeline applies multiple refinement steps:
    - Deduplication (merge similar questions)
    - Tagging (filter for applicability)
    - Selection (beam search for diverse subset)
    """

    @property
    def method_name(self) -> str:
        return "feedback"

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        # Refinement parameters
        dedup_threshold: float = 0.85,
        max_questions: int = 20,
        beam_width: int = 5,
        # Batching
        batch_size: int = 100,
        # Unit testing
        max_unit_test_references: int = 20,
        # API keys
        embedding_api_key: Optional[str] = None,
        # Custom prompt
        custom_prompt: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.temperature = temperature
        self.dedup_threshold = dedup_threshold
        self.max_questions = max_questions
        self.beam_width = beam_width
        self.batch_size = batch_size
        self.max_unit_test_references = max_unit_test_references
        self.embedding_api_key = embedding_api_key

        # Load generation prompt template (custom or default)
        if custom_prompt is not None:
            if isinstance(custom_prompt, Path):
                template_str = custom_prompt.read_text(encoding="utf-8")
            else:
                template_str = custom_prompt
        else:
            template_str = load_template("generators/feedback", "generate")
        self._generate_template = PromptTemplate(template_str)

    def generate(
        self,
        observations: List[str],
        domain: str = "general responses",
        skip_dedup: bool = False,
        skip_tagging: bool = False,
        skip_selection: bool = False,
        skip_unit_testing: bool = True,
        references: Optional[List[str]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Checklist:
        """Generate a checklist from observations.

        Args:
            observations: List of evaluative observation strings (feedback,
                review comments, quality notes, etc.)
            domain: Domain description for the prompt
            skip_dedup: Skip deduplication step
            skip_tagging: Skip tagging/filtering step
            skip_selection: Skip subset selection step
            skip_unit_testing: Skip unit testing step (default True)
            references: Optional reference targets for unit testing
            verbose: Print progress at each pipeline stage
            **kwargs: Additional arguments

        Returns:
            Generated and refined Checklist
        """
        if not observations:
            return Checklist(
                items=[],
                source_method="feedback",
                generation_level="corpus",
                metadata={"observation_count": 0},
            )

        # Step 1: Generate initial questions from observations
        raw_questions = self._generate_questions(observations, domain=domain)

        # Convert to checklist items
        items = []
        for i, q in enumerate(raw_questions):
            items.append(
                ChecklistItem(
                    id=f"fb-{i}",
                    question=q["question"],
                    metadata={
                        "source_feedback_indices": q.get("source_feedback_indices", []),
                    },
                )
            )

        checklist = Checklist(
            items=items,
            source_method="feedback",
            generation_level="corpus",
            metadata={
                "observation_count": len(observations),
                "raw_question_count": len(items),
            },
        )

        if verbose:
            print(f"[InductiveGenerator] Generated {len(items)} raw questions from {len(observations)} observations")

        # Step 2: Deduplicate
        if not skip_dedup and len(checklist.items) > 1:
            before_count = len(checklist.items)
            dedup = Deduplicator(
                similarity_threshold=self.dedup_threshold,
                model=self.model,
                api_key=self.api_key,
                embedding_api_key=self.embedding_api_key,
            )
            checklist = dedup.refine(checklist)
            if verbose:
                after_count = len(checklist.items)
                clusters_merged = checklist.metadata.get("clusters_merged", 0)
                print(f"[InductiveGenerator] Deduplication: {before_count} → {after_count} questions ({clusters_merged} clusters merged)")

        # Step 3: Tag and filter
        if not skip_tagging and len(checklist.items) > 0:
            before_count = len(checklist.items)
            tagger = Tagger(
                model=self.model,
                api_key=self.api_key,
            )
            checklist = tagger.refine(checklist)
            if verbose:
                after_count = len(checklist.items)
                filtered_count = checklist.metadata.get("filtered_count", 0)
                print(f"[InductiveGenerator] Tagging: {before_count} → {after_count} questions ({filtered_count} filtered out)")

        # Step 4: Unit test (optional, requires references)
        if not skip_unit_testing and references is not None and len(checklist.items) > 0:
            import random as _random
            before_count = len(checklist.items)
            refs = references
            if len(refs) > self.max_unit_test_references:
                _random.seed(0)
                refs = _random.sample(refs, self.max_unit_test_references)
            raw_samples = [{"id": f"ref-{i}", "text": r} for i, r in enumerate(refs)]
            unit_tester = UnitTester(
                model=self.model,
                api_key=self.api_key,
            )
            checklist = unit_tester.refine(checklist, raw_samples=raw_samples)
            if verbose:
                after_count = len(checklist.items)
                print(f"[InductiveGenerator] Unit testing: {before_count} → {after_count} questions")

        # Step 5: Select diverse subset
        if not skip_selection and len(checklist.items) > self.max_questions:
            before_count = len(checklist.items)
            selector = Selector(
                max_questions=self.max_questions,
                beam_width=self.beam_width,
                embedding_api_key=self.embedding_api_key,
                observations=observations,
            )
            checklist = selector.refine(checklist)
            if verbose:
                after_count = len(checklist.items)
                diversity_score = checklist.metadata.get("diversity_score", 0)
                print(f"[InductiveGenerator] Selection: {before_count} → {after_count} questions (diversity={diversity_score:.3f})")

        if verbose:
            print(f"[InductiveGenerator] Final checklist: {len(checklist.items)} questions")

        return checklist

    def _generate_questions(
        self,
        observations: List[str],
        domain: str = "general responses",
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generate questions from observations using LLM, batched for scalability.

        Splits observations into batches of ``self.batch_size`` items, makes one
        LLM call per batch, and pools results. Observation indices in each batch
        prompt use **global** numbering so ``source_feedback_indices`` are
        correct across batches.

        Args:
            observations: List of observation strings
            domain: Domain description
            verbose: Print per-batch progress

        Returns:
            List of question dicts with 'question' and 'source_feedback_indices'
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "generated_questions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "source_feedback_indices": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
                                },
                                "required": ["question", "source_feedback_indices"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["questions"],
                    "additionalProperties": False,
                },
            },
        }

        client = self._get_or_create_client()
        n_batches = math.ceil(len(observations) / self.batch_size)
        all_questions: List[Dict[str, Any]] = []

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(observations))
            batch = observations[start:end]

            # Format with global indices
            feedback_text = "\n".join(
                f"[{start + i}] {f}" for i, f in enumerate(batch)
            )

            prompt = self._generate_template.format(
                domain=domain,
                feedback=feedback_text,
            )

            response = client.chat_completion(
                model=self.model or "openai/gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,
            )

            content = response["choices"][0]["message"]["content"]

            try:
                result = json.loads(content)
                batch_questions = result.get("questions", [])
            except json.JSONDecodeError:
                batch_questions = []

            all_questions.extend(batch_questions)

            if verbose:
                print(
                    f"[InductiveGenerator] Batch {batch_idx + 1}/{n_batches}: "
                    f"items [{start}..{end - 1}] → {len(batch_questions)} questions"
                )

        return all_questions
