"""UnitTester refiner for validating LLM enforceability of checklist questions.

For each question, rewrites passing samples to fail, then verifies the LLM
correctly scores them as failing. Questions with low enforceability rates
are filtered out.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..models import Checklist, ChecklistItem
from ..prompts import load_template, PromptTemplate
from .base import ChecklistRefiner


class UnitTester(ChecklistRefiner):
    """Refiner that validates questions via unit test rewrites.

    Pipeline:
    1. For each question, find samples that pass (answer=Yes)
    2. LLM rewrites each sample to fail the criterion
    3. Score rewritten samples - should get "No"
    4. Enforceability rate = proportion of rewrites correctly failing
    5. Filter questions below threshold
    """

    def __init__(
        self,
        enforceability_threshold: float = 0.7,
        max_samples: int = 10,
        model: Optional[str] = None,
        scorer_model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        custom_prompt: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(model=model, temperature=temperature, api_key=api_key, **kwargs)
        self.enforceability_threshold = enforceability_threshold
        self.max_samples = max_samples
        self.scorer_model = scorer_model or model
        # Load rewrite prompt template
        if custom_prompt is not None:
            if isinstance(custom_prompt, Path):
                template_str = custom_prompt.read_text(encoding="utf-8")
            else:
                template_str = custom_prompt
        else:
            template_str = load_template("generators/feedback", "rewrite_fail")
        self._rewrite_template = PromptTemplate(template_str)

    @property
    def refiner_name(self) -> str:
        return "unit_tester"

    def refine(
        self,
        checklist: Checklist,
        samples: Optional[List[Dict[str, Any]]] = None,
        sample_scores: Optional[Dict[str, Dict[str, str]]] = None,
        raw_samples: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Checklist:
        """Filter checklist based on LLM enforceability.

        Args:
            checklist: Input checklist to validate
            samples: List of sample dicts with 'id' and 'text' keys
            sample_scores: Dict mapping sample_id -> {question_id -> "Yes"/"No"}
            raw_samples: Samples to auto-score when sample_scores not provided.
                         Each dict must have 'id' and 'text' keys.

        Returns:
            Checklist with only enforceable questions
        """
        # Auto-score raw_samples if provided without sample_scores
        if raw_samples and not sample_scores:
            sample_scores = self._auto_score_samples(checklist, raw_samples)
            samples = raw_samples
        else:
            samples = samples or []
            sample_scores = sample_scores or {}

        if len(checklist.items) == 0 or len(samples) == 0:
            return self._create_refined_checklist(
                checklist,
                [],
                metadata_updates={
                    "filtered_count": len(checklist.items),
                    "enforceability_rates": {},
                },
            )

        passing_items: List[ChecklistItem] = []
        enforceability_rates: Dict[str, float] = {}
        filtered_count = 0

        for item in checklist.items:
            # Find samples that pass this question
            passing_samples = self._get_passing_samples(
                item.id, samples, sample_scores
            )

            if len(passing_samples) == 0:
                # No passing samples - can't test enforceability
                filtered_count += 1
                enforceability_rates[item.id] = 0.0
                continue

            # Limit samples
            if len(passing_samples) > self.max_samples:
                random.seed(0)  # Reproducibility
                passing_samples = random.sample(passing_samples, self.max_samples)

            # Test enforceability
            rate = self._compute_enforceability(item, passing_samples)
            enforceability_rates[item.id] = rate

            if rate >= self.enforceability_threshold:
                # Question passes - add enforceability metadata
                item_with_metadata = ChecklistItem(
                    id=item.id,
                    question=item.question,
                    weight=item.weight,
                    category=item.category,
                    metadata={
                        **(item.metadata or {}),
                        "enforceability_rate": rate,
                        "samples_tested": len(passing_samples),
                    },
                )
                passing_items.append(item_with_metadata)
            else:
                filtered_count += 1

        return self._create_refined_checklist(
            checklist,
            passing_items,
            metadata_updates={
                "filtered_count": filtered_count,
                "enforceability_rates": enforceability_rates,
                "enforceability_threshold": self.enforceability_threshold,
            },
        )

    def _get_passing_samples(
        self,
        question_id: str,
        samples: List[Dict[str, Any]],
        sample_scores: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Get samples that pass a specific question."""
        passing = []
        for sample in samples:
            sample_id = sample["id"]
            if sample_id in sample_scores:
                if sample_scores[sample_id].get(question_id) == "Yes":
                    passing.append(sample)
        return passing

    def _compute_enforceability(
        self,
        item: ChecklistItem,
        passing_samples: List[Dict[str, Any]],
    ) -> float:
        """Compute enforceability rate for a question.

        Args:
            item: Checklist item to test
            passing_samples: Samples that pass this question

        Returns:
            Enforceability rate (0.0 to 1.0)
        """
        if len(passing_samples) == 0:
            return 0.0

        correct_failures = 0
        for sample in passing_samples:
            # Rewrite sample to fail
            rewritten = self._rewrite_sample(item.question, sample["text"])

            # Score rewritten sample
            score = self._score_sample(item.id, item.question, rewritten)

            if score == "No":
                correct_failures += 1

        return correct_failures / len(passing_samples)

    def _rewrite_sample(self, question: str, sample_text: str) -> str:
        """Rewrite a passing sample to fail the criterion.

        Args:
            question: The checklist question
            sample_text: The sample text to rewrite

        Returns:
            Rewritten sample that should fail
        """
        prompt = self._rewrite_template.format(
            question=question,
            sample=sample_text,
        )

        response = self._call_model(prompt)
        return response.strip()

    def _score_sample(
        self,
        question_id: str,
        question: str,
        sample_text: str,
    ) -> str:
        """Score a sample against a question.

        Args:
            question_id: ID of the question
            question: The checklist question
            sample_text: The sample to score

        Returns:
            "Yes" or "No"
        """
        prompt = f"""Evaluate the following sample against the criterion.

Question: {question}

Sample:
{sample_text}

Does this sample meet the criterion? Answer only "Yes" or "No"."""

        response = self._call_model(prompt)
        response = response.strip().lower()

        # Normalize response
        if "yes" in response:
            return "Yes"
        elif "no" in response:
            return "No"
        else:
            # Ambiguous - treat as failure (conservative)
            return "No"

    def _auto_score_samples(
        self,
        checklist: Checklist,
        raw_samples: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        """Score each sample against every checklist question.

        Builds the sample_scores dict that ``refine()`` normally expects as
        a pre-computed input.

        Args:
            checklist: Checklist with items to score against
            raw_samples: Sample dicts with 'id' and 'text' keys

        Returns:
            Dict mapping sample_id -> {question_id -> "Yes"/"No"}
        """
        scores: Dict[str, Dict[str, str]] = {}
        for sample in raw_samples:
            sample_id = sample["id"]
            sample_text = sample["text"]
            scores[sample_id] = {}
            for item in checklist.items:
                result = self._score_sample(item.id, item.question, sample_text)
                scores[sample_id][item.id] = result
        return scores
