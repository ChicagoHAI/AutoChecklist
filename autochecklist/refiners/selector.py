"""Selector refiner for optimal subset selection using beam search.

Selects a diverse subset of checklist questions using embedding similarity.
Uses beam search to explore multiple candidate subsets.

Without observations: Score(subset) = Diversity - λ·|subset|
With observations:    Score(k) = α·Coverage + (1-α)·Diversity - λ·k

- Diversity = 1 - average pairwise cosine similarity
- Coverage = fraction of observations covered by selected questions
- α = trade-off between coverage and diversity (0 = diversity only, 1 = coverage only)
- λ = length penalty, k = subset size

"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    import numpy as np

from ..models import Checklist, ChecklistItem
from ..prompts import load_template, PromptTemplate
from ..utils.embeddings import get_embeddings, cosine_similarity
from .base import ChecklistRefiner


class Selector(ChecklistRefiner):
    """Refiner that selects optimal diverse subset via beam search.

    Since we lack source feedback mapping, uses embedding diversity only.
    Beam search explores multiple candidate subsets to find optimal score.
    """

    def __init__(
        self,
        max_questions: int = 20,
        beam_width: int = 5,
        length_penalty: float = 0.0005,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        # Coverage-aware selection
        observations: Optional[List[str]] = None,
        classifier_model: Optional[str] = None,
        alpha: float = 0.5,
        custom_prompt: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(model=model, temperature=temperature, api_key=api_key, **kwargs)
        self.max_questions = max_questions
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.embedding_api_key = embedding_api_key
        # Coverage
        self.observations = observations
        self.classifier_model = classifier_model or "openai/gpt-4o-mini"
        self.alpha = alpha
        # Populated during refine() when observations are provided
        self._feedback_assignment: Dict[int, Set[int]] = {}
        self._total_feedback_count: int = 0
        # Load classify template
        if custom_prompt is not None:
            if isinstance(custom_prompt, Path):
                template_str = custom_prompt.read_text(encoding="utf-8")
            else:
                template_str = custom_prompt
        else:
            template_str = load_template("generators/feedback", "classify")
        self._classify_template = PromptTemplate(template_str)

    @property
    def refiner_name(self) -> str:
        return "selector"

    def refine(self, checklist: Checklist, **kwargs: Any) -> Checklist:
        """Select optimal diverse subset of questions.

        Args:
            checklist: Input checklist to select from

        Returns:
            Checklist with selected subset
        """
        if len(checklist.items) == 0:
            return self._create_refined_checklist(
                checklist,
                [],
                metadata_updates={
                    "diversity_score": 0.0,
                    "beam_width": self.beam_width,
                },
            )

        # Classify feedback if provided (before beam search)
        if self.observations:
            self._classify_feedback(checklist)

        # If already small enough, return as-is
        if len(checklist.items) <= self.max_questions:
            diversity = self._compute_diversity(
                list(range(len(checklist.items))),
                self._get_similarity_matrix(checklist),
            )
            return self._create_refined_checklist(
                checklist,
                list(checklist.items),
                metadata_updates={
                    "diversity_score": diversity,
                    "beam_width": self.beam_width,
                },
            )

        # Get embeddings and similarity matrix
        similarity_matrix = self._get_similarity_matrix(checklist)

        # Run beam search
        selected_indices, diversity_score = self._beam_search(
            len(checklist.items),
            similarity_matrix,
        )

        # Build selected items with metadata
        selected_items = []
        for order, idx in enumerate(selected_indices):
            item = checklist.items[idx]
            selected_items.append(
                ChecklistItem(
                    id=item.id,
                    question=item.question,
                    weight=item.weight,
                    category=item.category,
                    metadata={
                        **(item.metadata or {}),
                        "selection_order": order,
                    },
                )
            )

        return self._create_refined_checklist(
            checklist,
            selected_items,
            metadata_updates={
                "diversity_score": diversity_score,
                "beam_width": self.beam_width,
                "length_penalty": self.length_penalty,
            },
        )

    def _get_similarity_matrix(self, checklist: Checklist) -> np.ndarray:
        """Compute similarity matrix for checklist questions."""
        questions = [item.question for item in checklist.items]
        embeddings = get_embeddings(questions, api_key=self.embedding_api_key)
        return cosine_similarity(embeddings)

    def _beam_search(
        self,
        n_items: int,
        similarity_matrix: np.ndarray,
    ) -> Tuple[List[int], float]:
        """Run beam search to find optimal subset.

        Args:
            n_items: Total number of items
            similarity_matrix: Pairwise similarity matrix

        Returns:
            Tuple of (selected indices, diversity score)
        """
        # Initialize beam with empty set
        # Each candidate is (frozenset of indices, score)
        beam: List[Tuple[frozenset, float]] = [(frozenset(), 0.0)]

        for step in range(self.max_questions):
            candidates = []

            for current_set, _ in beam:
                # Try adding each available item
                available = set(range(n_items)) - current_set

                for idx in available:
                    new_set = current_set | {idx}
                    score = self._score_subset(list(new_set), similarity_matrix)
                    candidates.append((new_set, score))

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[: self.beam_width]

        # Return best candidate
        if beam:
            best_set, best_score = max(beam, key=lambda x: x[1])
            selected = sorted(best_set)  # Sort for consistent ordering
            diversity = self._compute_diversity(selected, similarity_matrix)
            return selected, diversity
        else:
            return [], 0.0

    def _score_subset(
        self,
        indices: List[int],
        similarity_matrix: np.ndarray,
    ) -> float:
        """Score a subset based on diversity, coverage, and length penalty.

        Without observations: Score = Diversity - λ·k
        With observations:    Score = α·Coverage + (1-α)·Diversity - λ·k
        """
        if len(indices) == 0:
            return 0.0

        diversity = self._compute_diversity(indices, similarity_matrix)
        penalty = self.length_penalty * len(indices)

        if self._feedback_assignment:
            coverage = self._compute_coverage(indices)
            return (
                self.alpha * coverage
                + (1 - self.alpha) * diversity
                - penalty
            )

        return diversity - penalty

    def _compute_diversity(
        self,
        indices: List[int],
        similarity_matrix: np.ndarray,
    ) -> float:
        """Compute diversity score for a subset.

        Diversity = 1 - average pairwise similarity
        """
        if len(indices) <= 1:
            return 1.0  # Single item is maximally diverse

        total_sim = 0.0
        count = 0

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_sim += similarity_matrix[indices[i], indices[j]]
                count += 1

        avg_sim = total_sim / count if count > 0 else 0.0
        return 1.0 - avg_sim

    def _classify_feedback(self, checklist: Checklist) -> Dict[int, Set[int]]:
        """Classify each feedback item against the current checklist questions.

        For each feedback item, asks an LLM which questions cover it.
        Builds a mapping from question index to set of feedback indices.

        Args:
            checklist: Current checklist to classify against

        Returns:
            Dict mapping question index -> set of feedback indices it covers
        """
        if not self.observations:
            return {}

        # Format questions as numbered list (0-indexed)
        questions_text = "\n".join(
            f"{i}. {item.question}" for i, item in enumerate(checklist.items)
        )

        assignment: Dict[int, Set[int]] = {
            i: set() for i in range(len(checklist.items))
        }

        for fb_idx, fb_item in enumerate(self.observations):
            prompt = self._classify_template.format(
                questions=questions_text,
                feedback_item=fb_item,
            )
            response = self._call_model(prompt)

            # Parse question numbers from response
            nums = re.findall(r"\d+", response)
            for num_str in nums:
                q_idx = int(num_str)
                if 0 <= q_idx < len(checklist.items):
                    assignment[q_idx].add(fb_idx)

        self._feedback_assignment = assignment
        self._total_feedback_count = len(self.observations)
        return assignment

    def _compute_coverage(self, indices: List[int]) -> float:
        """Compute fraction of feedback items covered by selected questions.

        Args:
            indices: Indices of selected questions

        Returns:
            Coverage score between 0.0 and 1.0
        """
        if self._total_feedback_count == 0:
            return 0.0

        covered: Set[int] = set()
        for idx in indices:
            covered |= self._feedback_assignment.get(idx, set())

        return len(covered) / self._total_feedback_count
