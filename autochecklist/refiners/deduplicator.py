"""Deduplicator refiner for merging semantically similar checklist items.

Uses embedding similarity to find clusters of similar questions, then
uses an LLM to merge each cluster into a single representative question.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    import networkx as nx

from ..models import Checklist, ChecklistItem
from ..prompts import load_template, PromptTemplate
from ..utils.embeddings import get_embeddings, cosine_similarity
from .base import ChecklistRefiner


class Deduplicator(ChecklistRefiner):
    """Refiner that merges semantically similar checklist questions.

    Pipeline:
    1. Compute embeddings for all questions
    2. Build similarity graph (edge if cosine >= threshold)
    3. Find connected components (clusters)
    4. Keep isolated nodes (unique questions) as-is
    5. Use LLM to merge multi-node clusters into single questions
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        custom_prompt: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(model=model, temperature=temperature, api_key=api_key, **kwargs)
        self.similarity_threshold = similarity_threshold
        self.embedding_api_key = embedding_api_key
        # Load merge prompt template
        if custom_prompt is not None:
            if isinstance(custom_prompt, Path):
                template_str = custom_prompt.read_text(encoding="utf-8")
            else:
                template_str = custom_prompt
        else:
            template_str = load_template("generators/feedback", "merge")
        self._merge_template = PromptTemplate(template_str)

    @property
    def refiner_name(self) -> str:
        return "deduplicator"

    def refine(self, checklist: Checklist, **kwargs: Any) -> Checklist:
        """Deduplicate the checklist by merging similar questions.

        Args:
            checklist: Input checklist to deduplicate

        Returns:
            Checklist with similar questions merged
        """
        if len(checklist.items) <= 1:
            return self._create_refined_checklist(
                checklist,
                list(checklist.items),
                metadata_updates={"clusters_merged": 0},
            )

        # Get questions and compute embeddings
        questions = [item.question for item in checklist.items]
        embeddings = get_embeddings(questions, api_key=self.embedding_api_key)

        # Build similarity graph
        graph, clusters = self._build_similarity_graph(
            checklist.items, embeddings
        )

        # Process clusters
        refined_items: List[ChecklistItem] = []
        clusters_merged = 0

        for cluster_ids in clusters:
            cluster_items = [
                item for item in checklist.items if item.id in cluster_ids
            ]

            if len(cluster_items) == 1:
                # Isolated node - keep as-is
                refined_items.append(cluster_items[0])
            else:
                # Multi-node cluster - merge with LLM
                merged_item = self._merge_cluster(cluster_items)
                refined_items.append(merged_item)
                clusters_merged += 1

        return self._create_refined_checklist(
            checklist,
            refined_items,
            metadata_updates={
                "clusters_merged": clusters_merged,
                "similarity_threshold": self.similarity_threshold,
            },
        )

    def _build_similarity_graph(
        self,
        items: List[ChecklistItem],
        embeddings,
    ) -> Tuple[nx.Graph, List[Set[str]]]:
        """Build similarity graph and find connected components.

        Args:
            items: Checklist items
            embeddings: Numpy array of embeddings

        Returns:
            Tuple of (graph, list of component sets)
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for checklist deduplication. "
                "Install with: pip install 'autochecklist[ml]'"
            ) from None

        G = nx.Graph()

        # Add all items as nodes
        for item in items:
            G.add_node(item.id)

        # Compute similarity matrix and add edges
        similarity_matrix = cosine_similarity(embeddings)

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    G.add_edge(items[i].id, items[j].id)

        # Find connected components
        components = list(nx.connected_components(G))

        return G, components

    def _merge_cluster(self, items: List[ChecklistItem]) -> ChecklistItem:
        """Merge a cluster of similar questions into one.

        Args:
            items: List of similar checklist items

        Returns:
            Single merged ChecklistItem
        """
        # Format questions for the prompt
        questions_text = "\n".join(f"- {item.question}" for item in items)

        # Call LLM to merge
        prompt = self._merge_template.format(questions=questions_text)

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "merged_question",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"}
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        }

        response = self._call_model(prompt, response_format=response_format)

        # Parse response
        try:
            result = json.loads(response)
            merged_question = result["question"]
        except (json.JSONDecodeError, KeyError):
            # Fallback: try to extract question from response
            merged_question = response.strip()
            if merged_question.startswith('"') and merged_question.endswith('"'):
                merged_question = merged_question[1:-1]

        # Create merged item
        # Use first item's ID with -merged suffix (like reference impl)
        merged_id = f"{items[0].id}-merged"

        # Average weights if items have weights
        avg_weight = sum(item.weight for item in items) / len(items)

        return ChecklistItem(
            id=merged_id,
            question=merged_question,
            weight=avg_weight,
            metadata={
                "merged_from": [item.id for item in items],
                "original_questions": [item.question for item in items],
            },
        )
