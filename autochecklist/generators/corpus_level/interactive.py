"""InteractiveGenerator - interactive (think-aloud) based checklist generation, based on InteractEval.

InteractiveGenerator generates checklists from think-aloud attributes through a 5-stage pipeline:
1. Component Extraction: Find recurring themes from attributes
2. Attributes Clustering: Group attributes under components
3. Key Question Generation: 1 yes/no question per component
4. Sub-Question Generation: 2-3 sub-questions per component
5. Question Validation: Refine and validate final checklist

Reference: https://arxiv.org/abs/2409.07355 (InteractEval paper)
"""

import json
from typing import Any, Dict, List, Optional
import uuid

from ...models import Checklist, ChecklistItem, InteractiveInput
from ...prompts import load_template, PromptTemplate
from ..base import CorpusChecklistGenerator


class InteractiveGenerator(CorpusChecklistGenerator):
    """Generate checklists from interactive think-aloud attributes.

    InteractEval takes pre-collected think-aloud attributes (considerations
    about evaluating a dimension) and transforms them through a 5-stage
    pipeline into a validated checklist of yes/no questions.

    Args:
        model: OpenRouter model ID for generation
        max_components: Maximum number of components to extract (default 5)

    Example:
        >>> interacteval = InteractiveGenerator(model="openai/gpt-4o-mini")
        >>> input = InteractiveInput(
        ...     source="human_llm",
        ...     dimension="coherence",
        ...     attributes=["Check for logical flow", "Ensure consistency"],
        ... )
        >>> checklist = interacteval.generate(
        ...     inputs=[input],
        ...     rubric="Coherence measures the logical flow and consistency...",
        ... )
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_components: int = 5,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.max_components = max_components

        # Load prompt templates for each pipeline stage
        self._component_extraction_template = PromptTemplate(
            load_template("generators/interacteval", "component_extraction")
        )
        self._attributes_clustering_template = PromptTemplate(
            load_template("generators/interacteval", "attributes_clustering")
        )
        self._question_generation_template = PromptTemplate(
            load_template("generators/interacteval", "question_generation")
        )
        self._sub_question_generation_template = PromptTemplate(
            load_template("generators/interacteval", "sub_question_generation")
        )
        self._question_validation_template = PromptTemplate(
            load_template("generators/interacteval", "question_validation")
        )

    @property
    def method_name(self) -> str:
        """Return the method name for this generator."""
        return "interacteval"

    def generate(
        self,
        inputs: List[InteractiveInput],
        rubric: str = "",
        max_questions: Optional[int] = None,
        **kwargs: Any,
    ) -> Checklist:
        """Generate a checklist from think-aloud attributes.

        Args:
            inputs: List of InteractiveInput with attributes from human/LLM sources
            rubric: Definition/rubric for the evaluation dimension
            max_questions: Maximum number of questions to include
            **kwargs: Additional parameters

        Returns:
            Checklist with generated questions
        """
        if not inputs:
            return self._empty_checklist()

        # Combine attributes from all inputs
        all_attributes = []
        dimension = inputs[0].dimension
        source = inputs[0].source

        for inp in inputs:
            all_attributes.extend(inp.attributes)
            # Use the first non-None dimension
            if not dimension and inp.dimension:
                dimension = inp.dimension
            # Track combined sources
            if inp.source != source:
                source = "human_llm"

        if not all_attributes:
            return self._empty_checklist(dimension=dimension)

        # Run the 5-stage pipeline
        validated_questions = self._run_pipeline(
            all_attributes, rubric, dimension or "quality"
        )

        # Apply max_questions limit if specified
        if max_questions and len(validated_questions) > max_questions:
            validated_questions = validated_questions[:max_questions]

        # Convert to ChecklistItems
        items = []
        for q in validated_questions:
            item = ChecklistItem(
                id=str(uuid.uuid4()),
                question=q,
                category=dimension,
                metadata={
                    "dimension": dimension,
                },
            )
            items.append(item)

        return Checklist(
            id=str(uuid.uuid4()),
            items=items,
            source_method="interacteval",
            generation_level="corpus",
            metadata={
                "dimension": dimension,
                "attribute_count": len(all_attributes),
                "source": source,
                "rubric": rubric[:200] if rubric else None,  # Truncate for metadata
            },
        )

    def _run_pipeline(
        self,
        attributes: List[str],
        rubric: str,
        dimension: str,
    ) -> List[str]:
        """Run the 5-stage pipeline to generate validated questions.

        Args:
            attributes: List of think-aloud attributes
            rubric: Dimension definition/rubric
            dimension: Name of the evaluation dimension

        Returns:
            List of validated question strings
        """
        # Stage 1: Extract components
        components = self._extract_components(attributes, rubric, dimension)
        if not components:
            return []

        # Stage 2: Cluster attributes under components
        clustered = self._cluster_attributes(components, attributes)

        # Stage 3: Generate key questions
        key_questions = self._generate_key_questions(clustered, rubric, dimension)

        # Stage 4: Generate sub-questions
        sub_questions = self._generate_sub_questions(key_questions, rubric, dimension)

        # Stage 5: Validate and refine
        validated = self._validate_questions(sub_questions, rubric, dimension)

        return validated

    def _extract_components(
        self,
        attributes: List[str],
        rubric: str,
        dimension: str,
    ) -> List[str]:
        """Stage 1: Extract recurring components/themes from attributes."""
        attributes_text = "\n".join(f"- {attr}" for attr in attributes)

        prompt = self._component_extraction_template.format(
            dimension=dimension,
            rubric=rubric,
            attributes=attributes_text,
            max_components=self.max_components,
        )

        client = self._get_or_create_client()
        response = client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]
        return self._parse_list_response(content)

    def _cluster_attributes(
        self,
        components: List[str],
        attributes: List[str],
    ) -> Dict[str, List[str]]:
        """Stage 2: Cluster attributes under components."""
        components_text = "\n".join(f"- {c}" for c in components)
        attributes_text = "\n".join(f"- {attr}" for attr in attributes)

        prompt = self._attributes_clustering_template.format(
            components=components_text,
            attributes=attributes_text,
        )

        client = self._get_or_create_client()
        response = client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]
        return self._parse_dict_response(content)

    def _generate_key_questions(
        self,
        clustered: Dict[str, List[str]],
        rubric: str,
        dimension: str,
    ) -> Dict[str, str]:
        """Stage 3: Generate key questions for each component."""
        components_attributes_text = ""
        for component, attrs in clustered.items():
            components_attributes_text += f"\n**{component}**:\n"
            for attr in attrs:
                components_attributes_text += f"  - {attr}\n"

        prompt = self._question_generation_template.format(
            dimension=dimension,
            rubric=rubric,
            components_attributes=components_attributes_text,
        )

        client = self._get_or_create_client()
        response = client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]
        return self._parse_dict_response(content)

    def _generate_sub_questions(
        self,
        key_questions: Dict[str, str],
        rubric: str,
        dimension: str,
    ) -> Dict[str, List[str]]:
        """Stage 4: Generate sub-questions for each key question."""
        key_questions_text = ""
        for component, question in key_questions.items():
            key_questions_text += f"\n**{component}**: {question}\n"

        prompt = self._sub_question_generation_template.format(
            dimension=dimension,
            rubric=rubric,
            key_questions=key_questions_text,
        )

        client = self._get_or_create_client()
        response = client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]
        return self._parse_dict_list_response(content)

    def _validate_questions(
        self,
        sub_questions: Dict[str, List[str]],
        rubric: str,
        dimension: str,
    ) -> List[str]:
        """Stage 5: Validate and refine questions."""
        # Flatten all questions
        all_questions = []
        for component, questions in sub_questions.items():
            all_questions.extend(questions)

        all_questions_text = "\n".join(f"- {q}" for q in all_questions)

        prompt = self._question_validation_template.format(
            dimension=dimension,
            rubric=rubric,
            all_questions=all_questions_text,
        )

        client = self._get_or_create_client()
        response = client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # Lower temperature for validation
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]
        return self._parse_list_response(content)

    def _empty_checklist(self, dimension: str = None) -> Checklist:
        """Return an empty checklist."""
        return Checklist(
            id=str(uuid.uuid4()),
            items=[],
            source_method="interacteval",
            generation_level="corpus",
            metadata={
                "dimension": dimension,
                "attribute_count": 0,
                "source": None,
            },
        )

    def _parse_list_response(self, content: str) -> List[str]:
        """Parse LLM response to extract a list of strings."""
        try:
            # Try to find JSON in the response
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
            else:
                json_str = content.strip()

            data = json.loads(json_str)
            if isinstance(data, list):
                return [str(item) for item in data]
            # Handle JSON objects containing a list value (e.g., {"components": [...]})
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list) and value:
                        return [str(item) for item in value]
            return []

        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract a JSON array via bracket matching
            if "[" in content:
                try:
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    arr = json.loads(content[start:end])
                    if isinstance(arr, list):
                        return [str(item) for item in arr]
                except (json.JSONDecodeError, ValueError):
                    pass
            return []

    def _parse_dict_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response to extract a dictionary."""
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                return {}

            data = json.loads(json_str)
            if isinstance(data, dict):
                return data
            return {}

        except (json.JSONDecodeError, ValueError):
            return {}

    def _parse_dict_list_response(self, content: str) -> Dict[str, List[str]]:
        """Parse LLM response to extract a dict of lists."""
        result = self._parse_dict_response(content)
        # Ensure all values are lists
        for key, value in result.items():
            if not isinstance(value, list):
                result[key] = [value] if value else []
        return result
