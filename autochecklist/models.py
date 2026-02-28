"""Core data models for autochecklist."""

from datetime import datetime, timezone
from enum import Enum
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChecklistItemAnswer(str, Enum):
    """Possible answers for a checklist item."""
    YES = "yes"
    NO = "no"


class ConfidenceLevel(str, Enum):
    """Confidence levels for normalized scoring (RocketEval-style)."""
    NO_10 = "no_10"    # No (10% confidence)
    NO_30 = "no_30"    # No (30% confidence)
    UNSURE = "unsure"  # Unsure (50%)
    YES_70 = "yes_70"  # Yes (70% confidence)
    YES_90 = "yes_90"  # Yes (90% confidence)


class ChecklistItem(BaseModel):
    """A single checklist item (yes/no question)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str
    weight: float = Field(default=100.0, ge=0.0, le=100.0)  # RLCF uses 0-100
    category: Optional[str] = None  # For grouping (e.g., "factuality", "format")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class Checklist(BaseModel):
    """A collection of checklist items."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    items: List[ChecklistItem]
    source_method: str  # e.g., "tick", "rlcf", "checkeval"
    generation_level: str  # "instance" or "corpus"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional context
    input: Optional[str] = None  # For instance-level
    corpus_description: Optional[str] = None  # For corpus-level

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def by_category(self) -> Dict[str, "Checklist"]:
        """Group items by category, returning a dict of sub-checklists.

        Items with ``category=None`` are placed under ``"ungrouped"``.
        Key order matches the first occurrence of each category.
        """
        groups: OrderedDict[str, List[ChecklistItem]] = OrderedDict()
        for item in self.items:
            key = item.category or "ungrouped"
            groups.setdefault(key, []).append(item)
        return {
            name: Checklist(
                id=f"{self.id}_{name}",
                items=items,
                source_method=self.source_method,
                generation_level=self.generation_level,
                input=self.input,
                corpus_description=self.corpus_description,
                metadata={**self.metadata, "parent_id": self.id, "category": name},
            )
            for name, items in groups.items()
        }

    def save(self, path: str) -> None:
        """Save checklist to JSON file."""
        from .utils import save_json
        save_json(path, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: str) -> "Checklist":
        """Load checklist from JSON file."""
        from .utils import load_json
        return cls.model_validate(load_json(path))

    def to_text(self, numbered: bool = True) -> str:
        """Format checklist as text for prompts."""
        if numbered:
            return "\n".join(
                f"{i+1}. {item.question}"
                for i, item in enumerate(self.items)
            )
        return "\n".join(f"- {item.question}" for item in self.items)


class ItemScore(BaseModel):
    """Score for a single checklist item."""

    item_id: str
    answer: ChecklistItemAnswer
    confidence: Optional[float] = None  # 0-1, for normalized scoring
    confidence_level: Optional[ConfidenceLevel] = None  # RocketEval-style
    reasoning: Optional[str] = None


class Score(BaseModel):
    """Complete scoring result."""

    checklist_id: str
    response_id: Optional[str] = None

    item_scores: List[ItemScore]

    # Aggregated scores
    total_score: float  # 0-1, proportion of "yes"
    weighted_score: Optional[float] = None  # If weights are used (RLCF-style)
    normalized_score: Optional[float] = None  # RocketEval-style

    # Which metric Score.primary_score aliases
    primary_metric: str = "pass"  # "pass", "weighted", "normalized"

    @field_validator("primary_metric")
    @classmethod
    def _validate_primary_metric(cls, v: str) -> str:
        if v not in ("pass", "weighted", "normalized"):
            raise ValueError(
                f"primary_metric must be one of 'pass', 'weighted', 'normalized', got '{v}'"
            )
        return v

    # Metadata
    judge_model: str
    scoring_method: str  # e.g., "batch", "item", "normalized", "weighted"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Proportion of items answered 'yes'."""
        yes_count = sum(1 for s in self.item_scores if s.answer == ChecklistItemAnswer.YES)
        total = len(self.item_scores)
        return yes_count / total if total > 0 else 0.0

    @property
    def primary_score(self) -> float:
        """Returns whichever metric the pipeline designated as primary."""
        if self.primary_metric == "weighted":
            return self.weighted_score if self.weighted_score is not None else self.pass_rate
        elif self.primary_metric == "normalized":
            return self.normalized_score if self.normalized_score is not None else self.pass_rate
        return self.pass_rate

    @property
    def scaled_score_1_5(self) -> float:
        """Scale pass_rate to 1-5 range (InteractEval style).

        Formula: score = pass_rate * 4 + 1
        - pass_rate=0.0 → score=1.0
        - pass_rate=0.5 → score=3.0
        - pass_rate=1.0 → score=5.0
        """
        return self.pass_rate * 4 + 1


class GroupedScore(BaseModel):
    """Scoring result grouped by category (e.g., per-dimension scores).

    Produced when scoring sub-checklists from ``Checklist.by_category()``.
    """

    scores: Dict[str, Score]  # category_name -> Score
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def per_group_pass_rates(self) -> Dict[str, float]:
        """Pass rate for each category."""
        return {name: score.pass_rate for name, score in self.scores.items()}

    @property
    def pass_rate(self) -> float:
        """Macro-averaged pass rate (each category weighted equally)."""
        if not self.scores:
            return 0.0
        return sum(s.pass_rate for s in self.scores.values()) / len(self.scores)

    @property
    def micro_pass_rate(self) -> float:
        """Micro-averaged pass rate (pooled across all items)."""
        total_yes = 0
        total = 0
        for score in self.scores.values():
            for item_score in score.item_scores:
                total += 1
                if item_score.answer == ChecklistItemAnswer.YES:
                    total_yes += 1
        return total_yes / total if total > 0 else 0.0

    @property
    def mean_score(self) -> float:
        """Mean of Score.primary_score across all categories."""
        if not self.scores:
            return 0.0
        return sum(s.primary_score for s in self.scores.values()) / len(self.scores)

    def flatten(self) -> Score:
        """Merge all sub-scores into a single Score."""
        all_item_scores: List[ItemScore] = []
        judge_model = "unknown"
        scoring_method = "batch"
        for score in self.scores.values():
            all_item_scores.extend(score.item_scores)
            judge_model = score.judge_model
            scoring_method = score.scoring_method
        total_yes = sum(
            1 for s in all_item_scores if s.answer == ChecklistItemAnswer.YES
        )
        total = len(all_item_scores)
        return Score(
            checklist_id="grouped",
            item_scores=all_item_scores,
            total_score=total_yes / total if total > 0 else 0.0,
            judge_model=judge_model,
            scoring_method=scoring_method,
        )


# Input models for corpus-level generation

class FeedbackInput(BaseModel):
    """Input for feedback-based checklist generation."""
    feedback_text: str
    source: Optional[str] = None  # e.g., "user", "expert"
    category: Optional[str] = None


class DeductiveInput(BaseModel):
    """Input for deductive (dimension-based) generation (CheckEval, InteractEval)."""
    name: str
    definition: str
    sub_dimensions: Optional[List[str]] = None


class InteractiveInput(BaseModel):
    """Input for interactive (think-aloud based) generation (InteractEval)."""
    source: str  # "human" or "llm"
    dimension: str
    attributes: List[str]
    sample_context: Optional[str] = None


# ── LLM Response Schemas (wire-format contracts for structured output) ──


class GeneratedQuestion(BaseModel):
    """A single generated yes/no question (unweighted)."""
    question: str


class GeneratedWeightedQuestion(BaseModel):
    """A single generated yes/no question with importance weight."""
    question: str
    weight: int = Field(ge=0, le=100)


class ChecklistResponse(BaseModel):
    """LLM response schema for unweighted checklist generation (TICK, RocketEval)."""
    questions: List[GeneratedQuestion]


class WeightedChecklistResponse(BaseModel):
    """LLM response schema for weighted checklist generation (RLCF)."""
    questions: List[GeneratedWeightedQuestion]


class BatchAnswerItem(BaseModel):
    """A single answer in batch scoring (1-based question index)."""
    question_index: int
    answer: Literal["YES", "NO"]


class BatchScoringResponse(BaseModel):
    """LLM response schema for batch scoring (all items at once)."""
    answers: List[BatchAnswerItem]


class BatchAnswerItemReasoned(BaseModel):
    """A single answer with reasoning in batch scoring (1-based question index)."""
    question_index: int
    answer: Literal["YES", "NO"]
    reasoning: str


class BatchScoringResponseReasoned(BaseModel):
    """LLM response schema for batch scoring with per-item reasoning."""
    answers: List[BatchAnswerItemReasoned]


class ItemScoringResponse(BaseModel):
    """LLM response schema for per-item scoring (answer only)."""
    answer: Literal["YES", "NO"]


class ItemScoringResponseReasoned(BaseModel):
    """LLM response schema for per-item scoring with reasoning."""
    answer: Literal["YES", "NO"]
    reasoning: str


# Deprecated aliases — kept for backward compatibility
WeightedScoringResponse = ItemScoringResponse


def _make_strict_schema(schema: dict) -> dict:
    """Post-process a Pydantic JSON schema for OpenAI strict mode.

    OpenAI's structured output with ``strict: true`` requires:
    - ``additionalProperties: false`` on every object
    - No unsupported keywords (``title``, ``description`` are tolerated
      at the top level but must be present on all objects for consistency)

    This recursively patches all object definitions in the schema.
    """
    schema = schema.copy()

    def _patch_object(obj: dict) -> dict:
        obj = obj.copy()
        if obj.get("type") == "object":
            obj["additionalProperties"] = False
        # Recurse into properties
        if "properties" in obj:
            obj["properties"] = {
                k: _patch_object(v) for k, v in obj["properties"].items()
            }
        # Recurse into items (arrays)
        if "items" in obj and isinstance(obj["items"], dict):
            obj["items"] = _patch_object(obj["items"])
        return obj

    # Patch $defs
    if "$defs" in schema:
        schema["$defs"] = {
            k: _patch_object(v) for k, v in schema["$defs"].items()
        }

    return _patch_object(schema)


def to_response_format(model_cls: type[BaseModel], name: str) -> dict:
    """Convert a Pydantic model class to an OpenAI-compatible response_format dict.

    Args:
        model_cls: Pydantic model class defining the JSON schema.
        name: Name for the schema (e.g., "checklist_response").

    Returns:
        Dict suitable for passing as ``response_format`` to OpenAI-compatible APIs.
        The schema is post-processed for OpenAI strict mode compatibility.
    """
    raw_schema = model_cls.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": _make_strict_schema(raw_schema),
        },
    }
