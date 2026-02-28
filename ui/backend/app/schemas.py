"""Pydantic request/response models for the API."""

from typing import Optional

from pydantic import BaseModel, Field


class ChecklistItem(BaseModel):
    """A single checklist item."""

    question: str
    weight: float = 1.0
    passed: Optional[bool] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class Checklist(BaseModel):
    """A checklist with items and metadata."""

    items: list[ChecklistItem]
    source_method: str
    metadata: dict = Field(default_factory=dict)


class ScoreResult(BaseModel):
    """Score result for a checklist evaluation."""

    score: float
    max_score: float
    percentage: float
    primary_metric: str = "pass"  # "pass", "weighted", "normalized"
    item_results: list[dict] = Field(default_factory=list)


class MethodResult(BaseModel):
    """Result from a single evaluation method."""

    checklist: Checklist
    score: ScoreResult


class EvaluateRequest(BaseModel):
    """Request body for the evaluate endpoint."""

    input: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    reference: Optional[str] = None
    model: str = "openai/gpt-5-mini"
    methods: Optional[list[str]] = None
    scorer: Optional[str] = None
    scorer_config: Optional[dict] = None
    provider: Optional[str] = None
    generator_provider: Optional[str] = None
    generator_model: Optional[str] = None
    scorer_provider: Optional[str] = None
    scorer_model: Optional[str] = None
    custom_prompt: Optional[str] = None
    custom_scorer_prompt: Optional[str] = None
    candidate_model: Optional[str] = None
    pipeline_ids: Optional[list[str]] = None


# ---- Checklist Builder schemas ----


class DeductiveInputSchema(BaseModel):
    """A single dimension for DeductiveGenerator."""

    name: str = Field(..., min_length=1)
    definition: str = Field(..., min_length=1)
    sub_dimensions: list[str] = Field(default_factory=list)


class DeductiveGenerateRequest(BaseModel):
    """Request for deductive (dimension-based) checklist generation."""

    dimensions: list[DeductiveInputSchema] = Field(..., min_length=1)
    task_type: str = "general"
    augmentation_mode: str = "seed"
    provider: Optional[str] = None
    model: Optional[str] = None


class DeductiveGenerateResponse(BaseModel):
    """Response from deductive (dimension-based) checklist generation."""

    checklist: Checklist


class EvaluateResponse(BaseModel):
    """Response from the evaluate endpoint. Keys are method names."""

    results: dict[str, MethodResult]


class GenerateRequest(BaseModel):
    """Request for generate-only endpoint."""

    input: str = Field(..., min_length=1)
    target: Optional[str] = None
    reference: Optional[str] = None
    method: str = "tick"
    provider: Optional[str] = None
    model: Optional[str] = None
    custom_prompt: Optional[str] = None
    candidate_model: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response from generate-only endpoint."""

    method: str
    checklist: dict  # Serialized Checklist


class ScoreRequest(BaseModel):
    """Request for score-only endpoint."""

    input: Optional[str] = None
    target: str = Field(..., min_length=1)
    checklist_items: list[dict]  # List of {question, weight?}
    scorer: str = "batch"
    scorer_config: Optional[dict] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    custom_scorer_prompt: Optional[str] = None


class ScoreResponse(BaseModel):
    """Response from score-only endpoint."""

    score: dict  # Serialized Score


class StreamRequest(BaseModel):
    """Request for streaming evaluation."""

    input: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    reference: Optional[str] = None
    methods: Optional[list[str]] = None  # If None, auto-select based on reference
    scorer: Optional[str] = None  # If None, use default per method
    scorer_config: Optional[dict] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    generator_provider: Optional[str] = None
    generator_model: Optional[str] = None
    scorer_provider: Optional[str] = None
    scorer_model: Optional[str] = None
    custom_prompt: Optional[str] = None
    custom_scorer_prompt: Optional[str] = None
    candidate_model: Optional[str] = None
    pipeline_ids: Optional[list[str]] = None
