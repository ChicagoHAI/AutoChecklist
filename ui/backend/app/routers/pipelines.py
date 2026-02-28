"""CRUD endpoints for saved pipeline configurations stored as JSON files."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.storage import DATA_DIR, list_dir, read_json, write_json

router = APIRouter()

PIPELINES_DIR = DATA_DIR / "pipelines"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreatePipelineRequest(BaseModel):
    """Request body for creating a new pipeline."""
    name: str = Field(..., min_length=1)
    description: str = ""
    generator_class: str = Field(..., pattern=r"^(direct|contrastive)$")
    generator_prompt: str = Field(..., min_length=1)
    scorer_class: str = "batch"  # Kept for backward compat
    scorer_config: Optional[dict] = None
    scorer_prompt: str = ""
    output_format: str = "checklist"


class UpdatePipelineRequest(BaseModel):
    """Request body for updating a pipeline."""
    name: Optional[str] = None
    description: Optional[str] = None
    generator_class: Optional[str] = None
    generator_prompt: Optional[str] = None
    scorer_class: Optional[str] = None
    scorer_config: Optional[dict] = None
    scorer_prompt: Optional[str] = None
    output_format: Optional[str] = None


class PipelineSummary(BaseModel):
    """Summary returned in list endpoint."""
    id: str
    name: str
    description: str
    generator_class: str
    scorer_class: str
    scorer_config: Optional[dict] = None
    created_at: str
    updated_at: str


class PipelineConfig(BaseModel):
    """Full pipeline configuration detail."""
    id: str
    name: str
    description: str
    generator_class: str
    generator_prompt: str
    scorer_class: str
    scorer_config: Optional[dict] = None
    scorer_prompt: str
    output_format: str
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Migration: auto-create scorer_config from legacy scorer_class
# ---------------------------------------------------------------------------

_LEGACY_SCORER_TO_CONFIG = {
    "batch": {"mode": "batch", "primary_metric": "pass", "capture_reasoning": False},
    "item": {"mode": "item", "primary_metric": "pass", "capture_reasoning": True},
    "weighted": {"mode": "item", "primary_metric": "weighted", "capture_reasoning": False},
    "normalized": {"mode": "item", "primary_metric": "normalized", "capture_reasoning": False},
}


def _ensure_scorer_config(data: dict) -> dict:
    """Add scorer_config to pipeline data if missing (auto-migrate from scorer_class)."""
    if "scorer_config" not in data or data["scorer_config"] is None:
        scorer_class = data.get("scorer_class", "batch")
        data["scorer_config"] = _LEGACY_SCORER_TO_CONFIG.get(
            scorer_class,
            {"mode": "batch", "primary_metric": "pass", "capture_reasoning": False},
        )
    return data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=list[PipelineSummary])
async def list_pipelines():
    """List all saved pipelines (summary view)."""
    files = list_dir(PIPELINES_DIR)
    summaries = []
    for f in files:
        data = read_json(f)
        if data is None:
            continue
        data = _ensure_scorer_config(data)
        summaries.append(PipelineSummary(
            id=data.get("id", f.stem),
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            generator_class=data.get("generator_class", "direct"),
            scorer_class=data.get("scorer_class", "batch"),
            scorer_config=data.get("scorer_config"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        ))
    return summaries


@router.post("", response_model=PipelineConfig, status_code=201)
async def create_pipeline(request: CreatePipelineRequest):
    """Save a new pipeline configuration."""
    pipeline_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    pipeline_data = {
        "id": pipeline_id,
        "name": request.name,
        "description": request.description,
        "generator_class": request.generator_class,
        "generator_prompt": request.generator_prompt,
        "scorer_class": request.scorer_class,
        "scorer_config": request.scorer_config,
        "scorer_prompt": request.scorer_prompt,
        "output_format": request.output_format,
        "created_at": now,
        "updated_at": now,
    }
    pipeline_data = _ensure_scorer_config(pipeline_data)

    path = PIPELINES_DIR / f"{pipeline_id}.json"
    write_json(path, pipeline_data)
    return PipelineConfig(**pipeline_data)


@router.get("/{pipeline_id}", response_model=PipelineConfig)
async def get_pipeline(pipeline_id: str):
    """Get full pipeline configuration."""
    path = PIPELINES_DIR / f"{pipeline_id}.json"
    data = read_json(path)
    if data is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    data = _ensure_scorer_config(data)
    return PipelineConfig(**data)


@router.put("/{pipeline_id}", response_model=PipelineConfig)
async def update_pipeline(pipeline_id: str, request: UpdatePipelineRequest):
    """Update an existing pipeline."""
    path = PIPELINES_DIR / f"{pipeline_id}.json"
    data = read_json(path)
    if data is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if request.name is not None:
        data["name"] = request.name
    if request.description is not None:
        data["description"] = request.description
    if request.generator_class is not None:
        data["generator_class"] = request.generator_class
    if request.generator_prompt is not None:
        data["generator_prompt"] = request.generator_prompt
    if request.scorer_class is not None:
        data["scorer_class"] = request.scorer_class
    if request.scorer_config is not None:
        data["scorer_config"] = request.scorer_config
    if request.scorer_prompt is not None:
        data["scorer_prompt"] = request.scorer_prompt
    if request.output_format is not None:
        data["output_format"] = request.output_format

    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data = _ensure_scorer_config(data)
    write_json(path, data)
    return PipelineConfig(**data)


@router.delete("/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline."""
    path = PIPELINES_DIR / f"{pipeline_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pipeline not found")
    path.unlink()
    return {"status": "deleted", "id": pipeline_id}
