"""CRUD endpoints for saved prompt templates stored as JSON files."""

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.storage import DATA_DIR, list_dir, read_json, write_json

from autochecklist.generators.instance_level.pipeline_presets import PIPELINE_PRESETS
from autochecklist.prompts import load_template

router = APIRouter()

TEMPLATES_DIR = DATA_DIR / "prompt_templates"

# Regex to find placeholders like {input}, {target}, {reference}
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
KNOWN_PLACEHOLDERS = {"input", "target", "reference"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreatePromptTemplateRequest(BaseModel):
    """Request body for creating a new prompt template."""
    name: str = Field(..., min_length=1)
    prompt_text: str = Field(..., min_length=1)
    description: str = ""
    metadata: dict = Field(default_factory=dict)


class UpdatePromptTemplateRequest(BaseModel):
    """Request body for updating a prompt template."""
    name: Optional[str] = None
    prompt_text: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None


class PromptTemplateSummary(BaseModel):
    """Summary returned in list endpoint."""
    id: str
    name: str
    description: str
    placeholders: list[str]
    created_at: str
    updated_at: str
    metadata: dict = Field(default_factory=dict)


class PromptTemplate(BaseModel):
    """Full prompt template detail."""
    id: str
    name: str
    prompt_text: str
    description: str
    placeholders: list[str]
    created_at: str
    updated_at: str
    metadata: dict = Field(default_factory=dict)


def _extract_placeholders(prompt_text: str) -> list[str]:
    """Extract known placeholders from prompt text."""
    found = _PLACEHOLDER_RE.findall(prompt_text)
    return sorted(set(p for p in found if p in KNOWN_PLACEHOLDERS))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=list[PromptTemplateSummary])
async def list_prompt_templates():
    """List all saved prompt templates (summary view)."""
    files = list_dir(TEMPLATES_DIR)
    summaries = []
    for f in files:
        data = read_json(f)
        if data is None:
            continue
        summaries.append(PromptTemplateSummary(
            id=data.get("id", f.stem),
            name=data.get("name", "Untitled"),
            description=data.get("description", ""),
            placeholders=data.get("placeholders", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {}),
        ))
    return summaries


@router.post("", response_model=PromptTemplate, status_code=201)
async def create_prompt_template(request: CreatePromptTemplateRequest):
    """Save a new prompt template."""
    template_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()

    template_data = {
        "id": template_id,
        "name": request.name,
        "prompt_text": request.prompt_text,
        "description": request.description,
        "placeholders": _extract_placeholders(request.prompt_text),
        "created_at": now,
        "updated_at": now,
        "metadata": request.metadata,
    }

    path = TEMPLATES_DIR / f"{template_id}.json"
    write_json(path, template_data)
    return PromptTemplate(**template_data)


@router.get("/{template_id}", response_model=PromptTemplate)
async def get_prompt_template(template_id: str):
    """Get full prompt template detail."""
    path = TEMPLATES_DIR / f"{template_id}.json"
    data = read_json(path)
    if data is None:
        raise HTTPException(status_code=404, detail="Prompt template not found")
    return PromptTemplate(**data)


@router.put("/{template_id}", response_model=PromptTemplate)
async def update_prompt_template(template_id: str, request: UpdatePromptTemplateRequest):
    """Update an existing prompt template."""
    path = TEMPLATES_DIR / f"{template_id}.json"
    data = read_json(path)
    if data is None:
        raise HTTPException(status_code=404, detail="Prompt template not found")

    if request.name is not None:
        data["name"] = request.name
    if request.prompt_text is not None:
        data["prompt_text"] = request.prompt_text
        data["placeholders"] = _extract_placeholders(request.prompt_text)
    if request.description is not None:
        data["description"] = request.description
    if request.metadata is not None:
        data["metadata"] = request.metadata

    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, data)
    return PromptTemplate(**data)


@router.delete("/{template_id}")
async def delete_prompt_template(template_id: str):
    """Delete a prompt template."""
    path = TEMPLATES_DIR / f"{template_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prompt template not found")
    path.unlink()
    return {"status": "deleted", "id": template_id}


@router.post("/seed-defaults")
async def seed_defaults():
    """Replace library with built-in generator and scorer prompt templates.

    Removes all existing templates (user and builtin) and seeds fresh copies
    from the autochecklist package's prompt files.
    """
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    # Remove all existing templates
    for f in list_dir(TEMPLATES_DIR):
        f.unlink()

    created = []
    now = datetime.now(timezone.utc).isoformat()

    def _seed(name: str, text: str, description: str, metadata: dict):
        template_id = uuid.uuid4().hex[:12]
        template_data = {
            "id": template_id,
            "name": name,
            "prompt_text": text,
            "description": description,
            "placeholders": _extract_placeholders(text),
            "created_at": now,
            "updated_at": now,
            "metadata": {"source": "builtin", **metadata},
        }
        write_json(TEMPLATES_DIR / f"{template_id}.json", template_data)
        created.append(name)

    # 1. Seed instance-level preset prompts (from PIPELINE_PRESETS)
    for preset_name, preset in PIPELINE_PRESETS.items():
        try:
            text = load_template(preset["template_dir"], preset["template_name"])
        except FileNotFoundError:
            continue
        raw_class = preset.get("generator_class", "")
        gen_class = "contrastive" if "Contrastive" in raw_class else "direct"
        _seed(
            name=preset_name,
            text=text,
            description=preset.get("description", ""),
            metadata={
                "preset": preset_name,
                "type": "generator",
                "generator_class": gen_class,
                "format_name": preset.get("format_name", "checklist"),
                "default_scorer": preset.get("default_scorer", "batch"),
            },
        )

    # 2. Seed scorer prompts
    scorer_templates = {
        "batch": {"mode": "batch", "primary_metric": "pass", "capture_reasoning": False},
        "item": {"mode": "item", "primary_metric": "pass", "capture_reasoning": False},
        "rlcf": {"mode": "item", "primary_metric": "weighted", "capture_reasoning": False},
        "rocketeval": {"mode": "item", "primary_metric": "normalized", "capture_reasoning": False},
    }
    for scorer_name, scorer_config in scorer_templates.items():
        try:
            text = load_template("scoring", scorer_name)
        except FileNotFoundError:
            continue
        _seed(
            name=scorer_name,
            text=text,
            description=f"Built-in {scorer_name} scoring prompt",
            metadata={
                "preset": f"scorer:{scorer_name}",
                "type": "scorer",
                "scorer_config": scorer_config,
            },
        )

    return {"seeded": created, "total": len(created)}
