"""CRUD endpoints for saved checklists stored as JSON files."""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.storage import DATA_DIR, list_dir, read_json, write_json, slugify

router = APIRouter()

CHECKLISTS_DIR = DATA_DIR / "checklists"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChecklistItemSchema(BaseModel):
    """A single checklist item with question and weight."""
    question: str
    weight: float = 1.0


class CreateChecklistRequest(BaseModel):
    """Request body for creating a new checklist."""
    name: str = Field(..., min_length=1)
    method: str = "custom"
    level: str = "instance"
    model: str = ""
    items: list[ChecklistItemSchema] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class UpdateChecklistRequest(BaseModel):
    """Request body for updating an existing checklist."""
    name: Optional[str] = None
    items: Optional[list[ChecklistItemSchema]] = None
    metadata: Optional[dict] = None


class ChecklistSummary(BaseModel):
    """Summary returned in list endpoint."""
    id: str
    name: str
    method: str
    level: str
    item_count: int
    created_at: str


class SavedChecklist(BaseModel):
    """Full checklist detail."""
    id: str
    name: str
    method: str
    level: str
    model: str
    items: list[ChecklistItemSchema]
    created_at: str
    updated_at: str
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_slug(name: str) -> str:
    """Generate a unique slug-based ID, appending -2, -3, etc. on collision."""
    base = slugify(name)
    candidate = base
    counter = 2
    while (CHECKLISTS_DIR / f"{candidate}.json").exists():
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=list[ChecklistSummary])
async def list_checklists():
    """List all saved checklists (summary view)."""
    files = list_dir(CHECKLISTS_DIR)
    summaries = []
    for f in files:
        data = read_json(f)
        if data is None:
            continue
        summaries.append(ChecklistSummary(
            id=data.get("id", f.stem),
            name=data.get("name", "Untitled"),
            method=data.get("method", ""),
            level=data.get("level", ""),
            item_count=len(data.get("items", [])),
            created_at=data.get("created_at", ""),
        ))
    return summaries


@router.post("", response_model=SavedChecklist, status_code=201)
async def create_checklist(request: CreateChecklistRequest):
    """Save a new checklist."""
    checklist_id = _unique_slug(request.name)
    now = datetime.now(timezone.utc).isoformat()

    checklist_data = {
        "id": checklist_id,
        "name": request.name,
        "method": request.method,
        "level": request.level,
        "model": request.model,
        "items": [item.model_dump() for item in request.items],
        "created_at": now,
        "updated_at": now,
        "metadata": request.metadata,
    }

    path = CHECKLISTS_DIR / f"{checklist_id}.json"
    write_json(path, checklist_data)
    return SavedChecklist(**checklist_data)


@router.get("/{checklist_id}", response_model=SavedChecklist)
async def get_checklist(checklist_id: str):
    """Get full checklist detail."""
    path = CHECKLISTS_DIR / f"{checklist_id}.json"
    data = read_json(path)
    if data is None:
        raise HTTPException(status_code=404, detail="Checklist not found")
    return SavedChecklist(**data)


@router.put("/{checklist_id}", response_model=SavedChecklist)
async def update_checklist(checklist_id: str, request: UpdateChecklistRequest):
    """Update an existing checklist (name, items, metadata)."""
    path = CHECKLISTS_DIR / f"{checklist_id}.json"
    data = read_json(path)
    if data is None:
        raise HTTPException(status_code=404, detail="Checklist not found")

    if request.name is not None:
        data["name"] = request.name
    if request.items is not None:
        data["items"] = [item.model_dump() for item in request.items]
    if request.metadata is not None:
        data["metadata"] = request.metadata

    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, data)
    return SavedChecklist(**data)


@router.delete("/{checklist_id}")
async def delete_checklist(checklist_id: str):
    """Delete a checklist."""
    path = CHECKLISTS_DIR / f"{checklist_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Checklist not found")
    path.unlink()
    return {"status": "deleted", "id": checklist_id}


@router.post("/import", response_model=SavedChecklist, status_code=201)
async def import_checklist(
    file: UploadFile = File(...),
    name: str = Form(...),
):
    """Import a checklist from a JSON or CSV file.

    Supported JSON formats:
      - Array of objects: [{"question": "...", "weight": 1.0}, ...]
      - Object with items key: {"items": [{"question": "...", "weight": 1.0}, ...]}

    Supported CSV format:
      - Columns: question (required), weight (optional, defaults to 1.0)
    """
    content = await file.read()
    text = content.decode("utf-8").strip()
    if not text:
        raise HTTPException(status_code=400, detail="File is empty")

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    items: list[dict] = []

    if ext == "csv":
        reader = csv.DictReader(io.StringIO(text))
        if "question" not in (reader.fieldnames or []):
            raise HTTPException(status_code=400, detail="CSV must have a 'question' column")
        for row in reader:
            q = row.get("question", "").strip()
            if not q:
                continue
            w = 1.0
            if "weight" in row and row["weight"]:
                try:
                    w = float(row["weight"])
                except ValueError:
                    pass
            items.append({"question": q, "weight": w})
    else:
        # JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        if isinstance(parsed, list):
            raw_items = parsed
        elif isinstance(parsed, dict) and "items" in parsed:
            raw_items = parsed["items"]
        else:
            raise HTTPException(
                status_code=400,
                detail="JSON must be an array of items or an object with an 'items' key",
            )

        for entry in raw_items:
            if not isinstance(entry, dict) or "question" not in entry:
                continue
            q = str(entry["question"]).strip()
            if not q:
                continue
            w = float(entry.get("weight", 1.0))
            items.append({"question": q, "weight": w})

    if not items:
        raise HTTPException(status_code=400, detail="No valid checklist items found in file")

    checklist_id = _unique_slug(name)
    now = datetime.now(timezone.utc).isoformat()

    checklist_data = {
        "id": checklist_id,
        "name": name,
        "method": "custom",
        "level": "instance",
        "model": "",
        "items": items,
        "created_at": now,
        "updated_at": now,
        "metadata": {"imported_from": file.filename},
    }

    path = CHECKLISTS_DIR / f"{checklist_id}.json"
    write_json(path, checklist_data)
    return SavedChecklist(**checklist_data)
