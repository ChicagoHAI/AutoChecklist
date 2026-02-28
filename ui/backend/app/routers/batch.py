"""Batch evaluation API endpoints."""

import csv
import io
import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.batch_runner import (
    cancel_batch,
    create_batch,
    delete_batch,
    get_batch_results,
    get_batch_status,
    list_batches,
    start_batch,
)
from app.services.settings_service import get_provider_kwargs

router = APIRouter()


class BatchStartRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    generator_model: Optional[str] = None
    scorer_model: Optional[str] = None
    candidate_model: Optional[str] = None


class BatchUploadPathRequest(BaseModel):
    file_path: str
    method: str = "tick"
    scorer: Optional[str] = None
    scorer_config: Optional[dict] = None
    pipeline_mode: str = "full"
    checklist_id: Optional[str] = None
    custom_prompt: Optional[str] = None
    pipeline_id: Optional[str] = None


def _parse_file(filename: str, text: str) -> list[dict]:
    """Parse file content based on extension. Returns list of row dicts."""
    data = []
    if filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        data = list(reader)
    elif filename.endswith(".jsonl"):
        for line in text.strip().split("\n"):
            if line.strip():
                data.append(json.loads(line))
    elif filename.endswith(".json"):
        parsed = json.loads(text)
        if isinstance(parsed, list):
            data = parsed
        else:
            raise HTTPException(400, "JSON file must contain an array")
    else:
        raise HTTPException(400, "Unsupported file type. Use .csv, .json, or .jsonl")
    return data


def _validate_data(data: list[dict]) -> None:
    """Validate parsed data has required columns and is non-empty."""
    if not data:
        raise HTTPException(400, "File contains no data")
    first = data[0]
    if "input" not in first or "target" not in first:
        raise HTTPException(400, "Data must have 'input' and 'target' columns")


@router.post("/upload")
async def upload_batch(
    file: UploadFile = File(...),
    method: str = Form("tick"),
    scorer: Optional[str] = Form(None),
    scorer_config: Optional[str] = Form(None),
    pipeline_mode: str = Form("full"),
    checklist_id: Optional[str] = Form(None),
    custom_prompt: Optional[str] = Form(None),
    pipeline_id: Optional[str] = Form(None),
):
    """Upload and validate a batch file. Returns batch_id for starting."""
    import json as _json

    content = await file.read()
    filename = file.filename or "upload"
    text = content.decode("utf-8")

    data = _parse_file(filename, text)
    _validate_data(data)

    # Parse scorer_config from JSON string if provided
    parsed_scorer_config = None
    if scorer_config:
        try:
            parsed_scorer_config = _json.loads(scorer_config)
        except (ValueError, TypeError):
            pass

    batch_id = create_batch(
        method=method,
        scorer=scorer,
        scorer_config=parsed_scorer_config,
        provider_kwargs={},
        filename=filename,
        data=data,
        pipeline_mode=pipeline_mode,
        checklist_id=checklist_id,
        custom_prompt=custom_prompt,
        pipeline_id=pipeline_id,
    )
    return {"batch_id": batch_id, "total_items": len(data), "filename": filename}


@router.post("/upload-path")
async def upload_batch_from_path(request: BatchUploadPathRequest):
    """Read a local file by path. No size limit — the file stays on disk."""
    from pathlib import Path

    path = Path(request.file_path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(400, f"File not found: {path}")
    if not path.is_file():
        raise HTTPException(400, f"Not a file: {path}")

    text = path.read_text(encoding="utf-8")
    data = _parse_file(path.name, text)
    _validate_data(data)

    batch_id = create_batch(
        method=request.method,
        scorer=request.scorer,
        scorer_config=request.scorer_config,
        provider_kwargs={},
        filename=path.name,
        data=data,
        pipeline_mode=request.pipeline_mode,
        checklist_id=request.checklist_id,
        custom_prompt=request.custom_prompt,
        pipeline_id=request.pipeline_id,
    )
    return {"batch_id": batch_id, "total_items": len(data), "filename": path.name}


@router.post("/{batch_id}/start")
async def start_batch_job(batch_id: str, request: BatchStartRequest = None):
    """Start a batch job."""
    if request is None:
        request = BatchStartRequest()

    provider_kwargs = get_provider_kwargs(
        provider_override=request.provider,
        model_override=request.model,
    )

    # Pass separate model overrides if provided
    if request.generator_model:
        provider_kwargs["generator_model"] = request.generator_model
    if request.scorer_model:
        provider_kwargs["scorer_model"] = request.scorer_model
    if request.candidate_model:
        provider_kwargs["candidate_model"] = request.candidate_model

    # Ensure reasoning_effort is set if any model is a reasoning model
    if "reasoning_effort" not in provider_kwargs:
        from app.services.settings_service import _is_reasoning_model
        models_to_check = [
            request.generator_model, request.scorer_model,
            request.candidate_model, request.model,
        ]
        if any(m and _is_reasoning_model(m) for m in models_to_check):
            provider_kwargs["reasoning_effort"] = "low"

    success = start_batch(batch_id, provider_kwargs)
    if not success:
        raise HTTPException(400, "Batch not found or already running")

    return {"status": "started", "batch_id": batch_id}


@router.get("")
async def list_all_batches():
    """List all batch jobs."""
    return list_batches()


@router.get("/{batch_id}")
async def get_batch(batch_id: str):
    """Get batch status and progress."""
    status = get_batch_status(batch_id)
    if not status:
        raise HTTPException(404, "Batch not found")
    return status


@router.get("/{batch_id}/results")
async def get_results(batch_id: str, offset: int = 0, limit: int = 100):
    """Get per-item results."""
    results = get_batch_results(batch_id, offset, limit)
    return {"results": results, "offset": offset, "limit": limit}


@router.post("/{batch_id}/cancel")
async def cancel_batch_job(batch_id: str):
    """Cancel a running batch job."""
    success = cancel_batch(batch_id)
    if not success:
        raise HTTPException(400, "Batch not running or not found")
    return {"status": "cancelled"}


@router.delete("/{batch_id}")
async def delete_batch_job(batch_id: str):
    """Delete a batch job and all its data."""
    success = delete_batch(batch_id)
    if not success:
        raise HTTPException(404, "Batch not found")
    return {"status": "deleted", "batch_id": batch_id}


@router.get("/{batch_id}/export")
async def export_results(batch_id: str, format: str = "json"):
    """Export batch results as JSON or CSV."""
    from fastapi.responses import Response

    results = get_batch_results(batch_id, offset=0, limit=10000)
    if not results:
        raise HTTPException(404, "No results found")

    if format == "csv":
        output = io.StringIO()
        if results:
            # Determine columns based on result shape
            has_scores = results[0].get("pass_rate") is not None
            has_checklist = "checklist_items" in results[0]
            fieldnames = ["index", "input", "target"]
            if has_scores:
                fieldnames += ["pass_rate", "total_score"]
            if has_checklist:
                fieldnames += ["checklist_items"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, "") for k in fieldnames}
                if has_checklist and "checklist_items" in r:
                    row["checklist_items"] = json.dumps(r["checklist_items"])
                writer.writerow(row)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.csv"},
        )
    else:
        return Response(
            content=json.dumps(results, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.json"},
        )
