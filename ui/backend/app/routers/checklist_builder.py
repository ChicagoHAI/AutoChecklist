"""Checklist builder endpoints: corpus-level generation from dimensions."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from autochecklist import DeductiveGenerator
from autochecklist.models import DeductiveInput as LibDeductiveInput

from app.schemas import (
    Checklist,
    ChecklistItem,
    DeductiveGenerateRequest,
    DeductiveGenerateResponse,
)
from app.services.settings_service import get_provider_kwargs

logger = logging.getLogger("ui.builder")

router = APIRouter()

executor = ThreadPoolExecutor(max_workers=4)


def _run_deductive_generate(request: DeductiveGenerateRequest) -> Checklist:
    """Run DeductiveGenerator.generate() synchronously (called from thread pool)."""
    pk = get_provider_kwargs(
        provider_override=request.provider, model_override=request.model
    )

    gen = DeductiveGenerator(
        model=pk.get("model"),
        provider=pk.get("provider"),
        api_key=pk.get("api_key"),
        base_url=pk.get("base_url"),
        augmentation_mode=request.augmentation_mode,
        task_type=request.task_type,
    )

    dimensions = [
        LibDeductiveInput(
            name=d.name,
            definition=d.definition,
            sub_dimensions=d.sub_dimensions if d.sub_dimensions else None,
        )
        for d in request.dimensions
    ]

    lib_checklist = gen.generate(dimensions=dimensions)

    # Include category in metadata for frontend grouping
    items_with_category = []
    for item in lib_checklist.items:
        ci = ChecklistItem(
            question=item.question,
            weight=getattr(item, "weight", 1.0),
        )
        items_with_category.append(ci)

    metadata = getattr(lib_checklist, "metadata", {}) or {}
    # Build per-item category info for frontend grouping
    item_categories = []
    for item in lib_checklist.items:
        item_categories.append(getattr(item, "category", None))
    if any(c is not None for c in item_categories):
        metadata["item_categories"] = item_categories

    return Checklist(
        items=items_with_category,
        source_method="checkeval",
        metadata=metadata,
    )


@router.post("/dimension", response_model=DeductiveGenerateResponse)
async def generate_dimension_checklist(request: DeductiveGenerateRequest):
    """Generate a checklist from dimension definitions using DeductiveGenerator."""
    try:
        loop = asyncio.get_event_loop()
        checklist = await loop.run_in_executor(
            executor, _run_deductive_generate, request
        )
        return DeductiveGenerateResponse(checklist=checklist)
    except Exception as exc:
        logger.exception("DeductiveGenerator failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
