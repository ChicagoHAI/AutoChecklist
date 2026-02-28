"""Evaluate endpoints: streaming, non-streaming, generate-only, score-only."""

import asyncio
import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from autochecklist.models import (
    Checklist as LibChecklist,
    ChecklistItem as LibChecklistItem,
    ChecklistItemAnswer,
)
from autochecklist.pipeline import DEFAULT_SCORERS

from app.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    GenerateRequest,
    GenerateResponse,
    ScoreRequest,
    ScoreResponse,
    Checklist,
    ChecklistItem,
    MethodResult,
    ScoreResult,
)
from app.services.checklist import (
    generate_only,
    score_only,
    METHODS_NO_REF,
    METHODS_WITH_REF,
    _SCORER_NAME_TO_CONFIG,
)
from app.services.settings_service import get_provider_kwargs
from app.services.storage import DATA_DIR, delete_data_dir, write_json, read_json

logger = logging.getLogger("ui.evaluate")

router = APIRouter()

# Track background eval tasks by eval_id for cancellation support
_background_tasks: dict[str, asyncio.Task] = {}


def _extract_error_message(exc: Exception) -> str:
    """Extract a human-readable error message, unwrapping RetryError and HTTPStatusError."""
    # Unwrap tenacity RetryError to get the real exception
    if hasattr(exc, "last_attempt"):
        inner = exc.last_attempt.exception()
        if inner is not None:
            exc = inner
    # httpx.HTTPStatusError — str() already includes our detailed message from _raise_with_detail
    return str(exc)


def _get_merged_provider_kwargs(
    provider: str | None = None, model: str | None = None
) -> dict:
    """Merge per-request provider/model with settings defaults."""
    return get_provider_kwargs(provider_override=provider, model_override=model)


def _build_gen_score_pks(request) -> tuple[dict, dict]:
    """Build separate provider kwargs for generator and scorer from a request.

    Falls back to request.provider/model when generator_*/scorer_* not set.
    Each dict contains: provider, api_key/base_url, model.
    """
    gen_pk = _get_merged_provider_kwargs(
        provider=request.generator_provider or request.provider,
        model=request.generator_model or request.model,
    )
    score_pk = _get_merged_provider_kwargs(
        provider=request.scorer_provider or request.provider,
        model=request.scorer_model or request.model,
    )
    return gen_pk, score_pk


def _convert_lib_checklist(checklist: LibChecklist, method: str) -> Checklist:
    """Convert library Checklist to API schema Checklist."""
    items = [
        ChecklistItem(
            question=item.question,
            weight=getattr(item, "weight", 1.0),
        )
        for item in checklist.items
    ]
    return Checklist(
        items=items,
        source_method=method,
        metadata=getattr(checklist, "metadata", {}),
    )


def _convert_lib_score(score_result, checklist_items: list[ChecklistItem]) -> ScoreResult:
    """Convert library Score to API schema ScoreResult."""
    item_results = []
    if hasattr(score_result, "item_scores"):
        for i, item_score in enumerate(score_result.item_scores):
            passed = None
            if hasattr(item_score, "answer"):
                if item_score.answer == ChecklistItemAnswer.YES:
                    passed = True
                elif item_score.answer == ChecklistItemAnswer.NO:
                    passed = False
            item_results.append({
                "question": checklist_items[i].question if i < len(checklist_items) else "",
                "passed": passed,
                "confidence": getattr(item_score, "confidence", None),
                "reasoning": getattr(item_score, "reasoning", None),
            })

    # score/max_score = pass count fraction (always meaningful)
    # percentage = primary_score as percentage (respects scorer config)
    primary = getattr(score_result, "primary_score", None)
    if primary is None:
        primary = getattr(score_result, "total_score", 0.0)
    applicable_items = len([r for r in item_results if r["passed"] is not None])
    passed_items = len([r for r in item_results if r["passed"] is True])
    max_score = applicable_items if applicable_items > 0 else 1

    return ScoreResult(
        score=float(passed_items),
        max_score=float(max_score),
        percentage=float(primary * 100),
        primary_metric=getattr(score_result, "primary_metric", "pass"),
        item_results=item_results,
    )


def _resolve_scorer(request, method: str | None = None) -> str | dict:
    """Resolve scorer from request, preferring scorer_config over scorer string.

    Returns a config dict (for ChecklistScorer) or a string name.
    """
    if hasattr(request, "scorer_config") and request.scorer_config:
        return request.scorer_config
    if hasattr(request, "scorer") and request.scorer:
        return request.scorer
    if method:
        return DEFAULT_SCORERS.get(method, "batch")
    return "batch"


def _select_methods(reference: str | None, requested: list[str] | None) -> list[str]:
    """Select methods based on reference availability and request."""
    if requested is not None:
        return requested
    methods = list(METHODS_NO_REF)
    if reference:
        methods.extend(METHODS_WITH_REF)
    return methods


def _save_evaluation(results: dict, request_data: dict) -> None:
    """Save evaluation results to data/evaluations/."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    path = DATA_DIR / "evaluations" / f"{ts}_{uid}.json"
    payload = {
        "timestamp": ts,
        "request": request_data,
        "results": results,
    }
    try:
        write_json(path, payload)
    except Exception:
        logger.exception("Failed to save evaluation to %s", path)


# ---------------------------------------------------------------------------
# Async eval - background job with polling
# ---------------------------------------------------------------------------


async def _run_pipeline_job(
    pipeline_id: str,
    request: EvaluateRequest,
    gen_pk: dict,
    score_pk: dict,
) -> tuple[str, str, object | None, object | None, str | None]:
    """Run a single pipeline config. Returns (key, pipeline_name, checklist, score, error)."""
    from app.services.checklist import _build_pipeline_from_config, run_in_executor, score_only as _score_only
    from typing import Any

    config_path = DATA_DIR / "pipelines" / f"{pipeline_id}.json"
    config_data = read_json(config_path)
    if config_data is None:
        return pipeline_id, pipeline_id, None, None, f"Pipeline {pipeline_id} not found"

    pipeline_name = config_data.get("name", pipeline_id)
    key = f"pipeline:{pipeline_id}"

    try:
        pipe = _build_pipeline_from_config(config_data, candidate_model=getattr(request, "candidate_model", None), **{**gen_pk})
        gen_kwargs: dict[str, Any] = {"input": request.input}
        if request.target:
            gen_kwargs["target"] = request.target
        if request.reference:
            gen_kwargs["reference"] = request.reference

        checklist = await run_in_executor(pipe.generate, **gen_kwargs)

        scorer_name = config_data.get("scorer_class", "batch")
        custom_scorer_prompt = config_data.get("scorer_prompt") or None
        score_result = await _score_only(
            checklist=checklist,
            target=request.target,
            input=request.input,
            scorer_name=scorer_name,
            custom_scorer_prompt=custom_scorer_prompt,
            **score_pk,
        )
        return key, pipeline_name, checklist, score_result, None
    except Exception as e:
        return key, pipeline_name, None, None, _extract_error_message(e)


async def _run_eval_job(
    eval_id: str, request: EvaluateRequest, gen_pk: dict, score_pk: dict, methods: list[str]
) -> None:
    """Background task: generate + score each method and pipelines, updating summary.json.

    All methods and pipelines run concurrently. An asyncio.Lock protects
    shared writes to all_results / completed / summary.json.
    """
    eval_dir = DATA_DIR / "evaluations" / eval_id
    all_results: dict = {}
    completed: list[str] = []
    lock = asyncio.Lock()

    async def _write_summary() -> None:
        """Write current state to summary.json. Caller must hold lock."""
        summary = read_json(eval_dir / "summary.json") or {}
        summary["results"] = all_results
        summary["completed_methods"] = completed
        write_json(eval_dir / "summary.json", summary)

    def _is_cancelled() -> bool:
        check = read_json(eval_dir / "summary.json") or {}
        return check.get("status") == "cancelled"

    async def _run_single_method(method: str) -> None:
        """Generate + score a single method, writing intermediate results."""
        if _is_cancelled():
            return
        try:
            # Generate checklist
            checklist = await generate_only(
                method=method,
                input=request.input,
                target=request.target,
                reference=request.reference,
                custom_prompt=request.custom_prompt,
                candidate_model=getattr(request, "candidate_model", None),
                **gen_pk,
            )
            converted_cl = _convert_lib_checklist(checklist, method)
            logger.info("[%s][%s] Generated %d items", eval_id[:8], method, len(converted_cl.items))

            # Write checklist immediately (before scoring) so frontend can show it
            async with lock:
                all_results[method] = {
                    "checklist": converted_cl.model_dump(),
                    "score": None,
                }
                await _write_summary()

            if _is_cancelled():
                return

            # Score it
            score_result = await score_only(
                checklist=checklist,
                target=request.target,
                input=request.input,
                scorer_name=_resolve_scorer(request, method),
                custom_scorer_prompt=getattr(request, "custom_scorer_prompt", None),
                **score_pk,
            )
            converted_score = _convert_lib_score(score_result, converted_cl.items)

            # Update items with pass/fail and reasoning
            for i, item_res in enumerate(converted_score.item_results):
                if i < len(converted_cl.items):
                    converted_cl.items[i].passed = item_res.get("passed")
                    converted_cl.items[i].confidence = item_res.get("confidence")
                    converted_cl.items[i].reasoning = item_res.get("reasoning")

            async with lock:
                all_results[method] = {
                    "checklist": converted_cl.model_dump(),
                    "score": converted_score.model_dump(),
                }
                completed.append(method)
                await _write_summary()
            logger.info("[%s][%s] Score: %.0f%%", eval_id[:8], method, converted_score.percentage)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            msg = _extract_error_message(e)
            logger.error("[%s][%s] Failed: %s", eval_id[:8], method, msg)
            async with lock:
                all_results[method] = {"error": msg}
                completed.append(method)
                await _write_summary()

    async def _run_single_pipeline(pid: str) -> None:
        """Run a single custom pipeline, writing results."""
        if _is_cancelled():
            return
        key, pname, checklist, score_result, error = await _run_pipeline_job(
            pid, request, gen_pk, score_pk,
        )
        if error:
            logger.error("[%s][%s] Pipeline failed: %s", eval_id[:8], pname, error)
            async with lock:
                all_results[key] = {"error": error, "pipeline_name": pname}
                completed.append(key)
                await _write_summary()
        else:
            converted_cl = _convert_lib_checklist(checklist, key)
            converted_score = _convert_lib_score(score_result, converted_cl.items)
            for i, item_res in enumerate(converted_score.item_results):
                if i < len(converted_cl.items):
                    converted_cl.items[i].passed = item_res.get("passed")
                    converted_cl.items[i].confidence = item_res.get("confidence")
                    converted_cl.items[i].reasoning = item_res.get("reasoning")
            async with lock:
                all_results[key] = {
                    "checklist": converted_cl.model_dump(),
                    "score": converted_score.model_dump(),
                    "pipeline_name": pname,
                }
                completed.append(key)
                await _write_summary()
            logger.info("[%s][%s] Pipeline score: %.0f%%", eval_id[:8], pname, converted_score.percentage)

    try:
        # Launch all methods and pipelines concurrently
        tasks: list[asyncio.Task] = []
        for method in methods:
            tasks.append(asyncio.create_task(_run_single_method(method)))
        pipeline_ids = getattr(request, "pipeline_ids", None) or []
        for pid in pipeline_ids:
            tasks.append(asyncio.create_task(_run_single_pipeline(pid)))

        for coro in asyncio.as_completed(tasks):
            await coro

        # Mark complete (only if not cancelled in the meantime)
        summary = read_json(eval_dir / "summary.json") or {}
        if summary.get("status") != "cancelled":
            summary["status"] = "completed"
            summary["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
            write_json(eval_dir / "summary.json", summary)
    except asyncio.CancelledError:
        # Cancel all sub-tasks
        for t in tasks:
            if not t.done():
                t.cancel()
        # Wait for cancellation to propagate
        await asyncio.gather(*tasks, return_exceptions=True)
        summary = read_json(eval_dir / "summary.json") or {}
        if summary.get("status") == "running":
            summary["status"] = "cancelled"
            write_json(eval_dir / "summary.json", summary)
    except Exception as e:
        msg = _extract_error_message(e)
        logger.error("[%s] Job failed: %s\n%s", eval_id[:8], msg, traceback.format_exc())
        summary = read_json(eval_dir / "summary.json") or {}
        summary["status"] = "failed"
        summary["error"] = msg
        write_json(eval_dir / "summary.json", summary)


@router.post("/async")
async def evaluate_async(request: EvaluateRequest):
    """Start an evaluation as a background job. Returns eval_id for polling."""
    gen_pk, score_pk = _build_gen_score_pks(request)

    methods = _select_methods(request.reference, request.methods)
    eval_id = f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    eval_dir = DATA_DIR / "evaluations" / eval_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    pipeline_ids = request.pipeline_ids or []
    # Include pipeline keys in the methods list so frontend can track them
    all_keys = methods + [f"pipeline:{pid}" for pid in pipeline_ids]

    summary = {
        "eval_id": eval_id,
        "status": "running",
        "methods": all_keys,
        "completed_methods": [],
        "results": {},
        "request": {
            "input": request.input,
            "target": request.target,
            "reference": request.reference,
            "methods": methods,
            "pipeline_ids": pipeline_ids,
            "model": gen_pk.get("model"),
            "provider": gen_pk.get("provider"),
            "generator_model": gen_pk.get("model"),
            "generator_provider": gen_pk.get("provider"),
            "scorer_model": score_pk.get("model"),
            "scorer_provider": score_pk.get("provider"),
        },
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    write_json(eval_dir / "summary.json", summary)

    task = asyncio.create_task(_run_eval_job(eval_id, request, gen_pk, score_pk, methods))
    _background_tasks[eval_id] = task
    task.add_done_callback(lambda t: _background_tasks.pop(eval_id, None))

    logger.info(
        "Async eval started: id=%s, methods=%s, gen_model=%s, score_model=%s",
        eval_id, methods, gen_pk.get("model"), score_pk.get("model"),
    )
    return {"eval_id": eval_id}


@router.get("/recent")
async def list_recent_evals(limit: int = 10):
    """List recent async evaluations."""
    eval_dir = DATA_DIR / "evaluations"
    if not eval_dir.exists():
        return []
    dirs = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )[:limit]
    results = []
    for d in dirs:
        summary = read_json(d / "summary.json")
        if summary:
            results.append(summary)
    return results


@router.get("/{eval_id}")
async def get_eval_status(eval_id: str):
    """Get the status and results of an async evaluation."""
    summary = read_json(DATA_DIR / "evaluations" / eval_id / "summary.json")
    if not summary:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return summary


@router.post("/{eval_id}/cancel")
async def cancel_evaluation(eval_id: str):
    """Cancel a running evaluation."""
    eval_dir = DATA_DIR / "evaluations" / eval_id
    summary = read_json(eval_dir / "summary.json")
    if not summary:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    if summary.get("status") != "running":
        return {"status": summary.get("status"), "id": eval_id}
    # Write cancelled immediately so frontend sees it on next poll
    summary["status"] = "cancelled"
    write_json(eval_dir / "summary.json", summary)
    # Best-effort cancel the asyncio task
    task = _background_tasks.pop(eval_id, None)
    if task and not task.done():
        task.cancel()
    return {"status": "cancelled", "id": eval_id}


@router.delete("/{eval_id}")
async def delete_evaluation(eval_id: str):
    """Delete an evaluation and its data."""
    # Cancel any running task before deleting
    task = _background_tasks.pop(eval_id, None)
    if task and not task.done():
        task.cancel()
    if not delete_data_dir("evaluations", eval_id):
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return {"status": "deleted", "id": eval_id}


# ---------------------------------------------------------------------------
# POST /api/evaluate/stream  - SSE streaming
# ---------------------------------------------------------------------------

@router.post("/stream")
async def evaluate_stream(request: EvaluateRequest):
    """Stream evaluation results as Server-Sent Events.

    Sends events:
    - event: checklist  data: {method, checklist}
    - event: score      data: {method, score, items}
    - event: done       data: {}
    - event: error      data: {method, phase, error}
    """
    gen_pk, score_pk = _build_gen_score_pks(request)

    methods = _select_methods(request.reference, request.methods)
    logger.info(
        "Stream eval: methods=%s, gen_model=%s, score_model=%s, has_ref=%s, instruction=%.60s...",
        methods, gen_pk.get("model"), score_pk.get("model"), bool(request.reference), request.input,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        all_results: dict = {}

        # Phase 1: generate checklists
        async def gen_one(method: str):
            try:
                logger.info("[%s] Generating checklist...", method)
                checklist = await generate_only(
                    method=method,
                    input=request.input,
                    target=request.target,
                    reference=request.reference,
                    custom_prompt=request.custom_prompt,
                    candidate_model=getattr(request, "candidate_model", None),
                    **gen_pk,
                )
                logger.info("[%s] Generated %d checklist items", method, len(checklist.items))
                return method, checklist, None
            except Exception as e:
                logger.error("[%s] Generation failed: %s\n%s", method, e, traceback.format_exc())
                return method, None, str(e)

        gen_tasks = [asyncio.create_task(gen_one(m)) for m in methods]
        checklists: dict = {}

        for coro in asyncio.as_completed(gen_tasks):
            method, checklist, error = await coro
            if error:
                yield f"event: error\ndata: {json.dumps({'method': method, 'phase': 'checklist', 'error': error})}\n\n"
            else:
                checklists[method] = checklist
                converted = _convert_lib_checklist(checklist, method)
                yield f"event: checklist\ndata: {json.dumps({'method': method, 'checklist': converted.model_dump()})}\n\n"

        # Phase 2: score checklists
        async def score_one(method: str):
            if method not in checklists:
                logger.warning("[%s] Skipping scoring — no checklist generated", method)
                return method, None, None, None  # Generation error already sent
            try:
                scorer_name = _resolve_scorer(request, method)
                logger.info("[%s] Scoring %d items...", method, len(checklists[method].items))
                # Use score_only with the already-generated checklist
                score_result = await score_only(
                    checklist=checklists[method],
                    target=request.target,
                    input=request.input,
                    scorer_name=scorer_name,
                    custom_scorer_prompt=getattr(request, "custom_scorer_prompt", None),
                    **score_pk,
                )
                converted_cl = _convert_lib_checklist(checklists[method], method)
                converted_score = _convert_lib_score(score_result, converted_cl.items)

                # Update checklist items with pass/fail and reasoning
                for i, item_res in enumerate(converted_score.item_results):
                    if i < len(converted_cl.items):
                        converted_cl.items[i].passed = item_res.get("passed")
                        converted_cl.items[i].confidence = item_res.get("confidence")
                        converted_cl.items[i].reasoning = item_res.get("reasoning")

                all_results[method] = {
                    "checklist": converted_cl.model_dump(),
                    "score": converted_score.model_dump(),
                }
                logger.info(
                    "[%s] Score: %s/%s (%.0f%%)",
                    method, converted_score.score, converted_score.max_score, converted_score.percentage,
                )
                return method, converted_score, converted_score.item_results, None
            except Exception as e:
                logger.error("[%s] Scoring failed: %s\n%s", method, e, traceback.format_exc())
                return method, None, None, str(e)

        score_tasks = [asyncio.create_task(score_one(m)) for m in methods]

        for coro in asyncio.as_completed(score_tasks):
            method, score_result, scored_items, error = await coro
            if error:
                yield f"event: error\ndata: {json.dumps({'method': method, 'phase': 'score', 'error': error})}\n\n"
            elif score_result:
                yield f"event: score\ndata: {json.dumps({'method': method, 'score': score_result.model_dump(), 'items': scored_items})}\n\n"

        # Phase 3: custom pipelines (sequential, each emits checklist + score)
        pipeline_ids = getattr(request, "pipeline_ids", None) or []
        for pid in pipeline_ids:
            key, pname, checklist, score_result, error = await _run_pipeline_job(
                pid, request, gen_pk, score_pk,
            )
            if error:
                yield f"event: error\ndata: {json.dumps({'method': key, 'phase': 'checklist', 'error': error, 'pipeline_name': pname})}\n\n"
            else:
                converted_cl = _convert_lib_checklist(checklist, key)
                yield f"event: checklist\ndata: {json.dumps({'method': key, 'checklist': converted_cl.model_dump(), 'pipeline_name': pname})}\n\n"
                converted_score = _convert_lib_score(score_result, converted_cl.items)
                for i, item_res in enumerate(converted_score.item_results):
                    if i < len(converted_cl.items):
                        converted_cl.items[i].passed = item_res.get("passed")
                        converted_cl.items[i].confidence = item_res.get("confidence")
                        converted_cl.items[i].reasoning = item_res.get("reasoning")
                all_results[key] = {
                    "checklist": converted_cl.model_dump(),
                    "score": converted_score.model_dump(),
                    "pipeline_name": pname,
                }
                yield f"event: score\ndata: {json.dumps({'method': key, 'score': converted_score.model_dump(), 'items': converted_score.item_results, 'pipeline_name': pname})}\n\n"

        # Save results
        _save_evaluation(
            results=all_results,
            request_data={
                "input": request.input,
                "target": request.target,
                "reference": request.reference,
                "methods": methods,
                "pipeline_ids": pipeline_ids,
                "model": gen_pk.get("model"),
                "provider": gen_pk.get("provider"),
            },
        )

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /api/evaluate  - Non-streaming (kept for programmatic use)
# ---------------------------------------------------------------------------

@router.post("", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """Evaluate an input-target pair using multiple checklist methods."""
    gen_pk, score_pk = _build_gen_score_pks(request)

    methods = _select_methods(request.reference, request.methods)

    async def run_method(method: str):
        try:
            checklist = await generate_only(
                method=method,
                input=request.input,
                target=request.target,
                reference=request.reference,
                custom_prompt=request.custom_prompt,
                candidate_model=getattr(request, "candidate_model", None),
                **gen_pk,
            )
            converted_cl = _convert_lib_checklist(checklist, method)

            scorer_name = _resolve_scorer(request, method)
            score_result = await score_only(
                checklist=checklist,
                target=request.target,
                input=request.input,
                scorer_name=scorer_name,
                custom_scorer_prompt=getattr(request, "custom_scorer_prompt", None),
                **score_pk,
            )
            converted_score = _convert_lib_score(score_result, converted_cl.items)

            # Update pass/fail and reasoning on items
            for i, item_res in enumerate(converted_score.item_results):
                if i < len(converted_cl.items):
                    converted_cl.items[i].passed = item_res.get("passed")
                    converted_cl.items[i].confidence = item_res.get("confidence")
                    converted_cl.items[i].reasoning = item_res.get("reasoning")

            return method, MethodResult(checklist=converted_cl, score=converted_score)
        except Exception as e:
            logger.error("Error in %s: %s", method, _extract_error_message(e))
            return method, e

    tasks = [asyncio.create_task(run_method(m)) for m in methods]
    results_map: dict[str, MethodResult] = {}
    for coro in asyncio.as_completed(tasks):
        method, result = await coro
        if not isinstance(result, Exception):
            results_map[method] = result

    if not results_map:
        raise HTTPException(status_code=500, detail="All evaluation methods failed")

    return EvaluateResponse(results=results_map)


# ---------------------------------------------------------------------------
# POST /api/evaluate/generate  - Generate only
# ---------------------------------------------------------------------------

@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a checklist without scoring."""
    pk = _get_merged_provider_kwargs(provider=request.provider, model=request.model)

    try:
        checklist = await generate_only(
            method=request.method,
            input=request.input,
            target=request.target,
            reference=request.reference,
            custom_prompt=request.custom_prompt,
            candidate_model=getattr(request, "candidate_model", None),
            **pk,
        )
        converted = _convert_lib_checklist(checklist, request.method)
        return GenerateResponse(
            method=request.method,
            checklist=converted.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# POST /api/evaluate/score  - Score only
# ---------------------------------------------------------------------------

@router.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest):
    """Score a response against a provided checklist."""
    pk = _get_merged_provider_kwargs(provider=request.provider, model=request.model)

    # Reconstruct a library Checklist from the request items
    lib_items = []
    for item_data in request.checklist_items:
        lib_items.append(
            LibChecklistItem(
                question=item_data["question"],
                weight=item_data.get("weight", 1.0),
            )
        )
    lib_checklist = LibChecklist(
        items=lib_items,
        source_method="custom",
        generation_level="instance",
    )

    # Build schema ChecklistItems for the conversion helper
    schema_items = [
        ChecklistItem(question=item_data["question"], weight=item_data.get("weight", 1.0))
        for item_data in request.checklist_items
    ]

    try:
        score_result = await score_only(
            checklist=lib_checklist,
            target=request.target,
            input=request.input,
            scorer_name=_resolve_scorer(request),
            custom_scorer_prompt=request.custom_scorer_prompt,
            **pk,
        )
        converted = _convert_lib_score(score_result, schema_items)
        return ScoreResponse(score=converted.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
