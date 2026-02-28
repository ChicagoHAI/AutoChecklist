"""Background batch job runner for batch evaluation."""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from autochecklist import ChecklistPipeline
from autochecklist.models import Checklist as LibChecklist, ChecklistItem as LibChecklistItem
from autochecklist.registry import get_scorer

from app.services.checklist import METHOD_CONFIG, _build_pipeline_from_config
from app.services.storage import DATA_DIR, delete_data_dir, read_json, write_json

logger = logging.getLogger("ui.batch")

# In-memory tracking of running jobs
_running_jobs: dict[str, threading.Thread] = {}
_cancel_flags: dict[str, threading.Event] = {}


def create_batch(
    method: str,
    scorer: Optional[str],
    provider_kwargs: dict,
    filename: str,
    data: list[dict],
    pipeline_mode: str = "full",
    checklist_id: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    scorer_config: Optional[dict] = None,
) -> str:
    """Create a new batch job directory and save config + input."""
    batch_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    batch_dir = DATA_DIR / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "method": method,
        "scorer": scorer,
        "filename": filename,
        "total_items": len(data),
        "pipeline_mode": pipeline_mode,
        **{k: v for k, v in provider_kwargs.items() if k != "api_key"},  # Don't persist API key
    }
    if scorer_config:
        config["scorer_config"] = scorer_config
    if checklist_id:
        config["checklist_id"] = checklist_id
        # Store checklist metadata for display
        cl_data = read_json(DATA_DIR / "checklists" / f"{checklist_id}.json")
        if cl_data:
            config["checklist_name"] = cl_data.get("name", "")
            config["checklist_level"] = cl_data.get("level", "")
            config["checklist_method"] = cl_data.get("method", "")
    if custom_prompt:
        config["custom_prompt"] = custom_prompt
    if pipeline_id:
        config["pipeline_id"] = pipeline_id
        # Store pipeline name for display
        pl_data = read_json(DATA_DIR / "pipelines" / f"{pipeline_id}.json")
        if pl_data:
            config["pipeline_name"] = pl_data.get("name", "")
    write_json(batch_dir / "config.json", config)

    # Save input data as JSONL
    with open(batch_dir / "input.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Initialize summary
    write_json(batch_dir / "summary.json", {
        "status": "pending",
        "total": len(data),
        "completed": 0,
        "failed": 0,
        "mean_score": None,
        "macro_pass_rate": None,
        "micro_pass_rate": None,
        "started_at": None,
        "completed_at": None,
    })

    logger.info("Batch %s created: mode=%s, method=%s, %d items from %s", batch_id, pipeline_mode, method, len(data), filename)
    return batch_id


def start_batch(batch_id: str, provider_kwargs: dict) -> bool:
    """Start a batch job in a background thread."""
    if batch_id in _running_jobs and _running_jobs[batch_id].is_alive():
        logger.warning("Batch %s already running", batch_id)
        return False  # Already running

    batch_dir = DATA_DIR / "batches" / batch_id
    config = read_json(batch_dir / "config.json")
    if not config:
        return False

    # Persist provider/model into config for display
    config_updated = False
    for key in ("provider", "model", "generator_model", "scorer_model", "candidate_model"):
        if key in provider_kwargs and provider_kwargs[key] and key not in config:
            config[key] = provider_kwargs[key]
            config_updated = True
    if config_updated:
        write_json(batch_dir / "config.json", config)

    cancel_event = threading.Event()
    _cancel_flags[batch_id] = cancel_event

    thread = threading.Thread(
        target=_run_batch_job,
        args=(batch_id, batch_dir, config, provider_kwargs, cancel_event),
        daemon=True,
    )
    _running_jobs[batch_id] = thread
    thread.start()
    return True


def _run_batch_job(batch_id: str, batch_dir: Path, config: dict, provider_kwargs: dict, cancel_event: threading.Event):
    """Background thread that runs the batch job."""
    _update_summary(batch_dir, {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})

    # Load input data
    data = []
    with open(batch_dir / "input.jsonl") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    pipeline_mode = config.get("pipeline_mode", "full")

    try:
        if pipeline_mode == "generate_only":
            _run_generate_only(batch_dir, config, data, provider_kwargs, cancel_event)
        elif pipeline_mode == "score_only":
            _run_score_only(batch_dir, config, data, provider_kwargs, cancel_event)
        else:
            _run_full_pipeline(batch_dir, config, data, provider_kwargs, cancel_event)
    except Exception as e:
        _update_summary(batch_dir, {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    finally:
        _running_jobs.pop(batch_id, None)
        _cancel_flags.pop(batch_id, None)


def _build_pipeline(config: dict, provider_kwargs: dict, need_scorer: bool = True) -> ChecklistPipeline:
    """Build a ChecklistPipeline from config or saved pipeline."""
    # If pipeline_id is set, load pipeline config and delegate
    pipeline_id = config.get("pipeline_id")
    if pipeline_id:
        pipeline_data = read_json(DATA_DIR / "pipelines" / f"{pipeline_id}.json")
        if pipeline_data:
            pk = dict(provider_kwargs)
            candidate_model = pk.pop("candidate_model", None)
            scorer_name = config.get("scorer") if need_scorer else None
            return _build_pipeline_from_config(
                pipeline_data,
                scorer_name=scorer_name,
                candidate_model=candidate_model,
                **pk,
            )

    method = config["method"]
    scorer = config.get("scorer")
    custom_prompt = config.get("custom_prompt")

    method_cfg = METHOD_CONFIG.get(method, {"generator": method})
    gen_name = method_cfg["generator"]
    gen_kwargs = dict(method_cfg.get("generator_kwargs", {}))

    # Remap flat 'model' → generator_model + scorer_model
    pk = dict(provider_kwargs)
    candidate_model = pk.pop("candidate_model", None)
    model = pk.pop("model", None)
    if model:
        pk.setdefault("generator_model", model)
        pk.setdefault("scorer_model", model)

    # Override candidate model if specified
    if candidate_model:
        gen_kwargs["candidate_models"] = [candidate_model]

    pipe_kwargs: dict = {
        "generator_kwargs": gen_kwargs or None,
        **pk,
    }
    # Use from_preset when we need default scorer auto-attached, generator= otherwise
    if need_scorer and not scorer:
        pipe_kwargs["from_preset"] = gen_name
    else:
        pipe_kwargs["generator"] = gen_name
        if need_scorer and scorer:
            pipe_kwargs["scorer"] = scorer
    if custom_prompt:
        pipe_kwargs["custom_prompt"] = custom_prompt

    return ChecklistPipeline(**pipe_kwargs)


def _run_full_pipeline(batch_dir: Path, config: dict, data: list[dict], provider_kwargs: dict, cancel_event: threading.Event):
    """Full pipeline: generate checklists and score each item."""
    import statistics
    from autochecklist.models import ChecklistItemAnswer

    pipe = _build_pipeline(config, provider_kwargs)
    results_path = batch_dir / "results.jsonl"
    scores: list[float] = []
    score_values: list[float] = []
    total_yes = 0
    total_applicable = 0

    with open(results_path, "w") as f:
        for i, item in enumerate(data):
            if cancel_event.is_set():
                break

            result = pipe(
                input=item.get("input"),
                target=item.get("target"),
                **{k: v for k, v in item.items() if k not in ("input", "target")},
            )

            score = result.score
            cl = result.checklist
            id_to_q = {ci.id: ci.question for ci in cl.items} if cl else {}

            record = {
                "index": i,
                "input": item.get("input", ""),
                "target": item.get("target", ""),
                "pass_rate": score.pass_rate if score else None,
                "primary_score": score.primary_score if score else None,
                "primary_metric": score.primary_metric if score else None,
                "total_score": score.total_score if score else None,
                "item_scores": [
                    {
                        "item_id": s.item_id,
                        "question": id_to_q.get(s.item_id, s.item_id),
                        "answer": s.answer.value,
                        "reasoning": s.reasoning,
                    }
                    for s in (score.item_scores if score else [])
                ],
            }
            if score and score.weighted_score is not None:
                record["weighted_score"] = score.weighted_score
            f.write(json.dumps(record) + "\n")

            if score:
                scores.append(score.pass_rate)
                score_values.append(score.primary_score)
                for s in score.item_scores:
                    total_applicable += 1
                    if s.answer == ChecklistItemAnswer.YES:
                        total_yes += 1
            _update_summary(batch_dir, {"completed": i + 1})

    if cancel_event.is_set():
        _update_summary(batch_dir, {
            "status": "cancelled",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    else:
        macro_pr = statistics.mean(scores) if scores else None
        micro_pr = total_yes / total_applicable if total_applicable > 0 else None
        mean_sc = statistics.mean(score_values) if score_values else None

        _update_summary(batch_dir, {
            "status": "completed",
            "completed": len(scores),
            "mean_score": mean_sc,
            "macro_pass_rate": macro_pr,
            "micro_pass_rate": micro_pr,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


def _run_generate_only(batch_dir: Path, config: dict, data: list[dict], provider_kwargs: dict, cancel_event: threading.Event):
    """Generate checklists without scoring."""
    pipe = _build_pipeline(config, provider_kwargs, need_scorer=False)
    results_path = batch_dir / "results.jsonl"

    with open(results_path, "w") as f:
        for i, item in enumerate(data):
            if cancel_event.is_set():
                break
            checklist = pipe.generate(
                input=item.get("input", ""),
                reference=item.get("reference"),
            )
            record = {
                "index": i,
                "input": item.get("input", ""),
                "target": item.get("target", ""),
                "pass_rate": None,
                "total_score": None,
                "checklist_items": [
                    {"question": ci.question, "weight": ci.weight}
                    for ci in checklist.items
                ],
            }
            f.write(json.dumps(record) + "\n")
            _update_summary(batch_dir, {"completed": i + 1})

    if cancel_event.is_set():
        _update_summary(batch_dir, {
            "status": "cancelled",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    else:
        _update_summary(batch_dir, {
            "status": "completed",
            "completed": i + 1 if data else 0,
            "macro_pass_rate": None,
            "micro_pass_rate": None,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


def _run_score_only(batch_dir: Path, config: dict, data: list[dict], provider_kwargs: dict, cancel_event: threading.Event):
    """Score responses against a saved checklist."""
    checklist_id = config.get("checklist_id")
    if not checklist_id:
        raise ValueError("score_only mode requires a checklist_id")

    # Load the saved checklist
    checklist_path = DATA_DIR / "checklists" / f"{checklist_id}.json"
    saved = read_json(checklist_path)
    if not saved:
        raise ValueError(f"Checklist {checklist_id} not found")

    items = [
        LibChecklistItem(question=it["question"], weight=it.get("weight", 1.0))
        for it in saved["items"]
    ]
    checklist = LibChecklist(items=items, source_method=saved.get("method", "custom"), generation_level=saved.get("level", "instance"))

    # Build scorer — filter provider_kwargs to only scorer-relevant params
    # Prefer scorer_config dict if present, otherwise fall back to scorer name
    scorer_cfg = config.get("scorer_config")
    if scorer_cfg:
        # scorer_config is already a config dict — use ChecklistScorer directly
        from autochecklist.scorers import ChecklistScorer as _ScorerCls
        scorer_kwargs_cfg = {k: v for k, v in provider_kwargs.items() if k not in ("generator_model", "candidate_model")}
        scorer_model = scorer_kwargs_cfg.pop("scorer_model", None)
        if scorer_model:
            scorer_kwargs_cfg["model"] = scorer_model
        elif "model" not in scorer_kwargs_cfg:
            flat_model = provider_kwargs.get("model")
            if flat_model:
                scorer_kwargs_cfg["model"] = flat_model
        max_tokens = max(2048, len(items) * 100)
        scorer_instance = _ScorerCls(max_tokens=max_tokens, **scorer_kwargs_cfg, **scorer_cfg)
    else:
        scorer_name = config.get("scorer") or "batch"
        scorer_cls = get_scorer(scorer_name)
        # Scale max_tokens based on checklist size to avoid truncated JSON
        max_tokens = max(2048, len(items) * 100)
        scorer_kwargs = {k: v for k, v in provider_kwargs.items() if k not in ("generator_model", "candidate_model")}
        # Remap scorer_model → model
        scorer_model = scorer_kwargs.pop("scorer_model", None)
        if scorer_model:
            scorer_kwargs["model"] = scorer_model
        elif "model" not in scorer_kwargs:
            # Fall back to flat 'model' if present in original kwargs
            flat_model = provider_kwargs.get("model")
            if flat_model:
                scorer_kwargs["model"] = flat_model
        scorer_instance = scorer_cls(max_tokens=max_tokens, **scorer_kwargs)

    from autochecklist.models import ChecklistItemAnswer

    results_path = batch_dir / "results.jsonl"
    scores = []
    score_values = []
    total_yes = 0
    total_applicable = 0

    with open(results_path, "w") as f:
        for i, item in enumerate(data):
            if cancel_event.is_set():
                break
            score = scorer_instance.score(
                checklist=checklist,
                target=item.get("target", ""),
                input=item.get("input"),
            )
            id_to_q = {ci.id: ci.question for ci in checklist.items}
            record = {
                "index": i,
                "input": item.get("input", ""),
                "target": item.get("target", ""),
                "pass_rate": score.pass_rate,
                "primary_score": score.primary_score,
                "primary_metric": score.primary_metric,
                "total_score": score.total_score,
                "item_scores": [
                    {
                        "item_id": s.item_id,
                        "question": id_to_q.get(s.item_id, s.item_id),
                        "answer": s.answer.value,
                        "reasoning": s.reasoning,
                    }
                    for s in score.item_scores
                ],
            }
            if score.weighted_score is not None:
                record["weighted_score"] = score.weighted_score
            f.write(json.dumps(record) + "\n")
            scores.append(score.pass_rate)
            score_values.append(score.primary_score)
            for s in score.item_scores:
                total_applicable += 1
                if s.answer == ChecklistItemAnswer.YES:
                    total_yes += 1
            _update_summary(batch_dir, {"completed": i + 1})

    if cancel_event.is_set():
        _update_summary(batch_dir, {
            "status": "cancelled",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    else:
        import statistics
        macro_pr = statistics.mean(scores) if scores else None
        micro_pr = total_yes / total_applicable if total_applicable > 0 else None
        mean_sc = statistics.mean(score_values) if score_values else None

        _update_summary(batch_dir, {
            "status": "completed",
            "completed": len(scores),
            "mean_score": mean_sc,
            "macro_pass_rate": macro_pr,
            "micro_pass_rate": micro_pr,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


def _update_summary(batch_dir: Path, updates: dict):
    """Update summary.json atomically."""
    summary = read_json(batch_dir / "summary.json") or {}
    summary.update(updates)
    write_json(batch_dir / "summary.json", summary)


def cancel_batch(batch_id: str) -> bool:
    """Cancel a running batch job."""
    if batch_id in _cancel_flags:
        _cancel_flags[batch_id].set()
    if batch_id in _running_jobs:
        batch_dir = DATA_DIR / "batches" / batch_id
        _update_summary(batch_dir, {"status": "cancelled", "completed_at": datetime.now(timezone.utc).isoformat()})
        _running_jobs.pop(batch_id, None)
        _cancel_flags.pop(batch_id, None)
        return True
    return False


def get_batch_status(batch_id: str) -> Optional[dict]:
    """Get batch job status."""
    batch_dir = DATA_DIR / "batches" / batch_id
    summary = read_json(batch_dir / "summary.json")
    if summary:
        summary["batch_id"] = batch_id
        config = read_json(batch_dir / "config.json")
        if config:
            summary["config"] = config
    return summary


def list_batches() -> list[dict]:
    """List all batch jobs."""
    batches_dir = DATA_DIR / "batches"
    if not batches_dir.exists():
        return []

    results = []
    for batch_dir in sorted(batches_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if batch_dir.is_dir():
            summary = read_json(batch_dir / "summary.json")
            if summary:
                summary["batch_id"] = batch_dir.name
                config = read_json(batch_dir / "config.json")
                if config:
                    summary["config"] = config
                results.append(summary)
    return results


def delete_batch(batch_id: str) -> bool:
    """Delete a batch job and all its data."""
    return delete_data_dir("batches", batch_id)


def get_batch_results(batch_id: str, offset: int = 0, limit: int = 100) -> list[dict]:
    """Get per-item results for a batch job."""
    results_path = DATA_DIR / "batches" / batch_id / "results.jsonl"
    if not results_path.exists():
        return []

    results = []
    with open(results_path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(results) >= limit:
                break
            if line.strip():
                results.append(json.loads(line))
    return results
