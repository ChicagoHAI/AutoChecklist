"""Registry information endpoints for discovering available components."""

from fastapi import APIRouter, HTTPException

from autochecklist import (
    list_generators_with_info,
    list_scorers_with_info,
    list_refiners_with_info,
)
from autochecklist.generators.instance_level.pipeline_presets import PIPELINE_PRESETS
from autochecklist.pipeline import DEFAULT_SCORERS
from autochecklist.prompts import load_template

from app.services.checklist import METHOD_CONFIG, METHOD_METADATA

router = APIRouter()


def _expand_generators() -> list[dict]:
    """Build the full generator list for the frontend.

    - Instance-level generators in METHOD_CONFIG get expanded into variants
      (e.g. rlcf_direct, rlcf_candidate, rlcf_candidates_only are separate entries)
    - Corpus-level generators (not in METHOD_CONFIG) are passed through as-is
    """
    base_generators = list_generators_with_info()
    base_info = {g["name"]: g for g in base_generators}

    # Track which base generators are covered by METHOD_CONFIG
    covered_bases = {config["generator"] for config in METHOD_CONFIG.values()}

    expanded = []

    # Expand METHOD_CONFIG variants (instance-level)
    # Package now provides description, detail, requires_reference, default_scorer
    # via PIPELINE_PRESETS; METHOD_METADATA only adds backend-specific overrides.
    for variant_name, config in METHOD_CONFIG.items():
        base_name = config["generator"]
        base = base_info.get(base_name, {})
        meta = METHOD_METADATA.get(variant_name, {})

        # Map library class names to frontend-friendly keys
        preset = PIPELINE_PRESETS.get(variant_name, {})
        raw_class = preset.get("generator_class", "")
        gen_class = "contrastive" if "Contrastive" in raw_class else "direct"

        expanded.append({
            "name": variant_name,
            "level": base.get("level", "instance"),
            "generator_class": gen_class,
            "description": base.get("description", ""),
            "detail": base.get("detail", ""),
            "requires_reference": base.get("requires_reference", False),
            "recommended_scorer": meta.get(
                "recommended_scorer",
                base.get("default_scorer", "batch"),
            ),
        })

    # Map corpus-level generator names to their class names
    CORPUS_GENERATOR_CLASSES = {
        "feedback": "inductive",
        "checkeval": "deductive",
        "interacteval": "interactive",
    }

    # Include corpus-level generators not covered by METHOD_CONFIG
    for gen in base_generators:
        if gen["name"] not in covered_bases:
            expanded.append({
                **gen,
                "generator_class": CORPUS_GENERATOR_CLASSES.get(gen["name"], ""),
                "requires_reference": False,
                "recommended_scorer": gen.get("default_scorer", "batch"),
            })

    return expanded


def _normalize_scorer_config(config) -> dict:
    """Normalize a DEFAULT_SCORERS entry to a frontend-friendly config dict."""
    if isinstance(config, dict):
        return {
            "mode": config.get("mode", "batch"),
            "primary_metric": config.get("primary_metric", "pass"),
            "capture_reasoning": config.get("capture_reasoning", False),
        }
    # Legacy string name → config
    _name_to_config = {
        "batch": {"mode": "batch", "primary_metric": "pass", "capture_reasoning": False},
        "item": {"mode": "item", "primary_metric": "pass", "capture_reasoning": True},
        "weighted": {"mode": "item", "primary_metric": "weighted", "capture_reasoning": False},
        "normalized": {"mode": "item", "primary_metric": "normalized", "capture_reasoning": False},
    }
    return _name_to_config.get(str(config), {"mode": "batch", "primary_metric": "pass", "capture_reasoning": False})


def _expand_default_scorers() -> dict[str, dict]:
    """Build default_scorers map that includes METHOD_CONFIG variant names.

    Merges the library's DEFAULT_SCORERS with METHOD_METADATA recommended scorers
    so the frontend has a complete mapping for all method names.
    Values are config dicts with mode, primary_metric, capture_reasoning.
    """
    scorers = {
        name: _normalize_scorer_config(config)
        for name, config in DEFAULT_SCORERS.items()
    }
    for variant_name in METHOD_CONFIG:
        if variant_name not in scorers:
            meta = METHOD_METADATA.get(variant_name, {})
            recommended = meta.get("recommended_scorer", "batch")
            scorers[variant_name] = _normalize_scorer_config(recommended)
    return scorers


@router.get("")
async def get_registry():
    """Get all registry info in one call."""
    return {
        "generators": _expand_generators(),
        "scorers": list_scorers_with_info(),
        "refiners": list_refiners_with_info(),
        "default_scorers": _expand_default_scorers(),
    }


@router.get("/generators")
async def get_generators():
    """List available generators with metadata."""
    return _expand_generators()


@router.get("/scorers")
async def get_scorers():
    """List available scorers with metadata."""
    return list_scorers_with_info()


@router.get("/refiners")
async def get_refiners():
    """List available refiners with metadata."""
    return list_refiners_with_info()


@router.get("/default-scorers")
async def get_default_scorers():
    """Get default scorer for each generator."""
    return _expand_default_scorers()


@router.get("/preset-prompt/{preset_name}")
async def get_preset_prompt(preset_name: str):
    """Get the prompt template text for a pipeline preset."""
    preset = PIPELINE_PRESETS.get(preset_name)
    if not preset:
        raise HTTPException(status_code=404, detail="Unknown preset")
    text = load_template(preset["template_dir"], preset["template_name"])
    return {
        "preset": preset_name,
        "prompt_text": text,
        "generator_class": preset["generator_class"],
    }


@router.get("/scorer-prompt/{scorer_name}")
async def get_scorer_prompt(scorer_name: str):
    """Get the prompt template text for a scorer."""
    try:
        text = load_template("scoring", scorer_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown scorer")
    return {
        "scorer": scorer_name,
        "prompt_text": text,
    }


@router.get("/format/{format_name}")
async def get_format(format_name: str):
    """Get an output format template (e.g. 'checklist', 'batch_scoring')."""
    try:
        text = load_template("formats", format_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown format")
    return {
        "format": format_name,
        "text": text,
    }
