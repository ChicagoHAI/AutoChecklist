"""Service layer wrapping auto-checklists library calls."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any

from autochecklist import ChecklistPipeline
from autochecklist.models import Checklist, Score
from autochecklist.scorers import ChecklistScorer

# Thread pool for running sync auto-checklists calls
executor = ThreadPoolExecutor(max_workers=10)

# Method constants
METHODS_NO_REF = ["tick", "rlcf_candidates_only"]
METHODS_WITH_REF = ["rlcf_direct", "rlcf_candidate", "rocketeval"]

# Map method names to pipeline preset generators.
# Contrastive methods need candidate_models so ContrastiveGenerator can auto-generate candidates.
METHOD_CONFIG = {
    "tick": {"generator": "tick"},
    "rlcf_direct": {"generator": "rlcf_direct"},
    "rlcf_candidate": {
        "generator": "rlcf_candidate",
        "generator_kwargs": {
            "candidate_models": ["openai/gpt-4o-mini"],
            "num_candidates": 3,
        },
    },
    "rlcf_candidates_only": {
        "generator": "rlcf_candidates_only",
        "generator_kwargs": {
            "candidate_models": ["openai/gpt-4o-mini"],
            "num_candidates": 3,
        },
    },
    "rocketeval": {"generator": "rocketeval"},
}

# Backend-specific metadata for each method (used by registry endpoint).
# Descriptions and requires_reference now come from the package's PIPELINE_PRESETS;
# this dict only holds backend-specific overrides like recommended_scorer.
METHOD_METADATA = {
    "tick": {"recommended_scorer": "batch"},
    "rlcf_direct": {"recommended_scorer": "weighted"},
    "rlcf_candidates_only": {"recommended_scorer": "weighted"},
    "rlcf_candidate": {"recommended_scorer": "weighted"},
    "rocketeval": {"recommended_scorer": "normalized"},
}


def _build_pipeline_from_config(
    config_data: dict,
    scorer_name: Optional[str] = None,
    candidate_model: Optional[str] = None,
    **provider_kwargs,
) -> ChecklistPipeline:
    """Build a ChecklistPipeline from a saved pipeline configuration.

    Maps generator_class to a base method preset:
      - "direct" → "tick"
      - "contrastive" → "rlcf_candidates_only"
    Then applies the custom generator/scorer prompts from the config.
    """
    gen_class = config_data.get("generator_class", "direct")
    base_method = "tick" if gen_class == "direct" else "rlcf_candidates_only"

    config = METHOD_CONFIG[base_method]
    gen_name = config["generator"]
    gen_kwargs = dict(config.get("generator_kwargs", {}))

    # Override candidate model only for contrastive methods (those with candidate_models in config)
    if candidate_model and "candidate_models" in gen_kwargs:
        gen_kwargs["candidate_models"] = [candidate_model]

    # Use scorer from config unless overridden
    effective_scorer = scorer_name or config_data.get("scorer_class", "batch")
    custom_prompt = config_data.get("generator_prompt")

    # Remap flat 'model' → generator_model + scorer_model
    model = provider_kwargs.pop("model", None)
    if model:
        provider_kwargs.setdefault("generator_model", model)
        provider_kwargs.setdefault("scorer_model", model)

    pipe = ChecklistPipeline(
        generator=gen_name,
        scorer=effective_scorer,
        generator_kwargs=gen_kwargs or None,
        custom_prompt=custom_prompt,
        **provider_kwargs,
    )

    return pipe


async def run_in_executor(func, *args, **kwargs):
    """Run a sync function in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


def _build_pipeline(
    method: str,
    scorer_name: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    candidate_model: Optional[str] = None,
    **provider_kwargs,
) -> ChecklistPipeline:
    """Build a ChecklistPipeline from a frontend method name or custom prompt.

    When custom_prompt is provided, the method is used only to select the
    base preset (defaults to "tick"); the custom prompt overrides the
    generator's template.

    provider_kwargs come from get_provider_kwargs() and contain:
    provider, model, api_key, base_url.  We remap 'model' to
    generator_model/scorer_model so everything passes cleanly to
    ChecklistPipeline.
    """
    if custom_prompt:
        # Custom prompt mode — use "tick" as base preset
        base_method = method if method in METHOD_CONFIG else "tick"
        config = METHOD_CONFIG[base_method]
    else:
        config = METHOD_CONFIG.get(method)
        if config is None:
            raise ValueError(f"Unknown method: {method}")

    gen_name = config["generator"]
    gen_kwargs = dict(config.get("generator_kwargs", {}))

    # Override candidate model only for contrastive methods (those with candidate_models in config)
    if candidate_model and "candidate_models" in gen_kwargs:
        gen_kwargs["candidate_models"] = [candidate_model]

    # Remap flat 'model' → generator_model + scorer_model
    model = provider_kwargs.pop("model", None)
    if model:
        provider_kwargs.setdefault("generator_model", model)
        provider_kwargs.setdefault("scorer_model", model)

    return ChecklistPipeline(
        generator=gen_name,
        scorer=scorer_name,
        generator_kwargs=gen_kwargs or None,
        custom_prompt=custom_prompt,
        **provider_kwargs,
    )


async def generate_only(
    method: str,
    input: str,
    target: Optional[str] = None,
    reference: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    candidate_model: Optional[str] = None,
    **provider_kwargs,
) -> Checklist:
    """Generate a checklist without scoring."""
    pipe = _build_pipeline(method, custom_prompt=custom_prompt, candidate_model=candidate_model, **provider_kwargs)

    gen_kwargs: dict[str, Any] = {"input": input}
    if target:
        gen_kwargs["target"] = target
    if reference:
        gen_kwargs["reference"] = reference

    return await run_in_executor(pipe.generate, **gen_kwargs)


_SCORER_NAME_TO_CONFIG = {
    "batch": {"mode": "batch"},
    "item": {"mode": "item", "capture_reasoning": True},
    "weighted": {"mode": "item", "primary_metric": "weighted"},
    "normalized": {"mode": "item", "use_logprobs": True, "primary_metric": "normalized"},
}


async def score_only(
    checklist: Checklist,
    target: str,
    input: Optional[str] = None,
    scorer_name: str | dict = "batch",
    custom_scorer_prompt: Optional[str] = None,
    **provider_kwargs,
) -> Score:
    """Score a response against an existing checklist."""
    max_tokens = max(2048, len(checklist.items) * 100)

    if isinstance(scorer_name, dict):
        # New-style config dict from DEFAULT_SCORERS
        # Resolve scorer_prompt → custom_prompt (template name → loaded text)
        scorer_config = dict(scorer_name)  # avoid mutating the preset dict
        # Normalized scoring requires logprobs for confidence values
        if scorer_config.get("primary_metric") == "normalized":
            scorer_config.setdefault("use_logprobs", True)
        if "scorer_prompt" in scorer_config:
            from autochecklist.prompts import load_template
            scorer_config["custom_prompt"] = load_template(
                "scoring", scorer_config.pop("scorer_prompt")
            )
        kwargs: dict[str, Any] = {**provider_kwargs, **scorer_config, "max_tokens": max_tokens}
    else:
        # Legacy string name — map to ChecklistScorer config
        config = _SCORER_NAME_TO_CONFIG.get(scorer_name, {"mode": "batch"})
        kwargs = {**provider_kwargs, **config, "max_tokens": max_tokens}

    if custom_scorer_prompt:
        kwargs["custom_prompt"] = custom_scorer_prompt

    scorer = ChecklistScorer(**kwargs)
    return await run_in_executor(
        scorer.score, checklist=checklist, target=target, input=input
    )
