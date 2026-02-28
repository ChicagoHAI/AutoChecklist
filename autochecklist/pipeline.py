"""Composable pipeline for checklist generation and scoring.

This module provides a pipeline API for end-to-end
checklist workflows: Generator → Refiners → Scorer.

Example:
    >>> from autochecklist import pipeline
    >>> pipe = pipeline("tick", generator_model="openai/gpt-4o-mini")
    >>> result = pipe("Write a haiku", target="Leaves fall gently...")
    >>> print(result.pass_rate)  # 0.85

    >>> # Split models
    >>> pipe = pipeline("tick", generator_model="gpt-4o", scorer_model="gpt-4o-mini")

    >>> # Generate only (no target → no scoring)
    >>> checklists = pipe.generate_batch(inputs=["Write a haiku", "Write a poem"])

    >>> # Resumable batch
    >>> result = pipe.run_batch(data, output_path="results.jsonl")
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from .models import Checklist, ChecklistItem, Score, ItemScore, ChecklistItemAnswer, GroupedScore
from .registry import get_generator, get_scorer, get_refiner

if TYPE_CHECKING:
    import pandas as pd


# Default scorer for each generator — read from pipeline presets + corpus defaults.
# Values can be strings (backward compat) or config dicts (new style).
def _build_default_scorers() -> dict:
    from .generators.instance_level.pipeline_presets import PIPELINE_PRESETS
    scorers = {
        name: preset["default_scorer"]
        for name, preset in PIPELINE_PRESETS.items()
    }
    # Add corpus generators (not in presets)
    scorers.update({
        "feedback": {"mode": "batch", "primary_metric": "pass"},
        "checkeval": {"mode": "batch", "primary_metric": "pass"},
        "interacteval": {"mode": "batch", "primary_metric": "pass"},
    })
    return scorers

DEFAULT_SCORERS = _build_default_scorers()


def _scorer_config_to_name(config) -> str:
    """Convert a scorer config dict to a backward-compat name string.

    Used by UI/registry code that expects scorer names.
    """
    if isinstance(config, str):
        return config
    if not isinstance(config, dict):
        return "batch"
    mode = config.get("mode", "batch")
    if mode == "batch":
        return "batch"
    if config.get("use_logprobs"):
        return "normalized"
    pm = config.get("primary_metric", "pass")
    if pm == "weighted":
        return "weighted"
    if config.get("capture_reasoning"):
        return "item"
    return "item"


# ── JSONL Helper Functions ──────────────────────────────────────────────────


def _load_completed_indices(path: Optional[str]) -> Dict[int, dict]:
    """Load completed records from JSONL. Returns {index: record}."""
    if path is None or not os.path.exists(path):
        return {}
    completed = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" not in record:
                continue
            completed[record["index"]] = record
    return completed


def _checklist_from_record(record: dict) -> Checklist:
    """Reconstruct Checklist from a JSONL record."""
    cl_data = record.get("checklist", {})
    items = [
        ChecklistItem(
            id=item.get("id", str(i)),
            question=item.get("question", ""),
            weight=item.get("weight", 1.0),
            category=item.get("category"),
        )
        for i, item in enumerate(cl_data.get("items", []))
    ]
    return Checklist(
        id=cl_data.get("id", ""),
        items=items,
        source_method=cl_data.get("source_method", "unknown"),
        generation_level=cl_data.get("generation_level", "instance"),
        input=cl_data.get("input"),
    )


def _score_from_record(record: dict) -> Score:
    """Reconstruct Score from a JSONL record."""
    sc_data = record.get("score", {})
    item_scores = [
        ItemScore(
            item_id=s.get("item_id", str(i)),
            answer=ChecklistItemAnswer(s.get("answer", "no")),
            reasoning=s.get("reasoning"),
        )
        for i, s in enumerate(sc_data.get("item_scores", []))
    ]
    return Score(
        checklist_id=sc_data.get("checklist_id", ""),
        item_scores=item_scores,
        total_score=sc_data.get("total_score", 0.0),
        weighted_score=sc_data.get("weighted_score"),
        normalized_score=sc_data.get("normalized_score"),
        judge_model=sc_data.get("judge_model", "unknown"),
        scoring_method=sc_data.get("scoring_method", "batch"),
    )


def _checklist_to_dict(checklist: Checklist) -> dict:
    """Serialize a Checklist to a dict for JSONL."""
    return {
        "id": checklist.id,
        "items": [
            {
                "id": item.id,
                "question": item.question,
                "weight": item.weight,
                "category": item.category,
            }
            for item in checklist.items
        ],
        "source_method": checklist.source_method,
        "generation_level": checklist.generation_level,
        "input": checklist.input,
    }


def _score_to_dict(score: Score) -> dict:
    """Serialize a Score to a dict for JSONL."""
    result = {
        "checklist_id": score.checklist_id,
        "item_scores": [
            {
                "item_id": s.item_id,
                "answer": s.answer.value,
                "reasoning": s.reasoning,
            }
            for s in score.item_scores
        ],
        "total_score": score.total_score,
        "judge_model": score.judge_model,
        "scoring_method": score.scoring_method,
    }
    if score.weighted_score is not None:
        result["weighted_score"] = score.weighted_score
    if score.normalized_score is not None:
        result["normalized_score"] = score.normalized_score
    return result


def _record_from_result(index: int, item: dict, checklist: Checklist, score: Score) -> dict:
    """Serialize one gen+score result to JSONL record."""
    return {
        "index": index,
        "input": item.get("input", ""),
        "target": item.get("target", ""),
        "checklist": _checklist_to_dict(checklist),
        "score": _score_to_dict(score),
    }


def _checklist_record(index: int, item: dict, checklist: Checklist) -> dict:
    """Serialize one generate-only result to JSONL record."""
    return {
        "index": index,
        "input": item.get("input", ""),
        "checklist": _checklist_to_dict(checklist),
    }


# ── Data Classes ────────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Result from a single pipeline execution.

    Attributes:
        checklist: Generated (and optionally refined) checklist
        score: Score object if target was provided, None otherwise
    """
    checklist: Checklist
    score: Optional[Score] = None

    @property
    def pass_rate(self) -> Optional[float]:
        """Shortcut to score.pass_rate."""
        return self.score.pass_rate if self.score else None

    @property
    def weighted_score(self) -> Optional[float]:
        """Shortcut to score.weighted_score."""
        return self.score.weighted_score if self.score else None

    @property
    def normalized_score(self) -> Optional[float]:
        """Shortcut to score.normalized_score."""
        return self.score.normalized_score if self.score else None


@dataclass
class BatchResult:
    """Result from batch corpus evaluation.

    Attributes:
        checklist: The checklist used for evaluation (shared if provided,
            otherwise each score references its own checklist)
        scores: List of Score objects, one per input
        data: Original input data
        checklists: Individual checklists when not using shared checklist
    """
    scores: List[Score]
    data: List[Dict[str, Any]]
    checklist: Optional[Checklist] = None
    checklists: List[Checklist] = field(default_factory=list)

    @property
    def macro_pass_rate(self) -> float:
        """Macro-averaged pass rate across all scored examples.

        Computes pass_rate for each example independently, then averages.
        Each example contributes equally regardless of checklist size.

        Example: If example A scores 2/4 (0.5) and example B scores 3/3 (1.0),
        macro_pass_rate = (0.5 + 1.0) / 2 = 0.75
        """
        if not self.scores:
            return 0.0
        return sum(s.pass_rate for s in self.scores) / len(self.scores)

    @property
    def micro_pass_rate(self) -> float:
        """Micro-averaged pass rate (DFPR: Decomposed Requirements Following Ratio).

        Pools all checklist items across all examples into a single count.
        Examples with more checklist items have proportionally more influence.

        Example: If example A scores 2/4 and example B scores 3/3,
        micro_pass_rate = (2 + 3) / (4 + 3) = 5/7 ≈ 0.714
        """
        total_yes = 0
        total = 0
        for score in self.scores:
            for item_score in score.item_scores:
                total += 1
                if item_score.answer == ChecklistItemAnswer.YES:
                    total_yes += 1
        return total_yes / total if total > 0 else 0.0

    @property
    def mean_score(self) -> float:
        """Mean of Score.primary_score across all examples.

        Respects each Score's primary_metric — averages weighted_score
        for weighted pipelines, normalized_score for normalized, pass_rate for pass.
        """
        if not self.scores:
            return 0.0
        return sum(s.primary_score for s in self.scores) / len(self.scores)

    def per_category_pass_rates(self) -> List[Dict[str, float]]:
        """Compute per-category pass rates for each example.

        Uses the checklist(s) to map item IDs to categories, then computes
        pass rates per category for each scored example.

        Returns:
            List of dicts, one per example, mapping category -> pass_rate
        """
        results = []
        for i, score in enumerate(self.scores):
            # Get the checklist for this example
            if self.checklist is not None:
                cl = self.checklist
            elif i < len(self.checklists):
                cl = self.checklists[i]
            else:
                results.append({})
                continue

            # Build item_id -> category mapping
            id_to_cat = {item.id: (item.category or "ungrouped") for item in cl.items}

            # Group item scores by category
            cat_yes: Dict[str, int] = {}
            cat_total: Dict[str, int] = {}
            for item_score in score.item_scores:
                cat = id_to_cat.get(item_score.item_id, "ungrouped")
                cat_total[cat] = cat_total.get(cat, 0) + 1
                if item_score.answer == ChecklistItemAnswer.YES:
                    cat_yes[cat] = cat_yes.get(cat, 0) + 1

            rates = {}
            for cat, total in cat_total.items():
                rates[cat] = cat_yes.get(cat, 0) / total if total > 0 else 0.0
            results.append(rates)

        return results

    def to_dataframe(self) -> "pd.DataFrame":
        """Export results to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        rows = []
        for i, (item, score) in enumerate(zip(self.data, self.scores)):
            row = {
                "index": i,
                "input": item.get("input", ""),
                "target": item.get("target", ""),
                "pass_rate": score.pass_rate,
            }
            if score.weighted_score is not None:
                row["weighted_score"] = score.weighted_score
            if score.normalized_score is not None:
                row["normalized_score"] = score.normalized_score

            for item_score in score.item_scores:
                row[f"item_{item_score.item_id}"] = item_score.answer.value

            rows.append(row)

        return pd.DataFrame(rows)

    def to_jsonl(self, path: str) -> None:
        """Export results to JSONL file."""
        with open(path, "w") as f:
            for item, score in zip(self.data, self.scores):
                record = {
                    "input": item.get("input", ""),
                    "target": item.get("target", ""),
                    "pass_rate": score.pass_rate,
                    "item_scores": [
                        {
                            "item_id": s.item_id,
                            "answer": s.answer.value,
                            "reasoning": s.reasoning,
                        }
                        for s in score.item_scores
                    ],
                }
                if score.weighted_score is not None:
                    record["weighted_score"] = score.weighted_score
                if score.normalized_score is not None:
                    record["normalized_score"] = score.normalized_score
                f.write(json.dumps(record) + "\n")


# ── Pipeline ────────────────────────────────────────────────────────────────


class ChecklistPipeline:
    """Chains: Generator → Refiners → Scorer.

    A composable pipeline for checklist-based evaluation. Three construction
    modes:

    1. **Preset**: ``ChecklistPipeline(from_preset="tick")`` — resolves
       generator AND auto-attaches the preset's default scorer.
    2. **Explicit components**: ``ChecklistPipeline(generator="tick", scorer="batch")``
       — resolves each component by name. No auto scorer.
    3. **Pre-configured instances**: ``ChecklistPipeline(generator=my_gen, scorer=my_scorer)``

    The :func:`pipeline` factory is equivalent to mode 1.

    Args:
        generator: Generator name string (e.g., ``"tick"``, ``"rlcf_direct"``)
            or a pre-configured generator instance.
        refiners: Optional list of refiner instances or name strings.
        scorer: Optional scorer instance or name string (e.g., ``"batch"``,
            ``"weighted"``). Not auto-resolved unless using ``from_preset``.
        generator_model: Model for the generator (used when generator is a string).
        scorer_model: Model for the scorer (used when scorer is a string).
        provider: LLM provider ("openrouter", "openai", "vllm").
        base_url: Override base URL for the LLM provider.
        client: Injected LLM client instance.
        api_key: API key for the provider.
        api_format: API format ("chat" or "responses").
        generator_kwargs: Extra kwargs passed to generator constructor.
        scorer_kwargs: Extra kwargs passed to scorer constructor.
        from_preset: Pipeline preset name (e.g., ``"tick"``). Resolves generator
            and auto-attaches default scorer. Mutually exclusive with
            ``generator``.

    Example:
        >>> pipe = ChecklistPipeline(from_preset="tick",
        ...     generator_model="gpt-4o", scorer_model="gpt-4o-mini")

        >>> pipe = ChecklistPipeline(generator="tick", scorer="batch")

        >>> gen = DirectGenerator(method_name="tick", model="gpt-4o")
        >>> pipe = ChecklistPipeline(generator=gen, scorer="batch")
    """

    def __init__(
        self,
        generator: Optional[Any] = None,
        refiners: Optional[List[Union[str, Any]]] = None,
        scorer: Optional[Union[str, Any]] = None,
        # Model config
        generator_model: Optional[str] = None,
        scorer_model: Optional[str] = None,
        # Provider config (shared)
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
        api_key: Optional[str] = None,
        api_format: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        # Component-specific kwargs passthrough
        generator_kwargs: Optional[Dict[str, Any]] = None,
        scorer_kwargs: Optional[Dict[str, Any]] = None,
        # Preset loading (resolves generator + default scorer)
        from_preset: Optional[str] = None,
        # Custom prompt override for generator
        custom_prompt: Optional[Union[str, Path]] = None,
    ):
        # Thread custom_prompt into generator_kwargs
        if custom_prompt is not None:
            generator_kwargs = generator_kwargs or {}
            generator_kwargs["custom_prompt"] = custom_prompt

        # from_preset: resolve generator string AND auto-attach default scorer
        if from_preset is not None:
            if generator is not None:
                raise ValueError("Cannot specify both from_preset and generator")
            generator = from_preset
            # Auto-resolve scorer from preset if not explicitly provided
            if scorer is None:
                scorer = DEFAULT_SCORERS.get(from_preset, "batch")

        # Shared provider kwargs
        provider_kwargs = {k: v for k, v in {
            "provider": provider, "base_url": base_url,
            "client": client, "api_key": api_key, "api_format": api_format,
            "reasoning_effort": reasoning_effort,
        }.items() if v is not None}

        # -- Generator resolution --
        if isinstance(generator, str):
            gen_cls = get_generator(generator)
            gen_kw = {**provider_kwargs, **(generator_kwargs or {})}
            if generator_model:
                gen_kw["model"] = generator_model
            self.generator = gen_cls(**gen_kw)
            self._generator_name = generator
        elif generator is not None:
            self.generator = generator
            self._generator_name = getattr(generator, 'method_name', 'custom')
        else:
            raise ValueError("Must provide generator (name string or instance)")

        # -- Refiner resolution --
        self.refiners = []
        if refiners:
            for refiner in refiners:
                if isinstance(refiner, str):
                    ref_cls = get_refiner(refiner)
                    ref_kw = {**provider_kwargs}
                    self.refiners.append(ref_cls(**ref_kw))
                else:
                    self.refiners.append(refiner)

        # -- Scorer resolution --
        if isinstance(scorer, dict):
            # Config dict from pipeline presets (new style)
            # Resolve scorer_prompt key → custom_prompt before constructing
            scorer = dict(scorer)  # avoid mutating the preset dict
            if "scorer_prompt" in scorer:
                prompt_ref = scorer.pop("scorer_prompt")
                try:
                    from .prompts import load_template
                    scorer["custom_prompt"] = load_template(
                        "scoring", prompt_ref
                    )
                except FileNotFoundError:
                    # Not a built-in name — treat as inline prompt text
                    scorer["custom_prompt"] = prompt_ref
            from .scorers import ChecklistScorer
            scorer_kw = {**provider_kwargs, **scorer, **(scorer_kwargs or {})}
            if scorer_model:
                scorer_kw["model"] = scorer_model
            self.scorer = ChecklistScorer(**scorer_kw)
        elif isinstance(scorer, str):
            scorer_cls = get_scorer(scorer)
            scorer_kw = {**provider_kwargs, **(scorer_kwargs or {})}
            if scorer_model:
                scorer_kw["model"] = scorer_model
            self.scorer = scorer_cls(**scorer_kw)
        elif scorer is not None:
            self.scorer = scorer  # pre-configured instance
        else:
            self.scorer = None  # generator instance without scorer

    @property
    def is_instance_level(self) -> bool:
        """Check if the generator is instance-level."""
        return self.generator.generation_level == "instance"

    @property
    def is_corpus_level(self) -> bool:
        """Check if the generator is corpus-level."""
        return self.generator.generation_level == "corpus"

    def __call__(
        self,
        input: Optional[str] = None,
        target: Optional[str] = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """Run the full pipeline: generate → refine → score.

        For instance-level generators, pass input and target.
        For corpus-level generators, pass the appropriate inputs via kwargs
        (e.g., feedback=..., dimensions=...).

        Args:
            input: Input instruction/query (for instance-level)
            target: Target response to evaluate (optional for generation-only)
            **kwargs: Additional arguments passed to generator

        Returns:
            PipelineResult with checklist and optional score
        """
        checklist = self.generate(input=input, **kwargs)

        score = None
        if target is not None:
            score = self.score(checklist, target, input=input)

        return PipelineResult(checklist=checklist, score=score)

    def generate(
        self,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> Checklist:
        """Generate a checklist (without scoring).

        Args:
            input: Input instruction/query (for instance-level)
            **kwargs: Additional arguments for generator

        Returns:
            Generated and refined checklist
        """
        if self.generator is None:
            raise RuntimeError("No generator configured. Provide generator (name or instance) to use generate().")

        if self.is_instance_level:
            if input is None:
                raise ValueError("input is required for instance-level generators")
            checklist = self.generator.generate(input=input, **kwargs)
        else:
            checklist = self.generator.generate(**kwargs)

        return self.refine(checklist)

    def refine(self, checklist: Checklist) -> Checklist:
        """Apply refiners to a checklist."""
        for refiner in self.refiners:
            checklist = refiner.refine(checklist)
        return checklist

    def score(
        self,
        checklist: Checklist,
        target: str,
        input: Optional[str] = None,
    ) -> Score:
        """Score a target response against a checklist.

        Args:
            checklist: Checklist to evaluate against
            target: Target response to score
            input: Optional input for context

        Returns:
            Score object
        """
        if self.scorer is None:
            raise RuntimeError(
                "No scorer configured. Provide scorer (name or instance) to use score(). Use pipeline() for automatic scorer defaults."
            )
        return self.scorer.score(
            checklist=checklist,
            target=target,
            input=input,
        )

    def score_group(
        self,
        sub_checklists: Dict[str, Checklist],
        target: str,
        input: Optional[str] = None,
    ) -> "GroupedScore":
        """Score a target against sub-checklists (one per category).

        Typically used with ``checklist.by_category()`` output.

        Args:
            sub_checklists: Dict mapping category name to sub-Checklist
            target: Target response to score
            input: Optional input for context

        Returns:
            GroupedScore with per-category Score objects
        """
        if self.scorer is None:
            raise RuntimeError("No scorer configured.")
        scores = {}
        for name, sub_cl in sub_checklists.items():
            scores[name] = self.score(sub_cl, target, input=input)
        return GroupedScore(scores=scores)

    def score_batch(
        self,
        checklist: Checklist,
        targets: List[str],
        inputs: Optional[List[str]] = None,
        show_progress: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Score]:
        """Score multiple targets against a single checklist."""
        return self._run_batch_scoring(
            checklist=checklist,
            targets=targets,
            inputs=inputs,
            show_progress=show_progress,
            on_progress=on_progress,
        )

    def generate_batch(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        inputs: Optional[List[str]] = None,
        show_progress: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
        output_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> List[Checklist]:
        """Generate checklists for a batch of inputs (no scoring).

        Only works for instance-level generators (1:1 input → checklist).
        For corpus-level generators, call generator.generate() directly.

        Args:
            data: List of dicts with "input" key
            inputs: List of input strings (convenience alternative to data)
            show_progress: Show progress bar
            on_progress: Callback(completed, total) fired after each item
            output_path: Path to JSONL file for incremental writes + resume
            overwrite: If True, delete existing output_path before starting

        Returns:
            List of Checklist objects
        """
        if self.generator is None:
            raise RuntimeError("No generator configured.")
        if not self.is_instance_level:
            raise RuntimeError(
                "generate_batch() only works for instance-level generators. "
                "Corpus-level generators produce one checklist from all inputs — "
                "call generator.generate() directly."
            )

        if data is None:
            if inputs is None:
                raise ValueError("Provide either 'data' or 'inputs'")
            data = [{"input": inp} for inp in inputs]

        total = len(data)

        # Resume logic
        if overwrite and output_path and os.path.exists(output_path):
            os.remove(output_path)
        completed = _load_completed_indices(output_path) if output_path else {}

        checklists: List[Optional[Checklist]] = [None] * total
        for idx, record in completed.items():
            if 0 <= idx < total:
                checklists[idx] = _checklist_from_record(record)

        out_file = open(output_path, "a") if output_path else None
        try:
            for i, item in enumerate(data):
                if i in completed:
                    if on_progress:
                        on_progress(i + 1, total)
                    continue
                checklist = self.generate(input=item.get("input", ""), **{
                    k: v for k, v in item.items() if k not in ("input", "target")
                })
                checklists[i] = checklist
                if out_file:
                    record = _checklist_record(i, item, checklist)
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                if on_progress:
                    on_progress(i + 1, total)
        finally:
            if out_file:
                out_file.close()

        return [c for c in checklists if c is not None]

    def run_batch(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        checklist: Optional[Checklist] = None,
        inputs: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        show_progress: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
        output_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> BatchResult:
        """Run batch evaluation on a corpus.

        Can be called with either:
        1. data: List of dicts with "input" and "target" keys
        2. inputs + targets: Separate lists

        Args:
            data: List of dicts with input/target pairs
            checklist: Optional shared checklist to use for all evaluations
            inputs: Optional list of inputs (alternative to data)
            targets: Optional list of targets (alternative to data)
            show_progress: Show progress bar
            on_progress: Optional callback
            output_path: Path to JSONL file for incremental writes + resume
            overwrite: If True, delete existing output_path before starting

        Returns:
            BatchResult with scores and aggregated metrics
        """
        if data is None:
            if inputs is None or targets is None:
                raise ValueError("Provide either 'data' or both 'inputs' and 'targets'")
            if len(inputs) != len(targets):
                raise ValueError("inputs and targets must have same length")
            data = [
                {"input": inp, "target": tgt}
                for inp, tgt in zip(inputs, targets)
            ]

        if output_path is not None and checklist is not None:
            raise ValueError(
                "output_path with shared checklist is not supported. "
                "Use score_batch() for shared-checklist scoring."
            )

        if checklist is not None:
            target_list = [d.get("target", "") for d in data]
            input_list = [d.get("input") for d in data]
            scores = self._run_batch_scoring(
                checklist=checklist,
                targets=target_list,
                inputs=input_list,
                show_progress=show_progress,
                on_progress=on_progress,
            )
            return BatchResult(
                scores=scores,
                data=data,
                checklist=checklist,
            )
        else:
            return self._run_batch_generation_and_scoring(
                data=data,
                show_progress=show_progress,
                on_progress=on_progress,
                output_path=output_path,
                overwrite=overwrite,
            )

    def _run_batch_generation_and_scoring(
        self,
        data: List[Dict[str, Any]],
        show_progress: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
        output_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> BatchResult:
        """Generate and score for each item in the batch."""
        total = len(data)

        # Resume logic
        if overwrite and output_path and os.path.exists(output_path):
            os.remove(output_path)
        completed = _load_completed_indices(output_path) if output_path else {}

        gen_pbar = None
        score_pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
                num_completed = len(completed)
                gen_pbar = tqdm(total=total, desc="Generating", initial=num_completed)
                score_pbar = tqdm(total=total, desc="Scoring", initial=num_completed)
            except ImportError:
                pass

        checklists: List[Optional[Checklist]] = [None] * total
        scores: List[Optional[Score]] = [None] * total

        # Load completed results
        for idx, record in completed.items():
            if 0 <= idx < total:
                checklists[idx] = _checklist_from_record(record)
                scores[idx] = _score_from_record(record)

        out_file = open(output_path, "a") if output_path else None
        try:
            for i, item in enumerate(data):
                if i in completed:
                    if on_progress:
                        on_progress(i + 1, total)
                    continue

                inp = item.get("input", "")
                tgt = item.get("target", "")

                cl = self.generate(input=inp)
                checklists[i] = cl
                if gen_pbar:
                    gen_pbar.update(1)

                sc = self.score(cl, tgt, input=inp)
                scores[i] = sc
                if score_pbar:
                    score_pbar.update(1)

                if out_file:
                    record = _record_from_result(i, item, cl, sc)
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()

                if on_progress:
                    on_progress(i + 1, total)

            return BatchResult(
                scores=[s for s in scores if s is not None],
                data=data,
                checklists=[c for c in checklists if c is not None],
            )
        finally:
            if out_file:
                out_file.close()
            if gen_pbar:
                gen_pbar.close()
            if score_pbar:
                score_pbar.close()

    def _run_batch_scoring(
        self,
        checklist: Checklist,
        targets: List[str],
        inputs: Optional[List[str]] = None,
        show_progress: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Score]:
        """Score multiple targets with concurrency support."""
        pbar = None
        total = len(targets)

        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total, desc="Scoring")
            except ImportError:
                pass

        try:
            scores = []
            for i, target in enumerate(targets):
                inp = inputs[i] if inputs else None
                score = self.score(checklist, target, input=inp)
                scores.append(score)
                if pbar:
                    pbar.update(1)
                if on_progress:
                    on_progress(i + 1, total)
            return scores
        finally:
            if pbar:
                pbar.close()

    def run_batch_from_file(
        self,
        path: str,
        checklist: Optional[Checklist] = None,
        input_key: str = "input",
        target_key: str = "target",
        show_progress: bool = False,
    ) -> BatchResult:
        """Run batch evaluation from a JSONL file."""
        data = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                data.append({
                    "input": obj.get(input_key, ""),
                    "target": obj.get(target_key, ""),
                })

        return self.run_batch(
            data=data,
            checklist=checklist,
            show_progress=show_progress,
        )


# ── Factory Function ────────────────────────────────────────────────────────


def pipeline(
    task: Optional[str] = None,
    generator_model: Optional[str] = None,
    scorer_model: Optional[str] = None,
    refiners: Optional[List[Union[str, Any]]] = None,
    scorer: Optional[Union[str, Any]] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Any = None,
    api_key: Optional[str] = None,
    api_format: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    generator_kwargs: Optional[Dict[str, Any]] = None,
    scorer_kwargs: Optional[Dict[str, Any]] = None,
    custom_prompt: Optional[Union[str, Path]] = None,
) -> ChecklistPipeline:
    """Convenience factory that creates a ChecklistPipeline from a pipeline preset.

    Equivalent to ``ChecklistPipeline(from_preset=task, ...)``. Resolves
    both the generator and the preset's default scorer automatically.

    Can also be called with ``custom_prompt=`` instead of a task name to
    create a pipeline from a custom prompt without registration.

    Args:
        task: Pipeline name or alias. Options:
            - Instance-level: "tick", "rlcf_direct", "rocketeval",
              "rlcf_candidate", "rlcf_candidates_only"
            - Corpus-level: "feedback", "checkeval", "interacteval"
            Can be None if custom_prompt is provided.
        generator_model: Model for the generator
        scorer_model: Model for the scorer
        refiners: Optional list of refiner names or instances
        scorer: Optional scorer name or instance (overrides preset default)
        provider: LLM provider ("openrouter", "openai", "vllm")
        base_url: Override base URL (e.g., vLLM server URL)
        client: Injected LLM client (e.g., VLLMOfflineClient)
        api_key: API key for the provider
        api_format: API format ("chat" or "responses")
        generator_kwargs: Extra kwargs for generator (e.g., candidate config)
        scorer_kwargs: Extra kwargs for scorer
        custom_prompt: Custom generator prompt. Pass a Path to load from
            file, or a str for raw prompt text. No registration needed.

    Returns:
        Configured ChecklistPipeline

    Example:
        >>> pipe = pipeline("tick", generator_model="gpt-4o-mini")
        >>> pipe = pipeline("tick", generator_model="gpt-4o", scorer_model="gpt-4o-mini")
        >>> pipe = pipeline(custom_prompt=Path("my_eval.md"), scorer_model="gpt-4o-mini")
    """
    # If custom_prompt provided without task, default to "tick" preset
    if task is None and custom_prompt is not None:
        task = "tick"

    if task is None:
        raise ValueError("Must provide 'task' (pipeline name) or 'custom_prompt'")

    return ChecklistPipeline(
        from_preset=task,
        generator_model=generator_model,
        scorer_model=scorer_model,
        refiners=refiners,
        scorer=scorer,
        provider=provider,
        base_url=base_url,
        client=client,
        api_key=api_key,
        api_format=api_format,
        reasoning_effort=reasoning_effort,
        generator_kwargs=generator_kwargs,
        scorer_kwargs=scorer_kwargs,
        custom_prompt=custom_prompt,
    )
