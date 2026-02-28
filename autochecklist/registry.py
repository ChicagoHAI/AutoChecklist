"""Component registry for dynamic discovery and instantiation.

This module provides a registry system for generators, scorers, and refiners
that enables:
- UI discovery: List available components for dropdown menus
- Config-driven instantiation: Create components from string names
- Custom extensions: Register user-defined components

Example:
    >>> from autochecklist import list_generators, get_generator
    >>> print(list_generators())  # ['tick', 'rlcf_direct', 'rocketeval', ...]
    >>> gen_cls = get_generator("tick")
    >>> generator = gen_cls(model="openai/gpt-4o-mini")
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Union


# Internal registries
_generators: Dict[str, Type] = {}
_scorers: Dict[str, Type] = {}
_refiners: Dict[str, Type] = {}

# Track built-in names to protect against accidental overrides
_builtin_generators: set = set()


# Registration decorators

def register_generator(name: str):
    """Decorator to register a generator class."""
    def decorator(cls: Type) -> Type:
        _generators[name] = cls
        return cls
    return decorator


def register_scorer(name: str):
    """Decorator to register a scorer class."""
    def decorator(cls: Type) -> Type:
        _scorers[name] = cls
        return cls
    return decorator


def register_refiner(name: str):
    """Decorator to register a refiner class."""
    def decorator(cls: Type) -> Type:
        _refiners[name] = cls
        return cls
    return decorator


# Discovery functions (for UI)

def list_generators() -> List[str]:
    """List all registered generator names."""
    return list(_generators.keys())


def list_scorers() -> List[str]:
    """List all registered scorer names."""
    return list(_scorers.keys())


def list_refiners() -> List[str]:
    """List all registered refiner names."""
    return list(_refiners.keys())


# Instantiation functions (for config-driven usage)

def get_generator(name: str) -> Type:
    """Get generator class by name."""
    if name not in _generators:
        raise KeyError(f"Unknown generator: {name}. Available: {list_generators()}")
    return _generators[name]


def get_scorer(name: str) -> Type:
    """Get scorer class by name."""
    if name not in _scorers:
        raise KeyError(f"Unknown scorer: {name}. Available: {list_scorers()}")
    return _scorers[name]


def get_refiner(name: str) -> Type:
    """Get refiner class by name."""
    if name not in _refiners:
        raise KeyError(f"Unknown refiner: {name}. Available: {list_refiners()}")
    return _refiners[name]


# Component metadata (for UI display)

def get_generator_info(name: str) -> Dict[str, Any]:
    """Get generator metadata for UI display."""
    cls = get_generator(name)
    try:
        instance = cls.__new__(cls)
        level = getattr(cls, '_generation_level', None)
        if level is None:
            instance = cls()
            level = instance.generation_level
    except Exception:
        level = "unknown"

    preset = getattr(cls, '_preset', {})

    raw_scorer = preset.get("default_scorer")
    # Normalize to string for backward compat (UI expects strings)
    from .pipeline import _scorer_config_to_name
    default_scorer = _scorer_config_to_name(raw_scorer) if raw_scorer else None

    return {
        "name": name,
        "level": level,
        "description": preset.get("description") or (cls.__doc__.split('\n')[0] if cls.__doc__ else ""),
        "detail": preset.get("detail") or _extract_docstring_detail(cls),
        "requires_reference": preset.get("requires_reference", False),
        "default_scorer": default_scorer,
    }


def _extract_docstring_detail(cls: type) -> str:
    """Extract the body of a docstring, excluding Example/Args sections."""
    if not cls.__doc__:
        return ""
    lines = cls.__doc__.strip().split('\n')
    body_lines = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.lower().startswith(("example", "args:", "attributes:", "note:")):
            break
        body_lines.append(stripped)
    detail = ' '.join(line for line in body_lines if line).strip()
    return detail


def get_scorer_info(name: str) -> Dict[str, Any]:
    """Get scorer metadata for UI display."""
    scorer_factory = get_scorer(name)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            instance = scorer_factory()
        method = instance.scoring_method
    except Exception:
        method = "unknown"

    doc = getattr(scorer_factory, "__doc__", "") or ""
    return {
        "name": name,
        "method": method,
        "description": doc.split('\n')[0] if doc else "",
        "detail": _extract_docstring_detail_from_doc(doc),
    }


def _extract_docstring_detail_from_doc(doc: str) -> str:
    """Extract body from a docstring string."""
    if not doc:
        return ""
    lines = doc.strip().split('\n')
    body_lines = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.lower().startswith(("example", "args:", "attributes:", "note:")):
            break
        body_lines.append(stripped)
    return ' '.join(line for line in body_lines if line).strip()


def get_refiner_info(name: str) -> Dict[str, Any]:
    """Get refiner metadata for UI display."""
    cls = get_refiner(name)
    return {
        "name": name,
        "description": cls.__doc__.split('\n')[0] if cls.__doc__ else "",
    }


def list_generators_with_info() -> List[Dict[str, Any]]:
    """List all generators with metadata (for UI dropdowns)."""
    return [get_generator_info(name) for name in list_generators()]


def list_scorers_with_info() -> List[Dict[str, Any]]:
    """List all scorers with metadata (for UI dropdowns)."""
    return [get_scorer_info(name) for name in list_scorers()]


def list_refiners_with_info() -> List[Dict[str, Any]]:
    """List all refiners with metadata (for UI dropdowns)."""
    return [get_refiner_info(name) for name in list_refiners()]


def _make_factory(generator_class: Type, method_name: str, preset: dict = None) -> Type:
    """Create a factory class that instantiates a generator with preset defaults."""
    class _Factory(generator_class):
        def __init__(self, **kwargs):
            super().__init__(method_name=method_name, **kwargs)

    _Factory.__name__ = f"{generator_class.__name__}({method_name})"
    if preset and preset.get("description"):
        detail = preset.get("detail", "")
        _Factory.__doc__ = preset["description"] + ("\n\n" + detail if detail else "")
    else:
        _Factory.__doc__ = (
            f"Auto-configured {generator_class.__name__} for '{method_name}' pipeline."
        )
    _Factory._preset = preset or {}
    return _Factory


def register_custom_generator(name: str, prompt_path: str) -> None:
    """Register a custom generator from a .md prompt file.

    Once registered, the generator can be used by name in pipeline() or get_generator().

    Args:
        name: Name to register the generator under
        prompt_path: Path to .md file containing the prompt template

    Example:
        >>> register_custom_generator("my_eval", "prompts/my_eval.md")
        >>> pipe = pipeline("my_eval")
    """
    from .generators.instance_level.direct import DirectGenerator

    # Read file content and pass as custom_prompt text
    prompt_text = Path(prompt_path).read_text(encoding="utf-8")

    class _CustomFactory(DirectGenerator):
        def __init__(self, **kwargs):
            super().__init__(custom_prompt=prompt_text, method_name=name, **kwargs)

    _CustomFactory.__name__ = f"DirectGenerator({name})"
    _CustomFactory.__doc__ = f"Custom generator '{name}' loaded from {prompt_path}"
    _generators[name] = _CustomFactory


def register_custom_scorer(
    name: str,
    prompt_path: str,
    mode: str = "batch",
    primary_metric: str = "pass",
    capture_reasoning: bool = False,
) -> None:
    """Register a custom scorer from a .md prompt file.

    Creates a ChecklistScorer with the custom prompt. Once registered,
    the scorer can be used by name in pipeline() or get_scorer().

    When ``primary_metric="normalized"``, logprobs are automatically enabled
    (logprobs are required for confidence-calibrated scoring).

    Args:
        name: Name to register the scorer under
        prompt_path: Path to .md file containing the scoring prompt template
        mode: Scoring mode — "batch" (all items in one call) or "item"
            (one item per call). Default: "batch".
        primary_metric: Which metric ``Score.primary_score`` aliases —
            "pass", "weighted", or "normalized". Default: "pass".
            Setting "normalized" auto-enables logprobs.
        capture_reasoning: Include per-item reasoning in output.

    Example:
        >>> register_custom_scorer("strict", "prompts/strict_scorer.md")
        >>> pipe = pipeline("tick", scorer="strict")

        >>> register_custom_scorer(
        ...     "weighted_strict", "prompts/strict_scorer.md",
        ...     mode="item", primary_metric="weighted",
        ... )
    """
    custom_prompt = Path(prompt_path).read_text(encoding="utf-8")
    _mode = mode
    _primary_metric = primary_metric
    _use_logprobs = primary_metric == "normalized"
    _capture_reasoning = capture_reasoning

    def _factory(**kwargs):
        from .scorers import ChecklistScorer
        kwargs.setdefault("custom_prompt", custom_prompt)
        return ChecklistScorer(
            mode=_mode,
            primary_metric=_primary_metric,
            use_logprobs=_use_logprobs,
            capture_reasoning=_capture_reasoning,
            **kwargs,
        )

    _factory.__name__ = f"CustomScorer({name})"
    _factory.__doc__ = f"Custom scorer '{name}' loaded from {prompt_path}"
    _scorers[name] = _factory


def register_custom_pipeline(
    name: str,
    pipeline: Optional[Any] = None,
    generator_prompt: Optional[Union[str, Path]] = None,
    generator_class: str = "direct",
    scorer: Optional[str] = None,
    scorer_mode: Optional[str] = None,
    scorer_prompt: Optional[Union[str, Path]] = None,
    primary_metric: Optional[str] = None,
    capture_reasoning: Optional[bool] = None,
    force: bool = False,
) -> None:
    """Register a custom pipeline as a reusable preset.

    Can register from either an instantiated pipeline or from config values.
    Once registered, the pipeline can be used by name:
        ``pipeline("my_eval", generator_model="openai/gpt-4o")``

    When ``primary_metric="normalized"``, logprobs are automatically enabled
    (logprobs are required for confidence-calibrated scoring).

    Args:
        name: Name to register the pipeline under.
        pipeline: An instantiated ChecklistPipeline to extract config from.
            Mutually exclusive with generator_prompt.
        generator_prompt: Custom generator prompt text, or Path to a prompt file.
            Mutually exclusive with pipeline.
        generator_class: Generator class to use ("direct" or "contrastive").
            Only used with generator_prompt. Default: "direct".
        scorer: Deprecated scorer name (e.g., "batch", "weighted"). Use
            ``scorer_mode`` and ``primary_metric`` instead.
        scorer_mode: Scoring mode — "batch" or "item". None means no
            default scorer is attached.
        scorer_prompt: Custom scorer prompt text, built-in name ("rlcf",
            "rocketeval"), or Path to a prompt file. None means use the
            default prompt for the mode.
        primary_metric: Which metric ``Score.primary_score`` aliases —
            "pass" (default), "weighted", or "normalized".
            Setting "normalized" auto-enables logprobs.
        capture_reasoning: Include per-item reasoning in output.
        force: If True, allow overriding built-in pipelines (with a warning).

    Raises:
        ValueError: If overriding a built-in name without force=True, if
            neither pipeline nor generator_prompt is provided, or if both
            ``scorer`` and ``scorer_mode`` are provided.

    Example:
        >>> # From config with scorer settings
        >>> register_custom_pipeline(
        ...     "my_eval",
        ...     generator_prompt="Generate yes/no questions for:\\n\\n{input}",
        ...     scorer_mode="item",
        ...     primary_metric="weighted",
        ... )
        >>> pipe = pipeline("my_eval", generator_model="openai/gpt-4o-mini")

        >>> # From instantiated pipeline
        >>> pipe = ChecklistPipeline(
        ...     generator=DirectGenerator(custom_prompt="...", model="gpt-4o-mini"),
        ...     scorer=ChecklistScorer(mode="item", primary_metric="weighted",
        ...                            model="gpt-4o-mini"),
        ... )
        >>> register_custom_pipeline("my_eval", pipe)
    """
    from .generators.instance_level.direct import DirectGenerator
    from .generators.instance_level.contrastive import ContrastiveGenerator

    # Validate: scorer (legacy name) and scorer_mode are mutually exclusive
    if scorer is not None and scorer_mode is not None:
        raise ValueError(
            "Cannot specify both 'scorer' (legacy name) and 'scorer_mode'. "
            "Use scorer_mode with primary_metric/capture_reasoning instead."
        )

    # Override protection
    if name in _builtin_generators and not force:
        raise ValueError(
            f"Cannot override built-in pipeline '{name}'. "
            f"Use force=True to override."
        )
    if name in _builtin_generators and force:
        warnings.warn(
            f"Overriding built-in pipeline '{name}'.",
            UserWarning,
            stacklevel=2,
        )

    # Extract config from pipeline instance or direct args
    if pipeline is not None and generator_prompt is not None:
        raise ValueError(
            "Provide either 'pipeline' or 'generator_prompt', not both."
        )

    scorer_config = None  # Will hold the config dict for DEFAULT_SCORERS

    if pipeline is not None:
        # Extract from instantiated pipeline
        gen = pipeline.generator
        gen_prompt_text = gen.prompt_text

        # Detect generator class
        if isinstance(gen, ContrastiveGenerator):
            gen_class_name = "contrastive"
        else:
            gen_class_name = "direct"

        # Extract full scorer config from pipeline instance
        if pipeline.scorer is not None:
            s = pipeline.scorer
            scorer_config = {
                "mode": s.mode,
                "primary_metric": s.primary_metric,
            }
            if s.use_logprobs:
                scorer_config["use_logprobs"] = True
            if s.capture_reasoning:
                scorer_config["capture_reasoning"] = True

            # Check if scorer has a custom prompt (differs from default)
            scorer_prompt_text = s.prompt_text
            try:
                from .scorers import ChecklistScorer as _SC
                default_instance = _SC(mode=s.mode)
                if default_instance.prompt_text != scorer_prompt_text:
                    scorer_config["scorer_prompt"] = scorer_prompt_text
            except Exception:
                scorer_config["scorer_prompt"] = scorer_prompt_text

    elif generator_prompt is not None:
        # Read from path if needed
        if isinstance(generator_prompt, Path):
            gen_prompt_text = generator_prompt.read_text(encoding="utf-8")
        else:
            gen_prompt_text = generator_prompt
        gen_class_name = generator_class

        # Build scorer config from flat kwargs or legacy scorer name
        if scorer_mode is not None or primary_metric is not None or \
           capture_reasoning is not None:
            scorer_config = {}
            if scorer_mode is not None:
                scorer_config["mode"] = scorer_mode
            if primary_metric is not None:
                scorer_config["primary_metric"] = primary_metric
                # Auto-enable logprobs for normalized metric
                if primary_metric == "normalized":
                    scorer_config["use_logprobs"] = True
            if capture_reasoning is not None and capture_reasoning:
                scorer_config["capture_reasoning"] = True
            # Handle scorer_prompt
            if scorer_prompt is not None:
                scorer_config["scorer_prompt"] = _read_scorer_prompt(scorer_prompt)
        elif scorer is not None:
            # Legacy path: map old scorer name string to config dict
            warnings.warn(
                "The 'scorer' parameter is deprecated. Use 'scorer_mode' and "
                "'primary_metric' instead (e.g., scorer_mode='item', "
                "primary_metric='weighted').",
                DeprecationWarning,
                stacklevel=2,
            )
            scorer_config = _scorer_name_to_config(scorer)
            # Add scorer_prompt if provided
            if scorer_prompt is not None:
                scorer_config["scorer_prompt"] = _read_scorer_prompt(scorer_prompt)
        elif scorer_prompt is not None:
            # scorer_prompt alone (no mode/metric kwargs) — default batch mode
            scorer_config = {"scorer_prompt": _read_scorer_prompt(scorer_prompt)}
    else:
        raise ValueError(
            "Must provide either 'pipeline' (ChecklistPipeline instance) "
            "or 'generator_prompt' (str or Path)."
        )

    # Create generator factory
    gen_cls_map = {
        "direct": DirectGenerator,
        "contrastive": ContrastiveGenerator,
    }
    gen_cls = gen_cls_map.get(gen_class_name)
    if gen_cls is None:
        raise ValueError(
            f"Unknown generator_class '{gen_class_name}'. "
            f"Available: {list(gen_cls_map.keys())}"
        )

    # Capture in closure
    _prompt = gen_prompt_text

    class _CustomFactory(gen_cls):
        def __init__(self, **kwargs):
            super().__init__(custom_prompt=_prompt, method_name=name, **kwargs)

    _CustomFactory.__name__ = f"{gen_cls.__name__}({name})"
    _CustomFactory.__doc__ = f"Custom pipeline '{name}'"
    _generators[name] = _CustomFactory

    # Store scorer config in DEFAULT_SCORERS
    if scorer_config is not None:
        from .pipeline import DEFAULT_SCORERS
        DEFAULT_SCORERS[name] = scorer_config


def _read_scorer_prompt(scorer_prompt: Union[str, Path]) -> str:
    """Read scorer_prompt to a string value for storage in config dicts.

    Path objects are read from disk. Strings are returned as-is (could be
    a built-in template name like ``"rlcf"`` or inline prompt text —
    resolved later by ``pipeline.py``).
    """
    if isinstance(scorer_prompt, Path):
        return scorer_prompt.read_text(encoding="utf-8")
    return scorer_prompt


def _scorer_name_to_config(scorer_name: str) -> dict:
    """Map a legacy scorer name string to a config dict.

    Used for backward compatibility when users pass ``scorer="weighted"``
    instead of ``scorer_mode="item", primary_metric="weighted"``.
    """
    mapping = {
        "batch": {"mode": "batch", "primary_metric": "pass"},
        "item": {"mode": "item", "primary_metric": "pass", "capture_reasoning": True},
        "weighted": {"mode": "item", "primary_metric": "weighted"},
        "normalized": {"mode": "item", "primary_metric": "normalized", "use_logprobs": True},
    }
    if scorer_name in mapping:
        return dict(mapping[scorer_name])
    # Unknown name — pass through as-is (will be resolved by registry lookup)
    return {"mode": "batch"}


def save_pipeline_config(name: str, path: Union[str, Path]) -> None:
    """Export a registered pipeline's config to JSON.

    The saved JSON uses flat scorer config keys (``scorer_mode``,
    ``primary_metric``, ``capture_reasoning``, ``scorer_prompt``)
    extracted from the ``DEFAULT_SCORERS`` entry.

    Logprobs are auto-derived from ``primary_metric="normalized"``
    and not stored separately in the config.

    In the output JSON:

    - ``scorer_mode``: ``null`` if no default scorer is configured.
    - ``scorer_prompt``: ``null`` means use the default prompt for the mode.
    - ``primary_metric``: ``null`` defaults to ``"pass"`` when loaded.

    Args:
        name: Name of a registered pipeline.
        path: Path to write the JSON config file.

    Raises:
        KeyError: If the pipeline name is not registered.
    """
    from .generators.instance_level.contrastive import ContrastiveGenerator
    from .pipeline import DEFAULT_SCORERS

    gen_cls = get_generator(name)
    # Instantiate to extract prompt text and class type
    try:
        gen_instance = gen_cls()
    except Exception:
        raise ValueError(
            f"Cannot save config for '{name}': generator requires arguments."
        )

    if isinstance(gen_instance, ContrastiveGenerator):
        gen_class_name = "contrastive"
    else:
        gen_class_name = "direct"

    config = {
        "name": name,
        "generator_class": gen_class_name,
        "generator_prompt": gen_instance.prompt_text,
    }

    # Extract scorer config from DEFAULT_SCORERS
    # use_logprobs is omitted — it's auto-derived from primary_metric="normalized"
    scorer_entry = DEFAULT_SCORERS.get(name)
    if isinstance(scorer_entry, dict):
        config["scorer_mode"] = scorer_entry.get("mode")
        config["scorer_prompt"] = scorer_entry.get("scorer_prompt")
        config["primary_metric"] = scorer_entry.get("primary_metric")
        config["capture_reasoning"] = scorer_entry.get("capture_reasoning", False)
    elif isinstance(scorer_entry, str):
        # Legacy: stored as a name string — convert to flat keys
        legacy_config = _scorer_name_to_config(scorer_entry)
        config["scorer_mode"] = legacy_config.get("mode")
        config["scorer_prompt"] = None
        config["primary_metric"] = legacy_config.get("primary_metric")
        config["capture_reasoning"] = legacy_config.get("capture_reasoning", False)
    else:
        # No scorer configured
        config["scorer_mode"] = None
        config["scorer_prompt"] = None
        config["primary_metric"] = None
        config["capture_reasoning"] = False

    path = Path(path)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_pipeline_config(path: Union[str, Path], force: bool = False) -> str:
    """Load and register a pipeline from a JSON config file.

    Supports both the new format (flat scorer config keys: ``scorer_mode``,
    ``primary_metric``, etc.) and the old format (``scorer`` name string +
    ``scorer_prompt`` text).

    Args:
        path: Path to the JSON config file.
        force: If True, allow overriding built-in pipelines.

    Returns:
        The pipeline name (for use with ``pipeline(name)``).
    """
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))

    name = config["name"]

    # Detect format: new style has "scorer_mode", old style has "scorer"
    if "scorer_mode" in config:
        # New format — flat scorer config keys
        # Note: use_logprobs is silently ignored if present in old JSON files
        # (it's auto-derived from primary_metric="normalized")
        register_custom_pipeline(
            name=name,
            generator_prompt=config.get("generator_prompt"),
            generator_class=config.get("generator_class", "direct"),
            scorer_mode=config.get("scorer_mode"),
            scorer_prompt=config.get("scorer_prompt"),
            primary_metric=config.get("primary_metric"),
            capture_reasoning=config.get("capture_reasoning"),
            force=force,
        )
    else:
        # Old format — backward compatibility
        register_custom_pipeline(
            name=name,
            generator_prompt=config.get("generator_prompt"),
            generator_class=config.get("generator_class", "direct"),
            scorer=config.get("scorer"),
            scorer_prompt=config.get("scorer_prompt"),
            force=force,
        )
    return name


def _auto_register():
    """Register all built-in components on import."""
    from .generators.instance_level.direct import DirectGenerator
    from .generators.instance_level.contrastive import ContrastiveGenerator
    from .generators.instance_level.pipeline_presets import PIPELINE_PRESETS
    from .generators.corpus_level import (
        InductiveGenerator, DeductiveGenerator, InteractiveGenerator,
    )
    from .scorers import BatchScorer, WeightedScorer, NormalizedScorer, ItemScorer
    from .refiners import Deduplicator, Tagger, UnitTester, Selector

    _generator_classes = {
        "DirectGenerator": DirectGenerator,
        "ContrastiveGenerator": ContrastiveGenerator,
    }

    for name, preset in PIPELINE_PRESETS.items():
        gen_cls = _generator_classes[preset["generator_class"]]
        _generators[name] = _make_factory(gen_cls, name, preset)

    _generators.update({
        "feedback": InductiveGenerator,
        "checkeval": DeductiveGenerator,
        "interacteval": InteractiveGenerator,
    })

    # Register deprecated factory functions — they emit DeprecationWarning
    # when called, but the registry needs callables that return scorer instances
    _scorers.update({
        "batch": BatchScorer,
        "item": ItemScorer,
        "weighted": WeightedScorer,
        "normalized": NormalizedScorer,
    })

    _refiners.update({
        "deduplicator": Deduplicator,
        "tagger": Tagger,
        "unit_tester": UnitTester,
        "selector": Selector,
    })

    # Track built-in names for override protection
    _builtin_generators.update(_generators.keys())


# Auto-register on module import
_auto_register()
