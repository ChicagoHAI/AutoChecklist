"""Checklist scorers."""

import warnings

from .base import ChecklistScorer


def BatchScorer(**kwargs):
    """Deprecated: use ``ChecklistScorer(mode='batch')``."""
    warnings.warn(
        "BatchScorer is deprecated, use ChecklistScorer(mode='batch')",
        DeprecationWarning,
        stacklevel=2,
    )
    return ChecklistScorer(mode="batch", **kwargs)


def ItemScorer(**kwargs):
    """Deprecated: use ``ChecklistScorer(mode='item', capture_reasoning=True)``."""
    warnings.warn(
        "ItemScorer is deprecated, use ChecklistScorer(mode='item', capture_reasoning=True)",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.setdefault("capture_reasoning", True)
    return ChecklistScorer(mode="item", **kwargs)


def WeightedScorer(**kwargs):
    """Deprecated: use ``ChecklistScorer(mode='item', primary_metric='weighted')``."""
    warnings.warn(
        "WeightedScorer is deprecated, use ChecklistScorer(mode='item', primary_metric='weighted')",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.setdefault("primary_metric", "weighted")
    return ChecklistScorer(mode="item", **kwargs)


def NormalizedScorer(**kwargs):
    """Deprecated: use ``ChecklistScorer(mode='item', use_logprobs=True, primary_metric='normalized')``."""
    warnings.warn(
        "NormalizedScorer is deprecated, use ChecklistScorer(mode='item', use_logprobs=True, primary_metric='normalized')",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.setdefault("primary_metric", "normalized")
    kwargs.setdefault("use_logprobs", True)
    return ChecklistScorer(mode="item", **kwargs)


__all__ = [
    "ChecklistScorer",
    "BatchScorer",
    "WeightedScorer",
    "NormalizedScorer",
    "ItemScorer",
]
