"""autochecklist: A library of checklist generation and scoring methods for LLM evaluation."""

from .models import (
    Checklist,
    ChecklistItem,
    Score,
    ItemScore,
    ChecklistItemAnswer,
    ConfidenceLevel,
    GroupedScore,
    DeductiveInput,
    FeedbackInput,
    InteractiveInput,
)
from .config import configure, get_config
from .generators.instance_level.direct import DirectGenerator
from .generators.instance_level.contrastive import ContrastiveGenerator
from .generators.corpus_level import InductiveGenerator, DeductiveGenerator, AugmentationMode, InteractiveGenerator
from .generators.base import (
    ChecklistGenerator,
    InstanceChecklistGenerator,
    CorpusChecklistGenerator,
)
from .scorers import BatchScorer, ChecklistScorer, WeightedScorer, NormalizedScorer, ItemScorer
from .refiners import ChecklistRefiner, Deduplicator, Tagger, UnitTester, Selector
from .pipeline import ChecklistPipeline, PipelineResult, BatchResult, pipeline
from .providers import LLMClient, LLMHTTPClient, VLLMOfflineClient, get_client
from .registry import (
    list_generators,
    list_scorers,
    list_refiners,
    get_generator,
    get_scorer,
    get_refiner,
    register_generator,
    register_scorer,
    register_refiner,
    register_custom_generator,
    register_custom_scorer,
    register_custom_pipeline,
    save_pipeline_config,
    load_pipeline_config,
    list_generators_with_info,
    list_scorers_with_info,
    list_refiners_with_info,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "Checklist",
    "ChecklistItem",
    "Score",
    "ItemScore",
    "ChecklistItemAnswer",
    "ConfidenceLevel",
    "GroupedScore",
    "DeductiveInput",
    "FeedbackInput",
    "InteractiveInput",
    # Config
    "configure",
    "get_config",
    # Generators - Instance level
    "DirectGenerator",
    "ContrastiveGenerator",
    # Generators - Corpus level
    "InductiveGenerator",
    "DeductiveGenerator",
    "AugmentationMode",
    "InteractiveGenerator",
    # Generators - Base
    "ChecklistGenerator",
    "InstanceChecklistGenerator",
    "CorpusChecklistGenerator",
    # Scorers
    "BatchScorer",
    "ChecklistScorer",
    "WeightedScorer",
    "NormalizedScorer",
    "ItemScorer",
    # Refiners
    "ChecklistRefiner",
    "Deduplicator",
    "Tagger",
    "UnitTester",
    "Selector",
    # Pipeline
    "ChecklistPipeline",
    "PipelineResult",
    "BatchResult",
    "pipeline",
    # Providers
    "LLMClient",
    "LLMHTTPClient",
    "VLLMOfflineClient",
    "get_client",
    # Registry
    "list_generators",
    "list_scorers",
    "list_refiners",
    "get_generator",
    "get_scorer",
    "get_refiner",
    "register_generator",
    "register_scorer",
    "register_refiner",
    "register_custom_generator",
    "register_custom_scorer",
    "register_custom_pipeline",
    "save_pipeline_config",
    "load_pipeline_config",
    "list_generators_with_info",
    "list_scorers_with_info",
    "list_refiners_with_info",
]
