# autochecklist Library

A library for checklist generation and scoring methods for LLM evaluation. A checklist is a list of yes/no questions used to evaluate LLM outputs.

The core API is **5 generator abstractions**, each representing a different reasoning approach to producing evaluation criteria. Pipeline presets are named configurations that configure these generators with specific prompt templates and default scorers.

## Core Abstraction Hierarchy

```
ChecklistGenerator (ABC)
├── InstanceChecklistGenerator  → One checklist per input
│   ├── DirectGenerator         → Prompt → checklist (direct inference)
│   └── ContrastiveGenerator    → Candidates → checklist (counterfactual reasoning)
└── CorpusChecklistGenerator    → One checklist per dataset
    ├── InductiveGenerator      → Observations → criteria (inductive, bottom-up)
    ├── DeductiveGenerator      → Dimensions → criteria (deductive, top-down)
    └── InteractiveGenerator     → Eval sessions → criteria (protocol analysis)

ChecklistScorer               → Single configurable scorer (structured JSON output)
├── mode="batch"      → Evaluates ALL items in one LLM call (1-based Q-indexing)
│   └── capture_reasoning=True  → Per-item reasoning in batch mode
├── mode="item"       → Evaluates ONE item per LLM call
│   ├── capture_reasoning=True  → Per-item reasoning (TICK-style)
│   ├── primary_metric="weighted" → Uses item weights (0-100)
│   └── primary_metric="normalized" → Confidence via logprobs (auto-enabled)
├── scorer_prompt     → Pipeline presets set paper-specific prompts (rlcf, rocketeval)
├── Score.primary_score → Aliases primary_metric (pass_rate/weighted_score/normalized_score)
└── Deprecated: BatchScorer, ItemScorer, WeightedScorer, NormalizedScorer (factory functions)

ChecklistRefiner (ABC)  → Pipeline stages for corpus-level refinement
├── Deduplicator  → Merge similar items via embeddings
├── Tagger        → Filter by applicability/specificity
├── UnitTester    → Validate enforceability
└── Selector      → Beam search for diversity

ChecklistPipeline     → End-to-end workflow (Generator → Refiners → Scorer)
├── pipeline()        → Factory: resolves generator + auto-attaches default scorer
├── from_preset=      → Same as pipeline(): resolves generator + default scorer
├── generator=        → Resolves generator only (no auto scorer)
├── PipelineResult    → Single evaluation result
└── BatchResult       → Batch evaluation with metrics

Registry              → Component discovery and instantiation
├── list_generators() → Discover available generators
├── get_generator()   → Get class by name (returns factory subclass)
└── register_*()      → Register custom components
```

## Pipeline Presets & Scorer Pairing

Instance-level presets are defined in `generators/instance_level/pipeline_presets.py`:

| Preset | Generator | Default Scorer Config | Primary Metric |
|--------|-----------|----------------------|----------------|
| `tick` | DirectGenerator | `mode="batch"` | pass_rate |
| `rocketeval` | DirectGenerator | `mode="item", use_logprobs=true, scorer_prompt="rocketeval"` | normalized_score |
| `rlcf_direct` | DirectGenerator | `mode="item", scorer_prompt="rlcf"` | weighted_score |
| `rlcf_candidate` | ContrastiveGenerator | `mode="item", scorer_prompt="rlcf"` | weighted_score |
| `rlcf_candidates_only` | ContrastiveGenerator | `mode="item", scorer_prompt="rlcf"` | weighted_score |
| `openrubrics_pairwise` | ContrastiveGenerator | `mode="batch"` | pass_rate |
| `openrubrics_listwise` | ContrastiveGenerator | `mode="batch"` | pass_rate |

Corpus-level generators (feedback, checkeval, interacteval) are not in presets — they remain as separate classes. Default scorer: `{mode: "batch", primary_metric: "pass"}`.

## Key Data Models (`models.py`)

- **ChecklistItem**: Single yes/no question with optional weight/category (frozen/immutable)
- **Checklist**: Collection of items with metadata (source_method, generation_level, `input`). Has `by_category()` to split into sub-checklists, `save()`/`load()` for JSON persistence
- **Score**: Scoring result with item_scores list, aggregate scores (pass_rate, weighted_score, normalized_score — all always computed), `primary_metric` field, and `primary_score` property that aliases the primary metric
- **GroupedScore**: Per-category scoring result with `per_group_pass_rates`, macro/micro `pass_rate`, and `flatten()`
- **Input models**: `DeductiveInput`, `FeedbackInput`, `InteractiveInput` for corpus-level generators
- **Field naming**: `input` (instruction/query), `target` (response to evaluate) throughout

## LLM Providers

Multi-provider support via `LLMClient` Protocol in `providers/base.py`:

- **`LLMHTTPClient`** (`providers/http_client.py`): Provider-agnostic HTTP client for OpenRouter, OpenAI, vLLM server
- **`VLLMOfflineClient`** (`providers/vllm_offline.py`): Direct Python inference via `vllm.LLM` (no server needed)
- **`OpenRouterClient`** (`providers/openrouter.py`): Backward-compat alias of `LLMHTTPClient`
- **`get_client()`** (`providers/factory.py`): Factory that auto-selects client based on provider + base_url

All generators/scorers/refiners accept `provider`, `base_url`, `client`, and `api_format` params, threaded through `_get_or_create_client()` in the base classes. Embeddings use direct OpenAI API (requires `OPENAI_API_KEY` for corpus-level refiners).

`ContrastiveGenerator` additionally accepts `candidate_provider`, `candidate_base_url`, `candidate_api_key`, and `candidate_api_format` for using a separate LLM provider for candidate generation (via `_get_candidate_client()`). These are passed through `pipeline()` via `generator_kwargs` (e.g., `generator_kwargs={"candidate_provider": "vllm"}`).

## Structured Output

All instance-level generators and all scorers use `response_format` (JSON schema via Pydantic) for structured LLM output. The base `_call_model()` method handles fallback automatically:
- **Primary**: `response_format` with JSON schema (enforced by provider)
- **Fallback**: Schema-in-prompt (via format files in `prompts/formats/`) + JSON extraction (via `utils/parsing.py:extract_json()`)

Response schemas are defined in `models.py`: `ChecklistResponse`, `WeightedChecklistResponse`, `CategorizedChecklistResponse`, `BatchScoringResponse`, `BatchScoringResponseReasoned`, `ItemScoringResponse`, `ItemScoringResponseReasoned`. `WeightedScoringResponse` is a deprecated alias for `ItemScoringResponse`. Use `to_response_format(ModelClass, "name")` to convert to OpenAI-compatible dict.

Provider support:
- **OpenRouter**: `response_format` passthrough (model-dependent)
- **OpenAI Chat Completions**: `response_format` with `json_schema` type
- **OpenAI Responses API**: Translated to `text.format` (flattened schema)
- **vLLM offline**: `GuidedDecodingParams(json=schema)` for token-level enforcement

## Prompt System

Generator templates in `prompts/generators/{method}/*.md`, scorer templates in `prompts/scoring/*.md` (batch, item, rlcf, rocketeval), format instructions in `prompts/formats/*.md` (batch_scoring, batch_scoring_reasoned, item_scoring, item_scoring_reasoned). All loaded via `load_template(category, template_name)` and `load_format(format_name)`. Use `PromptTemplate` class for placeholder validation and formatting.

Placeholders use `{input}` and `{target}` (not instruction/response).

## Adding New Methods

For instance-level methods, add a new pipeline preset in `generators/instance_level/pipeline_presets.py`:
1. Add prompt template in `prompts/generators/{method}/`
2. Add entry to `PIPELINE_PRESETS` dict
3. Register via `_make_factory()` in `registry.py`

For corpus-level methods:
1. Create generator class in `generators/corpus_level/`
2. Inherit from `CorpusChecklistGenerator`
3. Export from `__init__.py` files
4. Register directly in `registry.py`

For custom prompts (no code changes needed):
```python
from autochecklist import register_custom_generator
register_custom_generator("my_eval", "prompts/my_eval.md")
```

## Pipeline Usage

Three ways to construct a pipeline:

| Construction | Generator | Default Scorer | Use Case |
|---|---|---|---|
| `pipeline("tick")` | Resolved | Auto-attached | Quick start with sensible defaults |
| `ChecklistPipeline(from_preset="tick")` | Resolved | Auto-attached | Same as `pipeline()` |
| `ChecklistPipeline(generator="tick")` | Resolved | None | Explicit composition — bring your own scorer |
| `ChecklistPipeline(generator=my_gen)` | Instance | None | Pre-configured components |

`pipeline()` is equivalent to `ChecklistPipeline(from_preset=...)`. All three component types — `generator`, `scorer`, and `refiners` — accept name strings or pre-configured instances.

```python
# Quick evaluation (OpenRouter default)
from autochecklist import pipeline
pipe = pipeline("tick", generator_model="openai/gpt-4o-mini", scorer_model="openai/gpt-4o-mini")
result = pipe(input="Write a haiku", target="Leaves fall...")
print(result.pass_rate)

# Split models (different model for generation vs scoring)
pipe = pipeline("tick", generator_model="openai/gpt-4o", scorer_model="openai/gpt-4o-mini")

# OpenAI direct
pipe = pipeline("tick", provider="openai", generator_model="gpt-4o-mini")

# vLLM server mode
pipe = pipeline("tick", provider="vllm", base_url="http://gpu:8000/v1", generator_model="meta-llama/Llama-3-8B")

# vLLM offline (no server, direct Python inference)
from autochecklist import VLLMOfflineClient
client = VLLMOfflineClient(model="google/gemma-3-1b-it")
pipe = pipeline("tick", client=client)

# Responses API (OpenAI)
pipe = pipeline("tick", provider="openai", generator_model="gpt-4o-mini", api_format="responses")

# Separate candidate provider (ContrastiveGenerator pipelines only)
pipe = pipeline("rlcf_candidate", generator_model="openai/gpt-4o-mini",
                generator_kwargs={"candidate_provider": "vllm", "candidate_base_url": "http://gpu:8000/v1"})

# Explicit composition — resolve generator by name, bring your own scorer
from autochecklist import ChecklistPipeline
pipe = ChecklistPipeline(generator="tick", scorer="item",
                         generator_model="openai/gpt-4o-mini", scorer_model="openai/gpt-4o-mini")

# Pre-configured component instances (full control)
from autochecklist import DirectGenerator, ChecklistScorer
gen = DirectGenerator(method_name="tick", model="gpt-4o", provider="openai")
scorer = ChecklistScorer(mode="batch", model="gpt-4o-mini", provider="openrouter")
pipe = ChecklistPipeline(generator=gen, scorer=scorer)

# Generate only (no scoring)
checklists = pipe.generate_batch(inputs=["Write a haiku", "Write a poem"])

# Resumable batch (writes JSONL, resumes on re-run)
result = pipe.run_batch(data, output_path="results.jsonl")

# Batch evaluation
data = [{"input": "...", "target": "..."}]
result = pipe.run_batch(data, show_progress=True)
print(result.macro_pass_rate)

# Component discovery
from autochecklist import list_generators, get_generator
print(list_generators())  # ['tick', 'rocketeval', 'rlcf_direct', ...]
gen_cls = get_generator("tick")
```
