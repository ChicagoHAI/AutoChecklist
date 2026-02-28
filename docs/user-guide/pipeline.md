# Pipeline & Composability

## The Composability Model

autochecklist is built around a three-stage pipeline where each component is independently configurable and replaceable:

```
Generator → Refiners → Scorer
```

Like UNIX pipes, each stage has a clear interface: generators produce `Checklist` objects, refiners take a `Checklist` in and return a refined `Checklist` out, and scorers take a `Checklist` plus a target response and return a `Score`. You can swap any component without affecting the others.

[Built-in pipelines](supported-pipelines.md) provide sensible defaults — each pipeline pairs a generator with its recommended scorer — but you can override any component.

## `pipeline()` vs `ChecklistPipeline`

Both create the same pipeline object. The difference is whether the default scorer is auto-attached:

| Construction | Generator | Default Scorer | Use Case |
|---|---|---|---|
| `pipeline("tick")` | Resolved | Auto-attached | Quick start with sensible defaults |
| `ChecklistPipeline(from_preset="tick")` | Resolved | Auto-attached | Same as `pipeline()` |
| `ChecklistPipeline(generator="tick")` | Resolved | None | Explicit composition — bring your own scorer |
| `ChecklistPipeline(generator=my_gen)` | Instance | None | Pre-configured components |

`pipeline()` is equivalent to `ChecklistPipeline(from_preset=...)`. All three component types — `generator`, `scorer`, and `refiners` — accept name strings or pre-configured instances.

## The `pipeline()` Factory

The `pipeline()` function is the recommended entry point. Pass a pipeline name and it auto-selects the generator class, prompt templates, and default scorer:

```python
from autochecklist import pipeline

pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `task` | Pipeline name: `"tick"`, `"rocketeval"`, `"rlcf_direct"`, `"rlcf_candidate"`, `"rlcf_candidates_only"`, `"feedback"`, `"checkeval"`, `"interacteval"` |
| `generator_model` | LLM model for checklist generation (e.g., `"openai/gpt-5-mini"`) |
| `scorer_model` | LLM model for scoring (optional; inherits provider defaults if omitted) |
| `scorer` | Override default scorer: `"batch"`, `"item"`, `"weighted"`, `"normalized"` |
| `refiners` | List of refiner names: `["deduplicator", "tagger", "selector"]` |
| `provider` | LLM provider: `"openrouter"` (default), `"openai"`, `"vllm"` |
| `base_url` | Custom API endpoint |
| `client` | Pre-configured LLM client (e.g., `VLLMOfflineClient`) |
| `api_format` | `"chat"` (default) or `"responses"` (OpenAI Responses API) |
| `generator_kwargs` | Extra keyword arguments passed to the generator (e.g., `candidate_provider`, `candidate_base_url`) |

## Single Evaluation

Call the pipeline directly with `input` and `target` to generate a checklist and score in one step:

```python
result = pipe(input="Write a haiku about autumn.", target="Leaves fall gently...")

print(result.pass_rate)        # 0.83
print(result.checklist)        # the generated Checklist
print(result.score)            # the full Score object
```

The returned `PipelineResult` gives access to:

- `result.checklist` — the generated (and refined, if applicable) checklist
- `result.score` — the full `Score` object
- `result.score.primary_score` — the primary metric value (aliases `pass_rate`, `weighted_score`, or `normalized_score` depending on pipeline)
- `result.pass_rate`, `result.weighted_score`, `result.normalized_score` — convenience accessors

## Generate Only

To inspect or save the checklist before scoring:

```python
checklist = pipe.generate(input="Write a haiku about autumn.")
print(checklist.to_text())

# Score later with the same or different pipeline
score = pipe.score(checklist, target="Leaves fall gently...", input="Write a haiku about autumn.")
```

## Batch Evaluation

### Instance-Level: Per-Input Generation + Scoring

For instance-level generators, `run_batch` generates a unique checklist for each input and scores it:

```python
data = [
    {"input": "Write a haiku", "target": "Leaves fall..."},
    {"input": "Write a limerick", "target": "There once was..."},
]
result = pipe.run_batch(data, show_progress=True)
print(f"Mean score: {result.mean_score:.0%}")
print(f"Macro pass rate: {result.macro_pass_rate:.0%}")
print(f"Micro pass rate (DRFR): {result.micro_pass_rate:.0%}")
```

`mean_score` averages `Score.primary_score` across all examples — it respects the pipeline's `primary_metric` setting. For pass-rate pipelines it equals `macro_pass_rate`; for weighted pipelines it averages weighted scores; for normalized pipelines it averages normalized scores.

`micro_pass_rate` computes the **Decomposed Requirements Following Ratio (DRFR)** from [InFoBench](https://arxiv.org/abs/2401.03601) — the total number of YES answers divided by the total number of applicable items across all examples. This differs from `macro_pass_rate` (mean of per-example pass rates) by giving equal weight to each checklist item rather than each example.

### Corpus-Level: Shared Checklist Scoring

For corpus-level generators, generate one checklist first, then score all responses against it:

```python
pipe = pipeline("feedback", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
checklist = pipe.generate(observations=["Too verbose", "Missing examples", ...])

result = pipe.run_batch(data, checklist=checklist, show_progress=True)
```

### BatchResult

The `BatchResult` object provides:

```python
result.mean_score        # mean of Score.primary_score (respects primary_metric)
result.macro_pass_rate   # macro-average: mean of per-example pass rates
result.micro_pass_rate   # micro-average (DRFR): total_yes / total_applicable across all items
result.scores            # list of individual Score objects

# Export
df = result.to_dataframe()
result.to_jsonl("results.jsonl")
```

## Adding Refiners

Pass refiner names to add refinement stages between generation and scoring:

```python
pipe = pipeline("feedback", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini",
                refiners=["deduplicator", "tagger", "selector"])
```

See [Refiners](refiners.md) for details on each refiner and recommended chaining order.

## Overriding Components

```python
# Override scorer
pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer="item")

# Use OpenAI directly instead of OpenRouter
pipe = pipeline("tick", provider="openai", generator_model="gpt-5-mini")

# Use vLLM server
pipe = pipeline("tick", provider="vllm", base_url="http://gpu:8000/v1",
                generator_model="meta-llama/Llama-3-8B")

# Separate provider for candidate generation (ContrastiveGenerator only)
pipe = pipeline("rlcf_candidate", generator_model="openai/gpt-5-mini",
                generator_kwargs={"candidate_provider": "vllm",
                                  "candidate_base_url": "http://gpu:8000/v1"})
```

## Explicit Composition

Use `ChecklistPipeline` directly when you want to pick each component yourself. All three accept name strings:

```python
from autochecklist import ChecklistPipeline

pipe = ChecklistPipeline(generator="tick", scorer="item",
                         generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
```

For components that need configuration beyond what a name string provides (e.g., `ContrastiveGenerator` with `candidate_models`), pass a pre-configured instance:

```python
from autochecklist import ChecklistPipeline, ContrastiveGenerator

gen = ContrastiveGenerator(
    method_name="rlcf_candidate",
    model="openai/gpt-5-mini",
    candidate_models=["openai/gpt-5-mini", "anthropic/claude-3-haiku"],
)
pipe = ChecklistPipeline(generator=gen, scorer="weighted", scorer_model="openai/gpt-5-mini")
result = pipe(input="...", target="...", reference="...")
```

You can also use components entirely independently, outside any pipeline:

```python
from autochecklist import DirectGenerator, ChecklistScorer
from autochecklist.refiners import Deduplicator

gen = DirectGenerator(method_name="tick", model="openai/gpt-5-mini")
checklist = gen.generate(input="Write a haiku")
refined = Deduplicator(model="openai/gpt-5-mini").refine(checklist)
score = ChecklistScorer(mode="batch", model="openai/gpt-5-mini").score(refined, target="Leaves fall...")
```

## Generator-Scorer Pairing

Each pipeline has a default scorer chosen to match the original paper's methodology. You can override any pairing.

| Pipeline | Level | Default Scorer | Primary Metric |
|---------|-------|---------------|----------------|
| `tick` | Instance | `batch` | `pass_rate` |
| `rocketeval` | Instance | `normalized` | `normalized_score` |
| `rlcf_direct` | Instance | `weighted` | `weighted_score` |
| `rlcf_candidate` | Instance | `weighted` | `weighted_score` |
| `rlcf_candidates_only` | Instance | `weighted` | `weighted_score` |
| `feedback` | Corpus | `batch` | `pass_rate` |
| `checkeval` | Corpus | `batch` | `pass_rate` |
| `interacteval` | Corpus | `batch` | `pass_rate` |

Override with: `pipeline("tick", scorer="item")`

## Custom Pipeline Registration

Register a custom pipeline (generator + scorer + prompts) as a reusable preset. Once registered, it works just like a built-in pipeline:

```python
from autochecklist import register_custom_pipeline, pipeline

# Register from config values with scorer settings
register_custom_pipeline(
    "my_eval",
    generator_prompt="Generate yes/no questions for:\n\n{input}",
    scorer_mode="item",
    primary_metric="weighted",
)
pipe = pipeline("my_eval", generator_model="openai/gpt-5-mini")

# With logprobs-based scoring (auto-enabled by primary_metric="normalized")
register_custom_pipeline(
    "my_logprob_eval",
    generator_prompt="Generate yes/no questions for:\n\n{input}",
    scorer_mode="item",
    primary_metric="normalized",
)

# Or register from an existing pipeline instance
from autochecklist import ChecklistPipeline, DirectGenerator, ChecklistScorer
pipe = ChecklistPipeline(
    generator=DirectGenerator(custom_prompt="...", model="openai/gpt-5-mini"),
    scorer=ChecklistScorer(mode="item", primary_metric="weighted", model="openai/gpt-5-mini"),
)
register_custom_pipeline("my_eval_v2", pipe)
```

**Scorer config parameters:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `scorer_mode` | `"batch"`, `"item"` | All items in one call vs. one per call |
| `primary_metric` | `"pass"`, `"weighted"`, `"normalized"` | Which metric `Score.primary_score` aliases. `"normalized"` auto-enables logprobs. |
| `capture_reasoning` | `True`/`False` | Include per-item reasoning |
| `scorer_prompt` | text, built-in name (`"rlcf"`, `"rocketeval"`), or Path | Custom scorer prompt. `null`/omitted = use default prompt for the mode. |

### Save / Load Pipeline Configs

Export and share pipeline configs as JSON:

```python
from autochecklist import save_pipeline_config, load_pipeline_config

save_pipeline_config("my_eval", "my_eval.json")

# Later, or on another machine:
name = load_pipeline_config("my_eval.json")
pipe = pipeline(name, generator_model="openai/gpt-5-mini")
```

JSON format:

```json
{
  "name": "my_eval",
  "generator_class": "direct",
  "generator_prompt": "Generate yes/no questions for:\n\n{input}",
  "scorer_mode": "item",
  "scorer_prompt": null,
  "primary_metric": "weighted",
  "capture_reasoning": false
}
```

**Field notes:**
- `scorer_mode`: `null` = no default scorer attached; `"batch"` or `"item"` = attach a scorer.
- `scorer_prompt`: `null` = use the default prompt for the mode. Accepts a built-in name (`"rlcf"`, `"rocketeval"`) or inline prompt text — resolved automatically at pipeline construction time.
- `primary_metric`: `null` = defaults to `"pass"`. Other options: `"weighted"`, `"normalized"`. Setting `"normalized"` auto-enables logprobs for confidence-calibrated scoring.

### Override Protection

Built-in pipelines (`tick`, `rocketeval`, etc.) are protected. Overriding raises `ValueError` unless you pass `force=True`:

```python
register_custom_pipeline("tick", generator_prompt="...", force=True)  # warns but succeeds
```

Custom pipelines can be overridden silently.

## Component Discovery

```python
from autochecklist import list_generators, get_generator

print(list_generators())  # ['tick', 'rocketeval', 'rlcf_direct', ..., 'my_eval']
gen_cls = get_generator("tick")
```
