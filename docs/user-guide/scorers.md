# Scorers

## What is a Scorer?

A **scorer** is the final stage of the evaluation pipeline. It takes a target response and a [checklist](generators.md) and evaluates the response by answering each yes/no question. The result is a `Score` object with aggregate metrics and per-item answers.

All scoring modes use **structured JSON output** with automatic fallback — if the LLM provider supports JSON schema enforcement, it's used; otherwise, the schema is included in the prompt and the response is parsed.

## The Score Object

Every scorer returns a `Score` with these key properties:

| Property | Type | Description |
|----------|------|-------------|
| `primary_score` | `float` | **Primary metric** — aliases whichever metric the pipeline designated |
| `pass_rate` | `float` | Proportion of YES answers |
| `weighted_score` | `float` | Importance-weighted aggregation (always computed) |
| `normalized_score` | `float` | Logprob confidence average, or pass_rate if no logprobs |
| `item_scores` | `list[ItemScore]` | Per-question answers with optional reasoning/confidence |
| `scaled_score_1_5` | `float` | Convenience mapping: `pass_rate * 4 + 1` (range 1.0–5.0) |
| `primary_metric` | `str` | Which metric `primary_score` aliases: `"pass"`, `"weighted"`, or `"normalized"` |

All three aggregate metrics (`pass_rate`, `weighted_score`, `normalized_score`) are **always computed**. The `score` property returns whichever one the pipeline considers primary.

Each `ItemScore` contains:

- `answer`: `"yes"` or `"no"`
- `reasoning`: per-item explanation (when `capture_reasoning=True`)
- `confidence`: logprob-derived probability (when `use_logprobs=True`)
- `confidence_level`: categorical level like `yes_90`, `no_30`, `unsure` (when logprobs available)

## ChecklistScorer

There is a single, configurable `ChecklistScorer` class. The three key parameters are:

| Parameter | Values | Effect |
|-----------|--------|--------|
| `mode` | `"batch"` or `"item"` | Batch = 1 LLM call; Item = N calls (one per question) |
| `primary_metric` | `"pass"`, `"weighted"`, `"normalized"` | Which metric `Score.primary_score` aliases. `"normalized"` auto-enables logprobs. |
| `capture_reasoning` | `bool` | Item mode: include per-item reasoning |

### Batch Mode

Evaluates **all checklist items in a single LLM call**. Questions are formatted as `Q1: ...`, `Q2: ...` (1-based indexing), and the LLM returns YES/NO for each.

**Tradeoffs:** Fastest and cheapest (1 API call), but may lose accuracy on very long checklists.

Default for: `tick`, `feedback`, `checkeval`, `interacteval`.

```python
from autochecklist import ChecklistScorer

scorer = ChecklistScorer(mode="batch", model="openai/gpt-4o-mini")
score = scorer.score(checklist, target="...")
print(f"Pass rate: {score.pass_rate:.0%}")
```

**With reasoning** (`capture_reasoning=True`): Each answer includes a reasoning explanation. Useful for debugging checklist quality.

```python
scorer = ChecklistScorer(mode="batch", capture_reasoning=True, model="openai/gpt-4o-mini")
score = scorer.score(checklist, target="...")
for item_score in score.item_scores:
    print(f"{item_score.answer}: {item_score.reasoning}")
```

### Item Mode

Evaluates **one item per LLM call**. Supports three configurations:

**With reasoning** (`capture_reasoning=True`): Each call returns YES/NO plus a reasoning explanation. Most faithful to the TICK paper methodology.

```python
scorer = ChecklistScorer(mode="item", capture_reasoning=True, model="openai/gpt-4o-mini")
score = scorer.score(checklist, target="...")
for item_score in score.item_scores:
    print(f"{item_score.answer}: {item_score.reasoning}")
```

**Weighted metric** (`primary_metric="weighted"`): Uses item weights (0-100) for importance-weighted scoring. Designed for RLCF checklists.

$$\text{weighted\_score} = \frac{\sum_{i} w_i \times s_i}{\sum_{i} w_i}$$

Default for: `rlcf_direct`, `rlcf_candidate`, `rlcf_candidates_only`.

```python
scorer = ChecklistScorer(mode="item", primary_metric="weighted", model="openai/gpt-4o-mini")
score = scorer.score(checklist, target="...")
print(f"Weighted: {score.weighted_score:.2f}, Primary: {score.primary_score:.2f}")
```

**With logprobs** (`use_logprobs=True`): Captures logprobs for YES/NO tokens to produce confidence-calibrated scores.

$$\text{confidence} = \frac{P(\text{Yes})}{P(\text{Yes}) + P(\text{No})}$$

| Confidence Range | Answer | Level |
|-----------------|--------|-------|
| < 0.2 | NO | `no_10` |
| 0.2 – 0.4 | NO | `no_30` |
| 0.4 – 0.6 | NO | `unsure` |
| 0.6 – 0.8 | YES | `yes_70` |
| > 0.8 | YES | `yes_90` |

Falls back to binary YES/NO when the model doesn't support logprobs. Default for: `rocketeval`.

```python
scorer = ChecklistScorer(mode="item", use_logprobs=True, primary_metric="normalized", model="openai/gpt-4o-mini")
score = scorer.score(checklist, target="...")
print(f"Normalized: {score.normalized_score:.2f}")
```

## Choosing a Configuration

| Config | LLM Calls | Primary Metric | Per-Item Reasoning | Best For |
|--------|-----------|----------------|-------------------|----------|
| `mode="batch"` | 1 | `pass_rate` | No | Large checklists, cost-sensitive |
| `mode="batch", capture_reasoning=True` | 1 | `pass_rate` | Yes | Batch with explanations |
| `mode="item"` | N | `pass_rate` | No | Per-item evaluation |
| `mode="item", capture_reasoning=True` | N | `pass_rate` | Yes | Debugging, interpretability |
| `mode="item", primary_metric="weighted"` | N | `weighted_score` | No | RLCF weighted criteria |
| `mode="item", primary_metric="normalized"` | N | `normalized_score` | No | Confidence-aware (RocketEval) |

## Overriding Defaults

Each [built-in pipeline](supported-pipelines.md) has a default scorer, but you can override it:

```python
from autochecklist import pipeline, ChecklistScorer

# Use item mode with TICK for per-item reasoning
pipe = pipeline("tick", generator_model="openai/gpt-4o-mini", scorer="item")

# Use batch mode with RLCF for speed
pipe = pipeline("rlcf_direct", generator_model="openai/gpt-4o-mini", scorer="batch")

# Use a fully custom scorer
scorer = ChecklistScorer(mode="item", capture_reasoning=True, model="openai/gpt-4o-mini")
pipe = pipeline("tick", generator_model="openai/gpt-4o-mini", scorer=scorer)
```

All scorers also accept a `custom_prompt` parameter. For custom scorer prompts, use `register_custom_scorer()` with optional config kwargs (`mode`, `primary_metric`, `capture_reasoning`). See [Custom Prompts](custom-prompts.md).

## Deprecated Class Names

The old scorer class names (`BatchScorer`, `ItemScorer`, `WeightedScorer`, `NormalizedScorer`) still work as factory functions that emit `DeprecationWarning`. They create `ChecklistScorer` instances with the appropriate config:

| Old Name | Equivalent |
|----------|-----------|
| `BatchScorer(...)` | `ChecklistScorer(mode="batch", ...)` |
| `ItemScorer(...)` | `ChecklistScorer(mode="item", capture_reasoning=True, ...)` |
| `WeightedScorer(...)` | `ChecklistScorer(mode="item", primary_metric="weighted", ...)` |
| `NormalizedScorer(...)` | `ChecklistScorer(mode="item", use_logprobs=True, primary_metric="normalized", ...)` |

## Scorer Prompts

The scorer uses different prompt templates depending on the pipeline:

| Prompt | File | Used By |
|--------|------|---------|
| Default batch | `prompts/scoring/batch.md` | `tick`, corpus-level presets |
| Default item | `prompts/scoring/item.md` | Default for item mode |
| RLCF | `prompts/scoring/rlcf.md` | `rlcf_direct`, `rlcf_candidate`, `rlcf_candidates_only` |
| RocketEval | `prompts/scoring/rocketeval.md` | `rocketeval` (includes `{history}` placeholder) |

Pipeline presets automatically select their scorer prompt via the `scorer_prompt` key. You can override with `custom_prompt`:

```python
scorer = ChecklistScorer(mode="item", custom_prompt="Your custom prompt with {input}, {target}, {question}")
```

## Response Schemas

| Schema | Fields | Used When |
|--------|--------|-----------|
| `ItemScoringResponse` | `{answer}` | Item mode, no reasoning |
| `ItemScoringResponseReasoned` | `{answer, reasoning}` | Item mode + `capture_reasoning` |
| `BatchScoringResponse` | `{answers: [{question_index, answer}]}` | Batch mode |
| `BatchScoringResponseReasoned` | `{answers: [{question_index, answer, reasoning}]}` | Batch mode + `capture_reasoning` |

The old `WeightedScoringResponse` name is a deprecated alias for `ItemScoringResponse`.
