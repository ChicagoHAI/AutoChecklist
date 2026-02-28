# Supported Pipelines

This page documents each evaluation method implemented in autochecklist: the original paper's methodology, how it was evaluated, and how this library implements it.

!!! note "Implementation Scope"
    These pipelines aim to capture the **core algorithms** of each paper (the generation strategies, scoring formulas, and refinement pipelines) in a composable, provider-agnostic way. Some paper-specific details (supervised score predictors, dataset-specific tuning, exact prompts) are simplified or omitted. Each pipeline section below notes the specific differences.

## Overview

Pipelines are named presets that configure a [generator](generators.md) with a specific prompt template and default scorer. Each pipeline implements a method from a research paper.

| Method | Level | Pipeline Name | Generator Class | Approach | Primary Metric |
|--------|-------|-------------|-----------------|----------|----------------|
| TICK | Instance | `tick` | `DirectGenerator` | Direct inference | `pass_rate` |
| RocketEval | Instance | `rocketeval` | `DirectGenerator` | Direct inference | `normalized_score` |
| RLCF Direct | Instance | `rlcf_direct` | `DirectGenerator` | Direct inference | `weighted_score` |
| RLCF Candidate | Instance | `rlcf_candidate` | `ContrastiveGenerator` | Counterfactual reasoning | `weighted_score` |
| RLCF Candidates Only | Instance | `rlcf_candidates_only` | `ContrastiveGenerator` | Counterfactual reasoning | `weighted_score` |
| Feedback | Corpus | `feedback` | `InductiveGenerator` | Inductive (bottom-up) | `pass_rate` |
| CheckEval | Corpus | `checkeval` | `DeductiveGenerator` | Deductive (top-down) | `pass_rate` |
| InteractEval | Corpus | `interacteval` | `InteractiveGenerator` | Protocol analysis | `pass_rate` |

---

## Instance-Level Methods

### TICK

**Paper:** [arXiv:2410.03608](https://arxiv.org/abs/2410.03608)

**Original methodology:** TICK generates checklists from the input instruction alone, using few-shot prompting. The LLM reads an instruction and produces 2–8 yes/no questions that capture the task requirements. Each question is scored individually with chain-of-thought reasoning. Evaluated on InFoBench. The key insight is that task-specific checklists provide more fine-grained evaluation than single-score rubrics.

**Our implementation:**

| Setting | Value |
|---------|-------|
| Pipeline | `tick` |
| Generator | `DirectGenerator` with `tick/generate.md` template |
| Input required | `input` only |
| Temperature | 0.7 |
| Max items | 8 (min 2) |
| Response schema | `ChecklistResponse` (unweighted) |
| Primary metric | `pass_rate` |

```python
from autochecklist import pipeline

pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="Write a haiku about autumn.", target="Leaves fall gently...")
print(result.pass_rate)
```

!!! tip "Paper-Faithful Scoring"
    The original paper scores each item individually with reasoning. To match this behavior, override the scorer:
    ```python
    pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer="item")
    ```

**Differences from paper:**

| Paper feature | Our implementation |
|---------------|-------------------|
| Per-item scoring with reasoning | Default is batch mode for efficiency; use `scorer="item"` for paper-faithful behavior |
| Few-shot prompt with hand-crafted examples | Structured JSON output (`ChecklistResponse` schema) replaces few-shot format guidance |
| Dataset-level DRFR aggregation metric | `BatchResult.micro_pass_rate` implements DRFR (micro-averaged pass rate) for dataset-level aggregation |

---

### RocketEval

**Paper:** [arXiv:2503.05142](https://arxiv.org/abs/2503.05142)

**Original methodology:** RocketEval generates checklists from the input instruction, a reference response, and optionally conversation history. The key innovation is in scoring: instead of binary yes/no, it uses **logprob confidence calibration** — computing `P(Yes) / (P(Yes) + P(No))` from the model's token probabilities and mapping to 5 confidence levels. This produces finer-grained scores than binary answers. Evaluated on MT-Bench, AlpacaEval, WildBench, and Arena-Hard.

**Our implementation:**

| Setting | Value |
|---------|-------|
| Pipeline | `rocketeval` |
| Generator | `DirectGenerator` with `rocketeval/generate.md` template |
| Input required | `input` + `reference` |
| Temperature | 0.7 |
| Max items | 10 (min 1) |
| Response schema | `ChecklistResponse` (unweighted) |
| Primary metric | `normalized_score` (logprobs with structured fallback) |

```python
pipe = pipeline("rocketeval", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="Explain photosynthesis.", target="...", reference="...")
print(result.normalized_score)
```

**Differences from paper:**

| Paper feature | Our implementation |
|---------------|-------------------|
| Supervised score predictor (fitted per-query from annotations) | Not implemented; uses unsupervised arithmetic mean of normalized scores only |
| KL-divergence reweighting (`α_r`) blending supervised + unsupervised scores | Not implemented (requires annotation data) |
| Conversation history as generation context | Supported via template placeholders but not enforced |

---

### RLCF (Reinforcement Learning from Checklist Feedback)

**Paper:** [arXiv:2507.18624](https://arxiv.org/abs/2507.18624)

**Original methodology:** RLCF introduces **weighted checklist items** where each question has an importance weight from 0–100:

- **100** = critical requirement
- **75** = important quality factor
- **50** = good response indicator
- **25** = preference/style
- **< 25** = nice-to-have

The key insight is that not all criteria are equally important — weighting produces more discriminative evaluation signals, particularly useful for RLHF training. Evaluated on WildChat. Three generation modes are defined:

#### RLCF Direct

Prompts the LLM to write a weighted checklist from the input instruction and reference response.

| Setting | Value |
|---------|-------|
| Pipeline | `rlcf_direct` |
| Generator | `DirectGenerator` with `rlcf/direct.md` template |
| Input required | `input` + `reference` |
| Temperature | 0.0 |
| Max items | 7 (min 1) |
| Response schema | `WeightedChecklistResponse` |
| Primary metric | `weighted_score` |

```python
pipe = pipeline("rlcf_direct", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="...", target="...", reference="...")
print(result.weighted_score)
```

#### RLCF Candidate

First generates varied-quality candidate responses, then derives criteria by contrasting candidates against the reference.

| Setting | Value |
|---------|-------|
| Pipeline | `rlcf_candidate` |
| Generator | `ContrastiveGenerator` with `rlcf/candidate_based.md` template |
| Input required | `input` + `reference` + candidates (auto or manual) |
| Temperature | 0.0 |
| Max items | 7 (min 1) |
| Response schema | `WeightedChecklistResponse` |
| Primary metric | `weighted_score` |

```python
# Auto-generate candidates
pipe = pipeline("rlcf_candidate", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="...", target="...", reference="...")

# Or pass candidates manually
result = pipe(input="...", target="...", reference="...",
              candidates=["response A", "response B"])

# Or use a separate model for candidate generation
pipe = pipeline("rlcf_candidate", generator_model="openai/gpt-5-mini",
                generator_kwargs={"candidate_provider": "vllm",
                                  "candidate_base_url": "http://gpu:8000/v1"})
```

!!! info "Candidate Generation Strategy"
    The original paper generates candidates from different "worker" models to get naturally diverse responses. The `ContrastiveGenerator` supports this and a convenience fallback:

    - **Multiple `candidate_models`** (paper-faithful): Each model generates exactly **1 response** at temperature 0.7. Diversity comes from model differences. Use by instantiating `ContrastiveGenerator` directly and passing it to `ChecklistPipeline`.
    - **Single model** (convenience fallback): Generates `num_candidates` responses (default 4) at **temperature 0.9**. Diversity comes from high-temperature sampling. This is what `pipeline()` with `generator_kwargs={"candidate_provider": ...}` uses.

#### RLCF Candidates Only

Derives criteria from candidate diversity alone — no reference response needed.

| Setting | Value |
|---------|-------|
| Pipeline | `rlcf_candidates_only` |
| Generator | `ContrastiveGenerator` with `rlcf/candidates_only.md` template |
| Input required | `input` + candidates (auto or manual) |
| Temperature | 0.0 |
| Max items | 7 (min 1) |
| Response schema | `WeightedChecklistResponse` |
| Primary metric | `weighted_score` |

```python
pipe = pipeline("rlcf_candidates_only", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="...", target="...", candidates=["resp1", "resp2"])
```

**Differences from paper (all RLCF pipelines):**

| Paper feature | Our implementation |
|---------------|-------------------|
| Scoring via 0–100 continuous AI judge scores averaged across samples | Uses binary YES/NO per item with importance-weighted aggregation |
| Separability filtering (top 40% most discriminative pairs) | Not implemented; all generated items are kept |
| Direct Preference Optimization (DPO) training loop | Out of scope — this library focuses on checklist generation and scoring, not policy training |
| Exact verifier programs for discrete checks (e.g., format, length) | Not implemented; all scoring is LLM-based |

---

## Corpus-Level Methods

### Feedback (From Feedback to Checklists)

**Paper:** EMNLP 2025 Industry Track, "From Feedback to Checklists"

**Original methodology:** Transforms user/reviewer feedback into evaluation checklists through a 5-stage pipeline: (1) generate candidate questions from feedback batches, (2) merge redundant questions via embeddings, (3) filter by applicability and specificity, (4) validate enforceability via unit testing, (5) select a diverse subset via beam search. Evaluated on clinical note quality. The key insight is that real user feedback captures evaluation criteria that predefined rubrics miss.

**Our implementation:**

| Setting | Value |
|---------|-------|
| Registry name | `feedback` |
| Generator | `InductiveGenerator` |
| Input required | `observations` (list of strings) |
| Primary metric | `pass_rate` |
| Built-in refinement | Deduplicator → Tagger → Selector |

The generator has a built-in refinement pipeline that runs by default:

1. **Generate** — LLM creates candidate questions from batches of feedback comments
2. **Deduplicate** — merges similar questions via embedding similarity (threshold 0.85)
3. **Tag** — filters by applicability and specificity
4. **Select** — beam search for diverse subset (if more questions than `max_questions`)

Each stage can be selectively skipped. See [InductiveGenerator usage](generators.md#inductivegenerator) for code examples.

!!! info "Unit Testing Not Included by Default"
    The original paper includes enforceability validation (stage 4) in its pipeline. Our `InductiveGenerator` does **not** run the `UnitTester` by default because it requires pre-existing sample scores. You can add it as a standalone [refiner](refiners.md) after an initial scoring round.

!!! info "Selection Simplification"
    The original paper's selection step optimizes on **assignment matrices** — maximizing coverage of input feedback by ensuring each comment maps to at least one selected question (via `source_feedback_indices`). Our Selector simplifies this to embedding diversity as a proxy for coverage.

**Differences from paper:**

| Paper feature | Our implementation |
|---------------|-------------------|
| Enforceability unit testing (stage 4) in the default pipeline | Available as standalone `UnitTester` refiner, but not run by default (requires pre-existing sample scores) |
| Selection optimizes assignment matrices for feedback coverage | Simplified to embedding diversity via beam search; `source_feedback_indices` are tracked but not used in selection |
| Domain-specific (clinical note sections) | Generalized to any domain via the `domain` parameter |

---

### CheckEval

**Paper:** [arXiv:2403.18771](https://arxiv.org/abs/2403.18771)

**Original methodology:** Generates checklists from human-written evaluation dimension definitions through: seed question generation → augmentation (elaboration for granularity, diversification for alternative framings) → filtering (alignment check, dimension consistency, redundancy removal). Evaluated on SummEval and Topical-Chat. The key insight is that structured dimensions produce more systematic and complete evaluation criteria.

**Our implementation:**

| Setting | Value |
|---------|-------|
| Registry name | `checkeval` |
| Generator | `DeductiveGenerator` |
| Input required | `dimensions` (list of `DeductiveInput`) |
| Primary metric | `pass_rate` |
| Built-in refinement | Augmentation + optional filtering (with dedup) |

Three augmentation modes control question volume:

| Mode | Questions per Sub-Dimension | Description |
|------|----------------------------|-------------|
| `seed` | 2 | Minimal seed questions |
| `elaboration` | 5 | Detailed, granular questions |
| `diversification` | 4 | Alternative framings of criteria |

Optional filtering (`apply_filtering=True`) runs: alignment check → dimension consistency check → deduplication.

See [DeductiveGenerator usage](generators.md#deductivegenerator) for code examples.

**Differences from paper:**

| Paper feature | Our implementation |
|---------------|-------------------|
| Diversification and elaboration run **independently in parallel** from the same seeds, then merged before filtering | Use `augmentation_mode="combined"` to run both elaboration and diversification from seeds (paper-faithful). Individual modes (`"seed"`, `"elaboration"`, `"diversification"`) are also available. |
| Seed questions written by human experts | Seed questions are LLM-generated (from dimension + sub-dimension definitions) |
| Supervised weighting via linear regression on annotation data | Not implemented; all questions are equally weighted (uniform `pass_rate`) |

---

### InteractEval

**Paper:** [arXiv:2409.07355](https://arxiv.org/abs/2409.07355)

**Original methodology:** Collects think-aloud data from both human evaluators and LLMs, then extracts evaluation criteria through a 5-stage pipeline: (1) component extraction (recurring themes), (2) attribute clustering under components, (3) key question generation (1 per component), (4) sub-question generation (2–3 per component), (5) validation and refinement. Pass rate is scaled to a 1–5 dimension score. Evaluated on SummEval and ELLIPSE. The key insight is that combining human and LLM evaluation perspectives produces more comprehensive criteria.

**Our implementation:**

| Setting | Value |
|---------|-------|
| Registry name | `interacteval` |
| Generator | `InteractiveGenerator` |
| Input required | `inputs` (list of `InteractiveInput`) |
| Primary metric | `pass_rate` |
| Built-in refinement | 5-stage pipeline with validation |

The 5-stage pipeline:

1. **Component extraction** — identifies up to `max_components` (default 5) recurring themes
2. **Attribute clustering** — groups attributes under each component
3. **Key question generation** — 1 yes/no question per component
4. **Sub-question generation** — 2–3 sub-questions per component
5. **Validation** — refines and validates the final question set

See [InteractiveGenerator usage](generators.md#interactivegenerator) for code examples.

InteractEval-style 1–5 scoring is available via `result.score.scaled_score_1_5`.

**Differences from paper:**

| Paper feature | Our implementation |
|---------------|-------------------|
| Think-aloud collection protocol (4 humans + 4 LLMs with rubrics and sample texts) | Think-aloud data is provided as input (`InteractiveInput`); the collection protocol is external to the library |
| Validation uses 7 explicit criteria (yes/no answerable, dimension concepts, minimizes subjectivity, semantically distinct, positive framing, dimension-relevant, actionable) | Validation is handled by a single LLM call (stage 5) that checks these criteria holistically rather than as separate classification passes |
| Deduplication via LLM judgment in validation step | LLM-only dedup within the validation prompt; does not use embedding-based `Deduplicator` refiner (available as standalone but not wired in) |
| Five think-aloud conditions tested (single-LLM, single-human, multi-LLM, multi-human, combined) | All inputs are merged; the `source` field on `InteractiveInput` tracks provenance but doesn't affect the generation pipeline |

---

## Comparison Table

| Method | Level | Input Required | Reference? | Candidates? | Primary Metric | Built-in Refinement? |
|--------|-------|---------------|-----------|------------|-------------|---------------------|
| TICK | Instance | `input` | No | No | `pass_rate` | No |
| RocketEval | Instance | `input` | Yes | No | `normalized_score` | No |
| RLCF Direct | Instance | `input` | Yes | No | `weighted_score` | No |
| RLCF Candidate | Instance | `input` | Yes | Yes (auto) | `weighted_score` | No |
| RLCF Candidates Only | Instance | `input` | No | Yes (auto) | `weighted_score` | No |
| Feedback | Corpus | `observations` list | No | No | `pass_rate` | Yes (dedup, tag, select) |
| CheckEval | Corpus | `dimensions` | No | No | `pass_rate` | Yes (augment, filter, dedup) |
| InteractEval | Corpus | `think-aloud` | No | No | `pass_rate` (or 1–5) | Yes (5-stage pipeline) |
