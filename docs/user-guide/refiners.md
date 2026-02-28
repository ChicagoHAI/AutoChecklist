# Refiners

## What is a Refiner?

A **refiner** is an optional pipeline stage that sits between [generation](generators.md) and [scoring](scorers.md). It takes a `Checklist` in and returns a refined `Checklist` out — pruning low-quality questions, merging duplicates, or selecting a diverse subset.

Refiners are primarily designed for **corpus-level checklists**, where a generator may produce dozens or hundreds of candidate questions from broad input (feedback comments, evaluation dimensions, etc.) and you need to whittle them down to a high-quality, non-redundant set.

## Standalone Refiners vs Built-in Refinement

There are two ways refinement happens in autochecklist:

**Standalone refiners** — the four refiner classes (`Deduplicator`, `Tagger`, `UnitTester`, `Selector`) that you can explicitly add to a pipeline or chain manually.

**Built-in refinement** — some corpus-level generators use these same components internally as part of their generation pipeline:

- **`InductiveGenerator`**: runs Deduplicator → Tagger → Selector internally. Control with `skip_dedup`, `skip_tagging`, `skip_selection` flags.
- **`DeductiveGenerator`**: has augmentation (seed/elaboration/diversification) and filtering (alignment check, dimension consistency, redundancy removal via Deduplicator) built in. Control with `augmentation_mode` and `apply_filtering` params.
- **`InteractiveGenerator`**: runs a 5-stage pipeline with validation as the final stage. No skip flags — the stages are integral to the method.

!!! tip "When to Use Standalone Refiners"
    Use standalone refiners when you want to:

    - Apply refinement to instance-level checklists
    - Add refinement stages that a corpus generator doesn't include by default (e.g., adding `UnitTester` to `InductiveGenerator`, which doesn't run it by default)
    - Override or replace a generator's built-in refinement with your own configuration

## The Four Standalone Refiners

### Deduplicator

Merges semantically similar questions using embedding-based similarity detection.

**Algorithm:**

1. Compute embeddings for all questions (via OpenAI embeddings API)
2. Build a similarity graph — add an edge between questions whose cosine similarity ≥ threshold
3. Find connected components (clusters of similar questions)
4. Single-question clusters are kept as-is
5. Multi-question clusters are merged by the LLM into one representative question

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | `0.85` | Cosine similarity cutoff for considering questions duplicates |

Note: currently the Deduplicator uses the OpenAI Embeddings API directly and does not support other providers.

```python
from autochecklist.refiners import Deduplicator

dedup = Deduplicator(model="openai/gpt-5-mini", similarity_threshold=0.85)
refined = dedup.refine(checklist)
```

### Tagger

Filters questions by two quality criteria using zero-shot chain-of-thought LLM classification. A question must pass **both** criteria to be kept:

1. **Generally applicable** — the question can be answered Yes/No for any input (no N/A scenarios)
2. **Section specific** — the question evaluates a single focused aspect (not compound or cross-referencing)

```python
from autochecklist.refiners import Tagger

tagger = Tagger(model="openai/gpt-5-mini")
refined = tagger.refine(checklist)
print(f"Filtered {len(checklist.items) - len(refined.items)} questions")
```

### UnitTester

Validates **enforceability** — can an LLM scorer reliably distinguish responses that pass vs. fail each criterion?

**Algorithm:**

1. For each question, find sample responses that scored YES (passing)
2. LLM rewrites each passing sample to intentionally **fail** the criterion
3. Score the rewritten sample — it should now score NO
4. `enforceability_rate = correct_failures / total_passing_samples`
5. Filter out questions below the threshold

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enforceability_threshold` | `0.7` | Minimum enforceability rate to keep a question |
| `max_samples` | `10` | Maximum passing samples to test per question |

!!! info "Required Inputs"
    UnitTester requires pre-existing sample data and scores — you need response texts and their checklist scores from a prior scoring run. This makes it a later-stage refiner, best used after you've done an initial round of scoring.

```python
from autochecklist.refiners import UnitTester

tester = UnitTester(
    model="openai/gpt-5-mini",
    enforceability_threshold=0.7,
    max_samples=10,
)
refined = tester.refine(checklist, samples=samples, sample_scores=scores)
```

### Selector

Selects an optimally **diverse** subset of questions via beam search over embedding similarity.

**Algorithm:**

The objective function balances diversity against checklist length:

```
Score(subset) = Diversity - λ × |subset|
```

Where `Diversity = 1 - average_pairwise_cosine_similarity` and λ is the length penalty. Beam search explores candidate subsets, adding one question at a time, keeping the top candidates at each step.

If the checklist already has fewer questions than `max_questions`, it's returned unchanged.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_questions` | `20` | Maximum questions to select |
| `beam_width` | `5` | Number of candidate subsets to track during search |
| `length_penalty` | `0.0005` | Penalty per additional question (λ) |

!!! info "On Coverage vs Diversity"
    The original Feedback paper optimized selection on **assignment matrices** — maximizing coverage of input feedback by ensuring each comment maps to at least one selected question. Our Selector simplifies this to embedding diversity as a proxy for coverage. The `source_feedback_indices` metadata is tracked during generation but not used in the selection optimization.

```python
from autochecklist.refiners import Selector

selector = Selector(max_questions=15, beam_width=5)
refined = selector.refine(checklist)
```

## Chaining Refiners

The recommended order for chaining refiners is:

1. **Deduplicator** first — reduces volume, saving API calls in later stages
2. **Tagger** second — removes inapplicable or unfocused questions
3. **UnitTester** third — validates enforceability (expensive, so run after pruning)
4. **Selector** last — picks the optimal diverse subset from the refined pool

### Via Pipeline

```python
from autochecklist import pipeline

pipe = pipeline(
    "feedback",
    model="openai/gpt-5-mini",
    refiners=["deduplicator", "tagger", "selector"],
)
```

### Manual Chaining

```python
from autochecklist.refiners import Deduplicator, Tagger, Selector

checklist = generator.generate(observations=[...])

checklist = Deduplicator(model="openai/gpt-5-mini").refine(checklist)
checklist = Tagger(model="openai/gpt-5-mini").refine(checklist)
checklist = Selector(max_questions=15).refine(checklist)
```
