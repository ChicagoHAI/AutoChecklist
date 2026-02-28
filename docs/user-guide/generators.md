# Generators

## What is a Generator?

A **generator** is the first stage of the checklist evaluation pipeline. It takes evaluation context (an instruction, reference response, feedback, etc.) and produces a **checklist** — a list of yes/no questions that capture what makes a response good or bad.

For example, given the instruction *"Write a haiku about autumn"*, a generator might produce:

- Does the response follow the 5-7-5 syllable structure?
- Does the response relate to autumn themes?
- Does the response use concrete imagery?

These questions become the criteria against which a [scorer](scorers.md) later evaluates a target response.

## Five Generator Abstractions

The library provides 5 generator classes, each representing a distinct reasoning approach. See the [Five Generator Abstractions](#five-generator-abstractions) section below for an overview. Each class is customizable via prompt templates, chainable with [refiners](refiners.md), and pairable with [scorers](scorers.md).

### Instance-Level: One Checklist Per Input

Instance-level generators create a tailored checklist for each individual input. The criteria adapt to the specific task: a haiku prompt gets questions about syllable structure, while a code prompt gets questions about correctness and efficiency.

- **`DirectGenerator`** — *Direct inference*: The LLM reads a prompt template with the evaluation context and directly produces checklist questions. Different prompt templates (via [built-in pipelines](supported-pipelines.md)) specialize it for different methods: `tick`, `rocketeval`, `rlcf_direct`.

- **`ContrastiveGenerator`** — *Counterfactual reasoning*: Instead of generating criteria directly, this approach first generates candidate responses of varying quality, then derives criteria by analyzing what distinguishes good from bad responses. The contrast between candidates reveals evaluation dimensions that direct prompting might miss. Used by `rlcf_candidate` and `rlcf_candidates_only` pipelines.

### Corpus-Level: One Checklist Per Dataset

Corpus-level generators produce a **shared checklist** from dataset-wide quality signals, such as reviewer observations, evaluation dimensions, or think-aloud sessions. Every response in the dataset is then scored against the same criteria.

- **`InductiveGenerator`** — *Inductive reasoning (bottom-up)*: Starts from specific observations (reviewer feedback, quality notes, error reports) and induces general evaluation criteria. Like inductive reasoning, it moves from concrete examples to abstract principles. Has a built-in refinement pipeline (deduplication → tagging → selection).

- **`DeductiveGenerator`** — *Deductive reasoning (top-down)*: Starts from abstract evaluation dimensions and sub-dimensions defined by domain experts, then generates specific checklist questions. Like deductive reasoning, it moves from general principles to concrete criteria. Supports augmentation modes (seed, elaboration, diversification, combined) and optional filtering.

- **`InteractiveGenerator`** — *Protocol analysis*: Extracts criteria from think-aloud evaluation sessions where human and LLM evaluators verbalize their reasoning while assessing quality. This approach captures tacit evaluation knowledge that isn't easily expressed as explicit dimensions or feedback. Runs a 5-stage pipeline (component extraction → clustering → key questions → sub-questions → validation).

!!! info "Terminology"
    Throughout the library:

    - **`input`** = the instruction, query, or task being evaluated
    - **`target`** = the response being evaluated (e.g., an LLM response, an essay)
    - **`reference`** = a gold-standard "good" response, sometimes used to help generate the checklist


A visualization of this difference:
![Diagrams comparing generators](../instance-vs-corpus.png)

## Using Instance-Level Generators

### Custom Prompts

The primary way to create an instance-level generator — write your own prompt template:

```python
from autochecklist import DirectGenerator

gen = DirectGenerator(
    custom_prompt="Generate yes/no eval questions for:\n\n{input}",
    model="openai/gpt-5-mini",
)
checklist = gen.generate(input="Write a haiku about autumn.")
print(checklist.to_text())
```

See [Custom Prompts](custom-prompts.md) for the full guide (file-based prompts, placeholders, custom scorers, registration).

### DirectGenerator

Use a built-in pipeline via `method_name`:

```python
from autochecklist import DirectGenerator

gen = DirectGenerator(method_name="tick", model="openai/gpt-5-mini")
checklist = gen.generate(input="Write a haiku about autumn.")
print(checklist.to_text())
```

### ContrastiveGenerator: Candidate Options

The `ContrastiveGenerator` (used by `rlcf_candidate` and `rlcf_candidates_only`) works by comparing candidate responses of varying quality. You have three options for providing candidates:

**Auto-generate candidates** (default) — the generator creates them using its LLM:

```python
from autochecklist import pipeline

pipe = pipeline("rlcf_candidate", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="...", target="...", reference="...")
# Candidates generated automatically
```

**Pass candidates manually:**

```python
result = pipe(input="...", target="...", reference="...",
              candidates=["response A", "response B", "response C"])
```

**Use a separate (cheaper) model for candidate generation:**

```python
pipe = pipeline("rlcf_candidate", generator_model="openai/gpt-5-mini",
                generator_kwargs={"candidate_provider": "vllm",
                                  "candidate_base_url": "http://gpu:8000/v1"})
```

**Use multiple models for diverse candidates** (paper-faithful approach):

The original RLCF paper generates candidates using different "worker" models to get naturally diverse responses. To replicate this, instantiate the generator directly and pass it to `ChecklistPipeline`:

```python
from autochecklist import ContrastiveGenerator, ChecklistPipeline

gen = ContrastiveGenerator(
    method_name="rlcf_candidate",
    model="openai/gpt-5-mini",
    candidate_models=["openai/gpt-5-mini", "anthropic/claude-3-haiku", "meta-llama/llama-3-8b"],
)
pipe = ChecklistPipeline(generator=gen, scorer="weighted", scorer_model="openai/gpt-5-mini")
result = pipe(input="...", target="...", reference="...")
```

!!! info "How Candidate Generation Works"
    The strategy depends on whether you provide one or multiple `candidate_models`:

    - **Multiple models** (paper-faithful): Each model generates exactly **1 response** at temperature 0.7. Diversity comes from model differences.
    - **Single model** (convenience fallback): Generates `num_candidates` responses (default 4) at **temperature 0.9**. Diversity comes from high-temperature sampling.

    When using `pipeline()` with `generator_kwargs={"candidate_provider": ...}`, a single model is used. For multi-model candidates, instantiate `ContrastiveGenerator` directly and pass it to `ChecklistPipeline`.

See [Supported Pipelines](supported-pipelines.md) for full details on each method's inputs and behavior.

## Using Corpus-Level Generators

### InductiveGenerator

Derives checklist questions from a list of reviewer feedback comments. Internally runs a refinement pipeline (deduplication → tagging → selection) that you can selectively disable:

```python
from autochecklist import InductiveGenerator

gen = InductiveGenerator(model="openai/gpt-5-mini")
checklist = gen.generate(
    observations=["Response was too long", "Missing examples", "Good structure"],
    domain="technical documentation",
    skip_dedup=False,      # default: run deduplication
    skip_tagging=False,    # default: run tagging
    skip_selection=False,  # default: run selection
)
```

### DeductiveGenerator

Creates questions from structured evaluation dimensions. Supports three augmentation modes that control how many questions are generated per sub-dimension:

```python
from autochecklist import DeductiveGenerator
from autochecklist.models import DeductiveInput

dims = [
    DeductiveInput(
        name="Coherence",
        definition="Logical flow and consistency of ideas",
        sub_dimensions=["Sentence transitions", "Paragraph structure"]
    )
]

gen = DeductiveGenerator(model="openai/gpt-5-mini", augmentation_mode="elaboration")
checklist = gen.generate(dimensions=dims, apply_filtering=True)
```

Augmentation modes: `"seed"` (1–3 questions/sub-dimension), `"elaboration"` (3–5 detailed questions), `"diversification"` (alternative framings), `"combined"` (both elaboration and diversification from seeds — paper-faithful).

### InteractiveGenerator

Extracts evaluation criteria from think-aloud data through a 5-stage pipeline (component extraction → attribute clustering → key questions → sub-questions → validation):

```python
from autochecklist import InteractiveGenerator
from autochecklist.models import InteractiveInput

data = [
    InteractiveInput(
        source="human",
        dimension="Helpfulness",
        attributes=["addresses the question", "provides examples", "clear explanation"]
    )
]

gen = InteractiveGenerator(model="openai/gpt-5-mini", max_components=5)
checklist = gen.generate(inputs=data)
```

### Grouping by Category

Corpus-level generators that produce items across multiple dimensions (like `DeductiveGenerator` and `InteractiveGenerator`) set `category` on each `ChecklistItem`. Use `by_category()` to split the checklist into per-dimension sub-checklists:

```python
checklist = gen.generate(dimensions=dims)

# Split into sub-checklists by dimension
groups = checklist.by_category()
for name, sub_cl in groups.items():
    print(f"{name}: {len(sub_cl)} items")

# Score each dimension separately
from autochecklist import ChecklistScorer
scorer = ChecklistScorer(mode="batch", model="openai/gpt-5-mini")
for name, sub_cl in groups.items():
    score = scorer.score(sub_cl, target="...", input="...")
    print(f"{name}: {score.pass_rate:.0%}")
```

`DeductiveGenerator` also has a convenience method:

```python
groups = gen.generate_grouped(dimensions=dims)  # equivalent to gen.generate(...).by_category()
```

Checklists can be saved and loaded for later use:

```python
checklist.save("my_eval.json")
loaded = Checklist.load("my_eval.json")
```

!!! note "Built-in Refinement"
    Some corpus-level generators have built-in refinement stages that use the same components available as standalone [refiners](refiners.md). See [Supported Pipelines](supported-pipelines.md) for details on each generator's internal pipeline.
