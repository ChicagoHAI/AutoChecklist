# AutoChecklist

A library of composable pipelines for generating and scoring checklist criteria.

A **checklist** is a list of yes/no questions used. `autochecklist` provides **5 generator abstractions**, each representing a different reasoning approach to producing evaluation criteria, along with a configurable **`ChecklistScorer`** that consolidates three scoring strategies from literature. You can mix, extend, and customize all components.

### Terminology
- **`input`**: The instruction, query, or task given to the LLM being evaluated (e.g., "Write a haiku about autumn").
- **`target`**: The output being evaluated against the checklist (e.g., the haiku the LLM produced).
- **`reference`**: An optional gold-standard response used by some methods to improve checklist generation.


### Generator Abstractions

The core of the library is 5 generator classes, each implementing a distinct approach to producing checklists:

| Level | Generator | Approach | Analogy |
|-------|-----------|----------|---------|
| Instance | **`DirectGenerator`** | Prompt → checklist | Direct inference |
| Instance | **`ContrastiveGenerator`** | Candidates → checklist | Counterfactual reasoning |
| Corpus | **`InductiveGenerator`** | Observations → criteria | Inductive reasoning (bottom-up) |
| Corpus | **`DeductiveGenerator`** | Dimensions → criteria | Deductive reasoning (top-down) |
| Corpus | **`InteractiveGenerator`** | Eval sessions → criteria | Protocol analysis |

**Instance-level** generators produce one checklist per input — criteria are tailored to each specific task. **Corpus-level** generators produce one checklist for an entire dataset — criteria capture general quality patterns derived from higher-level signals.


Each generator is customizable via prompt templates (`.md` files with `{input}`, `{target}` placeholders). You can use the built-in paper implementations, write your own prompts, or chain generators with different refiners and scorers to build custom evaluation pipelines.

### Built-in Pipelines

The library includes built-in pipelines implementing methods from research papers ([TICK](https://arxiv.org/abs/2410.03608), [RocketEval](https://arxiv.org/abs/2503.05142), [RLCF](https://arxiv.org/abs/2507.18624), [CheckEval](https://arxiv.org/abs/2403.18771), [InteractEval](https://arxiv.org/abs/2409.07355), and more). See [Supported Pipelines](https://github.com/ChicagoHAI/AutoChecklist/blob/main/docs/user-guide/pipelines.md) for the full list and configuration details.

### Scoring

A single configurable `ChecklistScorer` class supports all scoring modes:

| Config | Description |
|--------|-------------|
| `mode="batch"` | All items in one LLM call (efficient) |
| `mode="batch", capture_reasoning=True` | Batch with per-item explanations |
| `mode="item"` | One item per call |
| `mode="item", capture_reasoning=True` | One item per call with reasoning |
| `mode="item", primary_metric="weighted"` | Item weights (0-100) for importance |
| `mode="item", use_logprobs=True` | Logprob confidence calibration |

### Refiners

Refiners are pipeline stages that clean up raw checklists before scoring. They're used by corpus-level generators internally, and can also be composed into custom pipelines:

- **Deduplicator** — merges semantically similar items via embeddings
- **Tagger** — filters by applicability and specificity
- **UnitTester** — validates that items are enforceable
- **Selector** — picks a diverse subset via beam search

## Installation

```bash
pip install autochecklist
```

### Optional extras

```bash
# ML dependencies for corpus-level refiners (embeddings, deduplication)
pip install "autochecklist[ml]"

# vLLM for offline GPU inference (no server needed)
pip install "autochecklist[vllm]"

# Everything
pip install "autochecklist[all]"
```

For development installation from source, see the [GitHub repository](https://github.com/ChicagoHAI/AutoChecklist).

## Using the Package

### Custom Prompts

Write a prompt template and generate a checklist:

```python
from autochecklist import DirectGenerator, ChecklistScorer

gen = DirectGenerator(
    custom_prompt="You are an expert evaluator. Generate yes/no checklist questions to score:\n\n{input}",
    model="openai/gpt-5-mini",
)
checklist = gen.generate(input="Write a haiku about autumn.")

scorer = ChecklistScorer(mode="batch", model="openai/gpt-5-mini")
score = scorer.score(checklist, target="Leaves fall gently down...")
print(f"Pass rate: {score.pass_rate:.0%}")
```

Scorers also take custom prompts. Prompts can also be loaded from `.md` files — see [Custom Prompts](https://github.com/ChicagoHAI/AutoChecklist/blob/main/docs/user-guide/custom-prompts.md) for the full guide (placeholders, custom scorers, registration).

### Custom Pipelines

Register a custom pipeline (generator + scorer + prompts) as a reusable unit:

```python
from autochecklist import register_custom_pipeline, pipeline

# Register from config
register_custom_pipeline(
    "my_eval",
    generator_prompt="Generate yes/no questions for:\n\n{input}",
    scorer="weighted",
)
pipe = pipeline("my_eval", generator_model="openai/gpt-5-mini")

# Or register from an existing pipeline instance
register_custom_pipeline("my_eval_v2", pipe)

# Save/load pipeline configs as JSON
from autochecklist import save_pipeline_config, load_pipeline_config
save_pipeline_config("my_eval", "my_eval.json")
load_pipeline_config("my_eval.json")  # registers and returns the name
```

### Built-in Pipelines

The library includes pipelines implementing methods from research papers. Use them via `method_name` or the `pipeline()` shorthand:

```python
from autochecklist import pipeline

pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="Write a haiku about autumn", target="Leaves fall gently...")
print(f"Pass rate: {result.pass_rate:.0%}")
```

See [Supported Pipelines](https://github.com/ChicagoHAI/AutoChecklist/blob/main/docs/user-guide/pipelines.md) for the full list of pipelines, paper details, and configuration options.

### Batch Evaluation

```python
data = [
    {"input": "Write a haiku", "target": "Leaves fall..."},
    {"input": "Write a limerick", "target": "There once was..."},
]
result = pipe.run_batch(data, show_progress=True)
print(f"Macro pass rate: {result.macro_pass_rate:.0%}")
```

For pipeline composition, provider configuration, and the full API, see the [Pipeline Guide](https://github.com/ChicagoHAI/AutoChecklist/blob/main/docs/user-guide/pipeline.md).

### Command-Line Interface

Run evaluations directly from the terminal:

```bash
# Full evaluation (generate + score)
autochecklist run --pipeline tick --data eval_data.jsonl -o results.jsonl \
  --generator-model openai/gpt-4o-mini --scorer-model openai/gpt-4o-mini

# Generate checklists only
autochecklist generate --pipeline tick --data inputs.jsonl -o checklists.jsonl \
  --generator-model openai/gpt-4o-mini

# Score with existing checklist
autochecklist score --data eval_data.jsonl --checklist checklist.json \
  -o results.jsonl --scorer-model openai/gpt-4o-mini

# List available pipelines
autochecklist list
```

API keys can be set via `--api-key`, environment variables (`OPENROUTER_API_KEY`), or a `.env` file. See the [CLI Guide](https://github.com/ChicagoHAI/AutoChecklist/blob/main/docs/user-guide/cli.md) for full details.

### Examples

Detailed examples with runnable code:

- **[custom_components_tutorial.ipynb](https://github.com/ChicagoHAI/AutoChecklist/blob/main/examples/custom_components_tutorial.ipynb)** - Create your own generators, scorers, and refiners
- **[pipeline_demo.ipynb](https://github.com/ChicagoHAI/AutoChecklist/blob/main/examples/pipeline_demo.ipynb)** - Pipeline API, registry, batch evaluation, export
- **[instance_level_demo.ipynb](https://github.com/ChicagoHAI/AutoChecklist/blob/main/examples/instance_level_demo.ipynb)** - DirectGenerator, ContrastiveGenerator (per-input checklists)
- **[corpus_level_demo.ipynb](https://github.com/ChicagoHAI/AutoChecklist/blob/main/examples/corpus_level_demo.ipynb)** - InductiveGenerator, DeductiveGenerator, InteractiveGenerator (per-dataset checklists)

## Links

<!-- - [Full Documentation](https://autochecklist.github.io) -->
- [GitHub Repository](https://github.com/ChicagoHAI/AutoChecklist) — contributing, UI, dev setup
- [Bug Tracker](https://github.com/ChicagoHAI/AutoChecklist/issues)

## Citation

```
TBA
```

## License

Apache-2.0 (see `LICENSE`)
