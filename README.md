<p align="center">
  <img src="logos/autochecklist_banner-round.png" alt="AutoChecklist" width="600">
</p>

---

`AutoChecklist` is an open-source library that unifies LLM-based checklist evaluation into composable pipelines, in a `pip`-installable Python package (`autochecklist`) with CLI and UI features. 

### Features
-  **Five checklist generator abstractions** that organize methods from research by their reasoning strategies for deriving evaluation criteria
-  **Composable pipelines** eight built-in configurations implementing published methods, compatible with a unified scorer that consolidates three scoring strategies from literature
-  **CLI** for off-the-shelf evaluation with pre-defined pipelines
-  **Multi-provider LLM backend** with support for OpenAI, OpenRouter, and vLLM
-  **Locally hosted UI** allowing for interactive prompt customization, pipeline configuration, and batch evaluation


## Installation

### `pip`

```bash
uv pip install autochecklist

# Optional: install vLLM for offline inference (needs GPU)
uv pip install "autochecklist[vllm]"
```


### From source:

```bash
# Clone the repository
git clone https://github.com/ChicagoHAI/AutoChecklist.git
cd AutoChecklist

# Install dependencies
uv sync


# Set up environment
cp .env.example .env
# Edit .env and add your API key(s):
#   OPENROUTER_API_KEY=sk-or-...   (for OpenRouter, the default provider)
#   OPENAI_API_KEY=sk-...          (for direct OpenAI or corpus-level embeddings)
```

To use in another project:
```bash
uv pip install -e /path/to/AutoChecklist
```

## Concepts

> [!TIP]
> Some terminology:
> - **`input`**: The instruction, query, or task given to the LLM being evaluated (e.g., "Write a haiku about autumn").
> - **`target`**: The output being evaluated against the checklist (e.g., the haiku the LLM produced).
> - **`reference`**: An optional gold-standard response used by some methods to improve checklist generation.


### Checklist Generator Abstractions



<p align="left">
  <img src="docs/instance-vs-corpus.png" alt="AutoChecklist" width="400">
</p>


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

The library includes built-in pipelines implementing methods from research papers ([TICK](https://arxiv.org/abs/2410.03608), [RocketEval](https://arxiv.org/abs/2503.05142), [RLCF](https://arxiv.org/abs/2507.18624), [CheckEval](https://arxiv.org/abs/2403.18771), [InteractEval](https://arxiv.org/abs/2409.07355), and more). See [Supported Pipelines](docs/user-guide/supported-pipelines.md) for the full list and configuration details.

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

Scorers also take custom prompts. Prompts can also be loaded from `.md` files — see [Custom Prompts](docs/user-guide/custom-prompts.md) for the full guide (placeholders, custom scorers, registration).

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

See [Supported Pipelines](docs/user-guide/supported-pipelines.md) for the full list of pipelines, paper details, and configuration options.

### Batch Evaluation

```python
data = [
    {"input": "Write a haiku", "target": "Leaves fall..."},
    {"input": "Write a limerick", "target": "There once was..."},
]
result = pipe.run_batch(data, show_progress=True)
print(f"Macro pass rate: {result.macro_pass_rate:.0%}")
```

For pipeline composition, provider configuration, and the full API, see the [Pipeline Guide](docs/user-guide/pipeline.md).

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

API keys can be set via `--api-key`, environment variables (`OPENROUTER_API_KEY`), or a `.env` file. See the [CLI Guide](docs/user-guide/cli.md) for full details.

### Examples

Detailed examples with runnable code:

- **[custom_components_tutorial.ipynb](examples/custom_components_tutorial.ipynb)** - Create your own generators, scorers, and refiners
- **[pipeline_demo.ipynb](examples/pipeline_demo.ipynb)** - Pipeline API, registry, batch evaluation, export
- **[instance_level_demo.ipynb](examples/instance_level_demo.ipynb)** - DirectGenerator, ContrastiveGenerator (per-input checklists)
- **[corpus_level_demo.ipynb](examples/corpus_level_demo.ipynb)** - InductiveGenerator, DeductiveGenerator, InteractiveGenerator (per-dataset checklists)


## UI

A web interface for demonstrating `autochecklist` methods. See [ui/README.md](ui/README.md) for details.

**Quick Start:**
```bash
cd ui
./launch_ui.sh
# Frontend: http://localhost:7860
# Backend:  http://localhost:7861
```

## Testing

> [!WARNING]
> Integration tests make **real LLM API calls** that incur costs. The default fast tests (`-m 'not integration and not vllm_offline and not openai_api'`) are free and use no external services. Only run integration tests if you have the required API keys set and are okay with the associated costs.

```bash
# Core library fast tests (recommended default — no API calls)
uv run pytest -v -rs tests --ignore=ui/backend/tests -m 'not integration and not vllm_offline and not openai_api'

# Core integration tests (real API calls — requires OPENROUTER_API_KEY)
uv run pytest -v -rs tests --ignore=ui/backend/tests -m integration

# Backend API tests (no API calls needed)
cd ui/backend
uv run pytest -v -rs tests
```

See [tests/README.md](tests/README.md) and [ui/backend/tests/README.md](ui/backend/tests/README.md) for details.


## Automatic Documentation

Preview the API docs locally with MkDocs:

```bash
uv run mkdocs serve                       # http://localhost:8000
uv run mkdocs serve -a localhost:7772      # custom port
uv run mkdocs serve -a 0.0.0.0:7772       # expose on network (e.g., http://your-server:7772)
```

API reference is auto-generated from docstrings via [mkdocs-autoapi](https://github.com/jcayers20/mkdocs-autoapi). New modules are discovered automatically — no manual page creation needed.

## Citation

```
TBA
```

## License

Apache-2.0 (see `LICENSE`)
