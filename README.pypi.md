# AutoChecklist

<p align="center">
  <a href="https://github.com/ChicagoHAI/AutoChecklist"><img src="https://img.shields.io/github/stars/ChicagoHAI/AutoChecklist?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License"></a>
  <a href="https://autochecklist.github.io/"><img src="https://img.shields.io/badge/site-autochecklist.github.io-purple?style=flat-square" alt="Site"></a>
</p>

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

The library includes built-in pipelines implementing methods from research papers ([TICK](https://arxiv.org/abs/2410.03608), [RocketEval](https://arxiv.org/abs/2503.05142), [RLCF](https://arxiv.org/abs/2507.18624), [OpenRubrics](https://arxiv.org/abs/2510.07743), [CheckEval](https://arxiv.org/abs/2403.18771), [InteractEval](https://arxiv.org/abs/2409.07355), and more). See [Supported Pipelines](https://autochecklist.github.io/user-guide/supported-pipelines/) for the full list and configuration details.

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

## Quick Start

```python
from autochecklist import pipeline

pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="Write a haiku about autumn.", target="Leaves fall gently down...")
print(f"Pass rate: {result.pass_rate:.0%}")
```

See the [Quick Start guide](https://autochecklist.github.io/getting-started/quickstart/) for custom prompts, batch evaluation, and more.

### CLI

```bash
autochecklist run --pipeline tick --data eval_data.jsonl -o results.jsonl \
  --generator-model openai/gpt-4o-mini --scorer-model openai/gpt-4o-mini
```

See the [CLI guide](https://autochecklist.github.io/user-guide/cli/) for all commands.

## Links

- [Documentation](https://autochecklist.github.io)
- [GitHub Repository](https://github.com/ChicagoHAI/AutoChecklist) — contributing, UI, dev setup
- [Bug Tracker](https://github.com/ChicagoHAI/AutoChecklist/issues)

## Citation

```
TBA
```

## License

Apache-2.0 (see `LICENSE`)
