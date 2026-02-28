# Quick Start

This page shows the fastest path to a working evaluation.

## 1) End-to-End in 10 Lines

Use `pipeline()` to generate a checklist and score a response in one call:

```python
from autochecklist import pipeline

pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(
    input="Write a haiku about autumn.",
    target="Leaves drift through cool dusk; amber fields breathe into night; geese stitch quiet skies.",
)
print(f"Pass rate: {result.pass_rate:.0%}")
```

See [Supported Pipelines](../user-guide/supported-pipelines.md) for other built-ins.

## 2) Custom Prompt (No Preset Needed)

If you want your own criteria style, pass a prompt directly:

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

## 3) Load Prompt from File

Load a prompt template from a `.md` file:

```python
from pathlib import Path
from autochecklist import DirectGenerator

gen = DirectGenerator(custom_prompt=Path("my_eval_prompt.md"), model="openai/gpt-5-mini")
checklist = gen.generate(input="Write a haiku about autumn.")
```

> **`str` vs `Path`**: `custom_prompt` accepts both — a `Path` object loads from a file, a `str` is used as raw prompt text directly.

See [Custom Prompts](../user-guide/custom-prompts.md) for the full guide (placeholders, custom scorers, registration).

## 4) Use a Built-In Method via Generator + Scorer

```python
from autochecklist import DirectGenerator, ChecklistScorer

gen = DirectGenerator(method_name="tick", model="openai/gpt-5-mini")
checklist = gen.generate(input="Write a haiku about autumn.")

scorer = ChecklistScorer(mode="batch", model="openai/gpt-5-mini")
score = scorer.score(checklist, target="Leaves fall gently down...")
print(f"Pass rate: {score.pass_rate:.0%}")
```

## 5) Pipeline API (Same Result, Less Wiring)

For convenience, `pipeline()` creates a pre-configured generator + scorer from a pipeline name:

```python
from autochecklist import pipeline

pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="Write a haiku about autumn", target="Leaves fall gently...")
print(f"Pass rate: {result.pass_rate:.0%}")
```

See the [Pipeline Guide](../user-guide/pipeline.md) for composition, providers, and the full API.

## 6) Batch Evaluation

```python
data = [
    {"input": "Write a haiku", "target": "Leaves fall..."},
    {"input": "Write a limerick", "target": "There once was..."},
]
result = pipe.run_batch(data, show_progress=True)
print(f"Macro pass rate: {result.macro_pass_rate:.0%}")
```
