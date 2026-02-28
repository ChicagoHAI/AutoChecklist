# Custom Prompts

Create your own generator or scorer by writing a prompt template.

## Custom Generator

### Inline Text

The simplest approach — pass a prompt string directly to `DirectGenerator`:

```python
from autochecklist import DirectGenerator

gen = DirectGenerator(
    custom_prompt="You are an expert evaluator. Generate yes/no checklist questions to score:\n\n{input}",
    model="openai/gpt-5-mini",
)
checklist = gen.generate(input="Write a haiku about autumn.")
```

> **`str` vs `Path`**: `custom_prompt` accepts both — a `str` is used as raw prompt text directly, a `Path` object loads from a file.

### From a File

Write a `.md` file with `{input}` placeholder:

```markdown
Generate yes/no evaluation questions for the following task.

**Input**: {input}

Output each question on its own line as a numbered list:
1. Does the response ...?
2. Is the response ...?
```

Load it with `DirectGenerator` or `pipeline()`:

```python
from pathlib import Path
from autochecklist import DirectGenerator

gen = DirectGenerator(custom_prompt=Path("my_generator.md"), model="openai/gpt-5-mini")
checklist = gen.generate(input="Write a haiku")
```

```python
from pathlib import Path
from autochecklist import pipeline

pipe = pipeline(custom_prompt=Path("my_generator.md"), scorer_model="openai/gpt-5-mini")
result = pipe(input="Write a haiku", target="Leaves fall...")
```

### Registration

Register a custom generator under a name for reuse across your project:

```python
from autochecklist import register_custom_generator, pipeline

register_custom_generator("my_eval", "my_generator.md")
pipe = pipeline("my_eval", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
result = pipe(input="Write a haiku", target="Leaves fall...")
```

## Custom Scorer

Write a `.md` file with `{input}`, `{target}`, and `{checklist}` placeholders:

```markdown
Evaluate the response against each criterion.

**Input**: {input}
**Response**: {target}
**Questions**:
{checklist}

For each question, answer: Q1: YES or NO, Q2: YES or NO, ...
```

Register and use with any generator:

```python
from autochecklist import register_custom_scorer, pipeline

register_custom_scorer("my_scorer", "my_scorer.md")
pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer="my_scorer")

# With scorer config (mode, primary_metric, etc.)
register_custom_scorer(
    "my_weighted_scorer", "my_scorer.md",
    mode="item", primary_metric="weighted",
)
```

## Combining Custom Generator and Scorer

```python
from autochecklist import register_custom_generator, register_custom_scorer, pipeline

register_custom_generator("code_review", "prompts/code_review.md")
register_custom_scorer("strict", "prompts/strict_scorer.md")

pipe = pipeline("code_review", scorer="strict")
```

## Placeholders

### Generator
| Placeholder | Required | Description |
|---|---|---|
| `{input}` | Yes | The input instruction/task |
| `{target}` | No | The response being evaluated |
| `{reference}` | No | A reference/gold response |

### Scorer
| Placeholder | Required | Description |
|---|---|---|
| `{input}` | Yes | The input instruction/task |
| `{target}` | Yes | The response being evaluated |
| `{checklist}` | Yes | Formatted as Q1: question, Q2: question, ... |

## How Output Parsing Works

**Generators** use **structured JSON output** (`ChecklistResponse` schema) by default. Your prompt template does not need to specify an output format — format instructions are automatically appended, and the LLM is guided to return structured JSON.

- If the LLM provider supports JSON schema enforcement (e.g., OpenAI, vLLM), it's used for guaranteed valid output
- Otherwise, the schema is included in the prompt and the response is parsed via JSON extraction

Custom generators registered via `register_custom_generator()` always use the unweighted `ChecklistResponse` schema. To use weighted output (`WeightedChecklistResponse`), instantiate `DirectGenerator` directly with `response_schema=WeightedChecklistResponse`.

**Scorers** also use structured JSON output (`BatchScoringResponse`, `ItemScoringResponse`, etc.) with the same provider-level enforcement and fallback. Your custom scorer prompt does not need to dictate the output format.
