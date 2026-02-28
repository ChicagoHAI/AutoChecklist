# Prompt Templates

This directory contains prompt templates used by generators and scorers. Templates are Markdown files with `{placeholder}` variables that get filled in at runtime.

For writing and registering custom prompts, see the [Custom Prompts guide](../../docs/user-guide/custom-prompts.md).

## Directory Structure

```
prompts/
├── generators/
│   ├── tick/generate.md          # TICK checklist generation
│   ├── rlcf/                     # RLCF (direct, candidate-based, candidates-only)
│   ├── rocketeval/generate.md    # RocketEval generation
│   ├── feedback/                 # InductiveGenerator (generate, tag, merge, rewrite)
│   ├── checkeval/                # DeductiveGenerator (generate, augment, filter)
│   └── interacteval/             # InteractiveGenerator (multi-step pipeline)
├── scoring/                      # Scorer prompts (batch, item, rlcf, rocketeval)
└── formats/                      # Output format instructions (structured JSON schemas)
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
