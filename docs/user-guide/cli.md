# Command-Line Interface

AutoChecklist provides a CLI for running evaluations directly from the terminal.

## Installation

```bash
pip install autochecklist
```

After installation, the `autochecklist` command is available globally.

## API Keys

AutoChecklist needs an API key for the LLM provider. Three options (in order of precedence):

1. **`--api-key` flag** — pass directly on the command line
2. **Environment variable** — `export OPENROUTER_API_KEY=sk-or-...` (or `OPENAI_API_KEY` for OpenAI provider)
3. **`.env` file** — create a `.env` file in your working directory with `OPENROUTER_API_KEY=sk-or-...`

## Subcommands

### `autochecklist run` — Full evaluation

Generate checklists and score targets in one step.

```bash
autochecklist run --pipeline tick --data eval_data.jsonl -o results.jsonl \
  --generator-model openai/gpt-4o-mini --scorer-model openai/gpt-4o-mini
```

| Flag | Required | Description |
|------|----------|-------------|
| `--pipeline` | No* | Pipeline name (tick, rocketeval, rlcf_direct, etc.) |
| `--config` | No | Path to pipeline config JSON file (see [Pipeline Configs](#pipeline-configs)) |
| `--data` | Yes | Path to input JSONL file |
| `-o, --output` | No | Output JSONL path (enables resume on re-run) |
| `--overwrite` | No | Overwrite output instead of resuming |
| `--generator-model` | No | Model for generation (e.g. `openai/gpt-4o-mini`) |
| `--scorer-model` | No | Model for scoring |
| `--scorer` | No | Override default scorer (batch, item, weighted, normalized) |
| `--generator-prompt` | No | Path to custom generator prompt (.md file) |
| `--scorer-prompt` | No | Path to custom scorer prompt (.md file) |
| `--provider` | No | LLM provider: openrouter (default), openai, vllm |
| `--base-url` | No | Custom base URL (e.g. vLLM server) |
| `--api-key` | No | API key (default: from environment) |
| `--api-format` | No | API format: chat (default), responses |
| `--input-key` | No | JSONL key for input field (default: `input`) |
| `--target-key` | No | JSONL key for target field (default: `target`) |

*Either `--pipeline`, `--generator-prompt`, or `--config` must be provided.

### `autochecklist generate` — Checklists only

Generate checklists without scoring.

```bash
autochecklist generate --pipeline tick --data inputs.jsonl -o checklists.jsonl \
  --generator-model openai/gpt-4o-mini
```

Same flags as `run` (including `--config`), minus `--scorer-model`, `--scorer`, `--scorer-prompt`, and `--target-key`.

### `autochecklist score` — Score only

Score targets against a pre-existing checklist.

```bash
autochecklist score --data eval_data.jsonl --checklist checklist.json \
  -o results.jsonl --scorer-model openai/gpt-4o-mini
```

| Flag | Required | Description |
|------|----------|-------------|
| `--data` | Yes | JSONL file with target field |
| `--checklist` | Yes | Path to checklist JSON file |
| `--scorer` | No | Scorer type (default: batch) |
| `--scorer-model` | No | Model for scoring |
| `--scorer-prompt` | No | Path to custom scorer prompt (.md file) |
| `-o, --output` | No | Output JSONL path |
| `--overwrite` | No | Overwrite output |

Plus the same provider flags (`--provider`, `--base-url`, `--api-key`, `--api-format`, `--input-key`, `--target-key`).

### `autochecklist list` — Discover components

```bash
autochecklist list                        # list generators (default)
autochecklist list --component scorers    # list scorers
autochecklist list --component refiners   # list refiners
```

## Input Format

The input JSONL file should have one JSON object per line:

```json
{"input": "Write a haiku about nature", "target": "Leaves fall gently down..."}
{"input": "Write a greeting", "target": "Hello! How are you?"}
```

Use `--input-key` and `--target-key` if your data uses different field names:

```bash
autochecklist run --pipeline tick --data data.jsonl --input-key instruction --target-key response
```

## Custom Prompts

Use `--generator-prompt` and `--scorer-prompt` to pass custom prompt templates without modifying code:

```bash
autochecklist run --generator-prompt my_eval.md --data data.jsonl \
  --generator-model openai/gpt-4o-mini --scorer-model openai/gpt-4o-mini
```

See [Custom Prompts](custom-prompts.md) for prompt template format details.

## Resumable Runs

When `--output` is set, results are written incrementally. If interrupted, re-running the same command resumes from where it left off. Use `--overwrite` to start fresh.

```bash
# First run (interrupted after 50/100 examples)
autochecklist run --pipeline tick --data data.jsonl -o results.jsonl --generator-model openai/gpt-4o-mini

# Resume (picks up at example 51)
autochecklist run --pipeline tick --data data.jsonl -o results.jsonl --generator-model openai/gpt-4o-mini

# Start over
autochecklist run --pipeline tick --data data.jsonl -o results.jsonl --overwrite --generator-model openai/gpt-4o-mini
```

## Pipeline Configs

Save and reuse custom pipeline configurations as JSON files. This is useful for sharing evaluation setups across teams or projects.

```bash
# Run with a pipeline config file
autochecklist run --config my_pipeline.json --data data.jsonl -o results.jsonl \
  --generator-model openai/gpt-4o-mini --scorer-model openai/gpt-4o-mini
```

Config JSON format:

```json
{
  "name": "my-eval",
  "generator_class": "direct",
  "generator_prompt": "Generate yes/no evaluation questions for:\n\n{input}",
  "scorer_mode": "item",
  "scorer_prompt": null,
  "primary_metric": "weighted",
  "capture_reasoning": false
}
```

See `register_custom_pipeline()` and `save_pipeline_config()` in the Python API for creating configs programmatically.
