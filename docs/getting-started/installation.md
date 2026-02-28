# Installation

## From Source

```bash
git clone https://github.com/ChicagoHAI/AutoChecklist.git
cd AutoChecklist
uv sync
```

## As a Dependency

```bash
uv pip install -e /path/to/autochecklist
```

## Optional: vLLM for Offline Inference

```bash
uv pip install "autochecklist[vllm]"
```

## Environment Setup

```bash
cp .env.example .env
```

Edit `.env` and add your API key(s):

- `OPENROUTER_API_KEY=sk-or-...` — for OpenRouter (the default provider)
- `OPENAI_API_KEY=sk-...` — for direct OpenAI or corpus-level embeddings
