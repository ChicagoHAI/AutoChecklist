# Providers

`autochecklist` supports multiple LLM providers through one common API.

## Which Provider Should You Use?

| Provider | Best for | Required env var |
|---|---|---|
| `openrouter` (default) | Access many models through one key | `OPENROUTER_API_KEY` |
| `openai` | Direct OpenAI usage | `OPENAI_API_KEY` |
| `vllm` | Self-hosted OpenAI-compatible endpoint | Usually none (depends on your server) |

## OpenRouter (Default)

```python
from autochecklist import pipeline

pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
# Uses OPENROUTER_API_KEY from environment
```

## OpenAI Direct

```python
from autochecklist import pipeline

pipe = pipeline("tick", provider="openai", generator_model="gpt-5-mini")
# Uses OPENAI_API_KEY from environment
```

## vLLM Server Mode

Connect to any OpenAI-compatible endpoint:

```python
from autochecklist import pipeline

pipe = pipeline("tick", provider="vllm", base_url="http://gpu:8000/v1", generator_model="meta-llama/Llama-3-8B")
```

## vLLM Offline (No Server)

Direct Python inference without a server:

```python
from autochecklist import VLLMOfflineClient, pipeline

client = VLLMOfflineClient(model="google/gemma-3-1b-it")
pipe = pipeline("tick", client=client)
```

## API Format: Chat vs Responses

Opt-in to OpenAI's Responses API format:

```python
from autochecklist import pipeline

pipe = pipeline("tick", provider="openai", generator_model="gpt-5-mini", api_format="responses")
```

Default is `api_format="chat"`. Keep that unless you specifically need Responses API behavior.
