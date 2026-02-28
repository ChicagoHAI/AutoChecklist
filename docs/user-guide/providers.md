# Providers

autochecklist supports multiple LLM providers.

## OpenRouter (Default)

```python
pipe = pipeline("tick", generator_model="openai/gpt-5-mini", scorer_model="openai/gpt-5-mini")
# Uses OPENROUTER_API_KEY from environment
```

## OpenAI Direct

```python
pipe = pipeline("tick", provider="openai", generator_model="gpt-5-mini")
# Uses OPENAI_API_KEY from environment
```

## vLLM Server Mode

Connect to any OpenAI-compatible server:

```python
pipe = pipeline("tick", provider="vllm", base_url="http://gpu:8000/v1", generator_model="meta-llama/Llama-3-8B")
```

## vLLM Offline (No Server)

Direct Python inference without a server:

```python
from autochecklist import VLLMOfflineClient, pipeline

client = VLLMOfflineClient(model="google/gemma-3-1b-it")
pipe = pipeline("tick", client=client)
```

## Responses API

Opt-in to OpenAI's Responses API format:

```python
pipe = pipeline("tick", provider="openai", generator_model="gpt-5-mini", api_format="responses")
```
