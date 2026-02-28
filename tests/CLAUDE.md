# Testing

## Separate Venvs

Library tests and backend tests use separate venvs and **must not be mixed**:
- **Library tests**: Run from project root — `uv run pytest tests/`
- **Backend tests**: Run from `ui/backend/` — `cd ui/backend && uv run pytest tests/`

Do NOT run backend tests from the project root — it triggers uv to create a phantom `auto-checklists/` directory due to conflicting project discovery.

## Test Commands

```bash
# Core library fast tests (recommended default — no API calls)
uv run pytest -v -rs tests --ignore=ui/backend/tests -m 'not integration and not vllm_offline and not openai_api'

# Core integration tests (real API calls — requires OPENROUTER_API_KEY)
uv run pytest -v -rs tests --ignore=ui/backend/tests -m integration

# Backend API tests (no API calls needed)
cd ui/backend && uv run pytest -v -rs tests

# Run specific test file
uv run pytest tests/test_generators/test_direct.py

# Run specific test
uv run pytest tests/test_generators/test_direct.py::TestDirectGeneratorConfig::test_tick_preset_loads
```

## API Key Requirements

| Scope | Key | Required For |
|-------|-----|-------------|
| Integration tests | `OPENROUTER_API_KEY` | Real LLM calls via OpenRouter |
| OpenAI provider tests | `OPENAI_API_KEY` | Direct OpenAI API tests |
| vLLM offline tests | GPU + vLLM installed | Marked `@pytest.mark.vllm_offline` |
| Unit tests | None | Parsing, config, models |
| Backend tests | None | 150+ unit tests, no external calls |

Tests skip automatically when the required API key is not set.

## Patterns

- Integration tests make **real LLM API calls** (not mocked) — they incur costs
- Uses `openai/gpt-4o-mini` by default for cost efficiency (~$0.05 for full suite)
- Provider tests in `tests/test_providers/` include unit tests (no key), OpenAI integration (need key), and vLLM offline (need GPU)
