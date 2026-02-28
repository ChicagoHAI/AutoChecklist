# Tests

Test suite for `autochecklist` (core library).

> **Warning:** Integration tests make **real LLM API calls** that incur costs. The fast/default subset is free and uses no external services. Only run integration tests (`-m integration`) if you have the required API keys set and are okay with the associated costs.

## Scope

- Path: `tests/`
- Current size: `538` collected tests
- Fast/default subset: `465` tests (`-m 'not integration and not vllm_offline and not openai_api'`)
- Optional integration subset: `73` tests (real APIs / offline inference)

This repo also has a separate backend suite in `ui/backend/tests/`.

## Run Commands

```bash
# Core fast tests (recommended default)
uv run pytest -q tests --ignore=ui/backend/tests -m 'not integration and not vllm_offline and not openai_api'

# Core integration tests only
uv run pytest -q tests --ignore=ui/backend/tests -m integration

# Core all tests (fast + integration)
uv run pytest -q tests --ignore=ui/backend/tests
```

## Markers

- `integration`: tests that make real API calls or rely on external runtimes
- `vllm_offline`: requires local vLLM + GPU
- `openai_api`: requires `OPENAI_API_KEY`

## Notes

- Many integration tests are guarded by `skipif` on required credentials.
- `tests/conftest.py` auto-adds the `integration` marker to known API/offline tests.
- Run backend tests separately from `ui/backend/` to avoid import-path collection issues.
