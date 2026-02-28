# Backend Tests

Test suite for the FastAPI backend under `ui/backend/app/`.

## Scope

- Path: `ui/backend/tests/`
- Current size: `150` collected tests
- Covers:
  - Routers (`evaluate`, `batch`, `checklists`, `settings`, `registry`)
  - Service layer (`batch_runner`, `settings_service`, `storage`)
  - API behavior with isolated temp data directories

## Run Commands

From repo root:

```bash
cd ui/backend
source ../../.venv/bin/activate
pytest -q tests
```

Or with `uv`:

```bash
cd ui/backend
uv run pytest -q tests
```

## Fixtures

- Shared fixture in `ui/backend/tests/conftest.py` patches app imports for backend-local test execution.
- Most tests use `tmp_data_dir` to isolate all file writes.

## Notes

- Keep backend tests separate from root `tests/` collection.
- Batch/evaluate tests intentionally mock checklist/scoring internals where API contract is the target.
