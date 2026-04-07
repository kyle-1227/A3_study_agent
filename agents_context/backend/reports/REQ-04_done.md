# REQ-04: Engineering cleanup — Done

## Summary

Four independent sub-tasks completed: pyproject.toml migration, CORS env config, graph lifespan refactor, .gitignore cleanup.

## Files Changed

| File | Change |
|------|--------|
| `pyproject.toml` | Added `[project]` with all dependencies (upper-bounded), `[project.optional-dependencies]` for dev deps, `[build-system]`; kept existing `[tool.pytest.ini_options]` |
| `requirements.txt` | Unchanged — kept for Docker compatibility |
| `.github/workflows/ci.yml` | Replaced `pip install -r requirements.txt` with `pip install uv && uv pip install --system -e ".[dev]"`; updated cache keys from `requirements.txt` to `pyproject.toml` |
| `app.py` | Replaced `allow_origins=["*"]` with `os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")`; removed `global graph` / `graph = None`; stored graph as `app.state.graph`; `generate_sse()` now takes `graph` parameter; endpoint passes `request.app.state.graph` |
| `.env.example` | Added `ALLOWED_ORIGINS` with default value |
| `.gitignore` | Added `*.pyc`; replaced `frontend/%TEMP%/` with `frontend/dist/` |
| `tests/test_sse_lifecycle.py` | Updated all `generate_sse()` calls to pass `mock_graph` as parameter instead of patching `app.graph`; removed `patch` import |
| `tests/test_checkpointer.py` | Updated `generate_sse()` calls to pass `mock_graph` parameter |
| `tests/test_app.py` | **New file** — 12 tests covering: no hardcoded wildcard CORS, CORS env config, no global graph, app.state.graph usage, generate_sse graph param, pyproject.toml structure, .env.example ALLOWED_ORIGINS |

## Verification

- `app.py` contains **zero** `allow_origins=["*"]`
- `app.py` contains **zero** `global graph` or module-level `graph = None`
- `app.state.graph` used in lifespan and endpoint
- **316 passed, 1 skipped, 0 failed**

## How to Run Tests

```bash
conda run -n gaokao_tutor env OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short
```

## Suggested Commit

```
git add pyproject.toml .github/workflows/ci.yml app.py .env.example .gitignore tests/test_sse_lifecycle.py tests/test_checkpointer.py tests/test_app.py
git commit -m "feat(eng): pyproject.toml migration, CORS env config, app.state graph, .gitignore cleanup (REQ-04)"
```

## Design Notes

- `requirements.txt` is intentionally kept — it serves as the Docker install source and can be regenerated from pyproject.toml via `uv pip compile`
- CORS default is `http://localhost:3000` (frontend dev server) — production deploys should set `ALLOWED_ORIGINS` env var
- `generate_sse()` now receives graph as an explicit parameter instead of reading a module global — this makes testing straightforward and eliminates the need for `global` statement
- `frontend/%TEMP%/` was a Windows artifact in .gitignore — replaced with `frontend/dist/`
