# Parent–Child RAG data plane

This package owns the strict, generation-scoped RAG data plane: page-aware
loading, parent/child splitting, immutable generation building, child Chroma,
per-subject BM25, authoritative SQLite parents, exact-generation runtime
loading, hybrid retrieval, evaluation projection, and safe regression traces.
It does not own Graph routing, streaming, frontend state, or deployment
activation decisions.

## Current verified state

- Candidate generation: `pc_20260715_98336c2_55`
- Lifecycle: `READY`, inactive
- Parents / children: `21,225 / 35,365`
- Orphan children / hydration failures: `0 / 0`
- Registry primary / previous / shadow: unset
- Retained Flat artifact: `flat_20260715_98336c2_53`
- Legacy rollback asset: repository-root `chroma_store`

`READY` means sealed artifact integrity passed. It does not mean retrieval
quality passed, and it does not route user traffic. Do not activate generation
55 or delete the legacy RAG.

The provider-backed Gold V2 engineering comparison regressed Recall@5 by 12
percentage points and MRR by 0.0475. A fixed 17-query stage diagnosis compared
`reranker_top_n=20` with `80`: hydrated Gold coverage rose from 6 to 8 spans,
but total P50 rose from 2679 to 4122 ms and P95 from 4932 to 7234 ms. Tuning is
closed; the Top80 policy is not production-eligible.

Gold V3 also remains incomplete: the current 22 approved authoring changes are
draft-only, the two independent semantic reviews are missing, and the 150-pair
chunk review is incomplete. V2 diagnostics must never be presented as a formal
production pass.

## Strict runtime rules

- Always pass an explicit config path and generation ID.
- Never resolve the active pointer in benchmark or diagnostic tooling.
- Candidate Vector, BM25, reranker, Parent Store, or hydration failure is a
  typed failure; never return Flat output as Candidate success.
- Open canonical Chroma only through marker-owned disposable snapshots.
- Keep API-key values only in the exact environment variables named by the
  strict config. Reports and logs may record presence booleans, never values.
- Runtime policy changes receive a new retrieval fingerprint. They do not
  mutate a sealed generation or authorize activation.

## Supported entrypoints

- `scripts/init_rag_runtime_config.py`: generate contained portable runtime
  paths from the tracked strict template.
- `scripts/probe_rag_providers.py`: live protocol and dimension probe with
  redacted output.
- `scripts/build_flat_baseline.py`: build a new isolated Flat artifact.
- `scripts/build_parent_child_generation.py`: build a new immutable generation.
- `scripts/run_parent_child_benchmark.py`: project the same GoldDataset through
  explicit Flat and Candidate arms.
- `scripts/diagnose_parent_child_regressions.py`: run a body-free 10–20 query
  stage trace on an exact READY generation.
- `scripts/validate_parent_child_candidate.py`: apply formal gates to complete,
  externally scored inputs.
- `scripts/manage_rag_generation.py`: registry-owned lifecycle and cleanup.

The authoritative commands, current fixed 17-query diagnostic, Gold workflow,
and activation prohibition are documented in
`docs/runbooks/parent_child_rag_local_build.md`. The safe A/B result is recorded
in `docs/reports/rag_parent_child_regression_diagnosis_20260715.md`.

## Focused verification

For changes in this package, start with:

```powershell
python -m compileall -q src/rag/parent_child scripts
ruff check src/rag/parent_child scripts/diagnose_parent_child_regressions.py
ruff format --check src/rag/parent_child scripts/diagnose_parent_child_regressions.py
python -m pytest -q `
  tests/test_parent_child_retrieval.py `
  tests/test_parent_child_regression_diagnostics.py `
  tests/test_parent_child_runtime_resources.py `
  tests/test_parent_child_chroma_runtime_snapshot.py `
  tests/test_parent_child_benchmarking.py
```

Run broader gates only once at integration. Missing security or architecture
tools must be reported as unavailable or failed, never as passed.
