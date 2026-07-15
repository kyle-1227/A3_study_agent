# A3 Study Agent production deployment

This runbook is the final Docker and canary procedure for the 2026-07-15
production convergence. Run commands from the repository root. Never print or
commit `.env` values.

## 1. Release state

- PostgreSQL is the required production checkpointer.
- The served graph uses strict user/thread identity, structured contracts,
  journal replay, status recovery, and explicit resource terminal states.
- Parent-Child generation `pc_20260715_98336c2_55` is sealed `READY` but is not
  approved for activation. Registry primary, previous, and shadow pointers are
  unset.
- The repository-root Flat `chroma_store` and Flat generation 53 are retained
  rollback assets. The base Compose deployment does not select or silently
  fall back to them; do not delete them during this release.
- Evidence rollout activation remains disabled. The six-case dataset is a
  smoke authoring set, not a human-sealed production benchmark.
- The served backend used by the web canary loads
  `config/rag/index.production-candidate.inactive.yaml` and the generation
  fixed by `PARENT_CHILD_GENERATION_ID`; this tracked top20 configuration is
  sealed-compatible with generation 55 and contains environment-variable
  names, never secret values. The pin does not update a registry pointer.

## 2. Required local assets

The Compose deployment requires explicit host locations for both source data
and the immutable Parent-Child index. In `.env`, configure:

- `COURSE_DATA_HOST_PATH`
- `PARENT_CHILD_INDEX_HOST_PATH`
- `PARENT_CHILD_GENERATION_ID`
- `POSTGRES_PASSWORD`
- `NEXT_PUBLIC_API_URL`
- `DEEPSEEK_API_KEY`
- `RAG_EMBEDDING_API_KEY`
- `RAG_RERANKER_API_KEY`
- `TAVILY_API_KEY`

Keep `EMBEDDING_API_KEY_ENV=RAG_EMBEDDING_API_KEY` and
`RERANKER_API_KEY_ENV=RAG_RERANKER_API_KEY`. The two host paths may be relative
when the data and index live in this checkout. Docker mounts the Parent-Child
index read-only and persists generated downloads in a named `artifacts` volume.
The sealed Chroma tree stays read-only; disposable runtime snapshots use the
separate writable `rag_runtime_chroma` volume mounted at its designated
subdirectory.

When Compose reads a different ignored env file, set `A3_ENV_FILE` to its
absolute path. The default is the repository-root `.env`.

## 3. Build and start

Validate the fully interpolated Compose model without rendering it to logs.
Required-variable interpolation fails here without revealing the missing
values; Docker validates the two host directories when it creates mounts:

```powershell
docker compose config --quiet
```

Build and wait for PostgreSQL, backend, and frontend readiness:

```powershell
docker compose up --detach --build --wait
docker compose ps
```

The backend image installs Chromium and ffmpeg because video animation is a
production resource type. Backend and frontend run as separate supervised
containers; either service failure is visible to Compose.

## 4. Service checks

Run these checks before any Provider-backed request:

```powershell
Invoke-WebRequest http://localhost:8000/health/live -UseBasicParsing
Invoke-WebRequest http://localhost:8000/health/ready -UseBasicParsing
Invoke-WebRequest http://localhost:8000/graph/manifest -UseBasicParsing
Invoke-WebRequest http://localhost:8000/subjects -UseBasicParsing
Invoke-WebRequest http://localhost:3000 -UseBasicParsing
```

`/health/live` proves only that the API process can answer. `/health/ready`
must return `health_ready_v1`, `status=ready`, `checkpointer_type=postgres`,
and `candidate_mode=inactive_canary`, plus the graph version, KnowledgeGraph
data/artifact identity, generation ID/manifest fingerprint, and evidence
orchestration fingerprint. It also performs a bounded PostgreSQL `SELECT 1`.
Only the typed, redacted 503 code may be recorded when readiness fails; never
print the DB URI.

The subject response must expose only the five production subjects. Internal
directories such as `evaluation`, `_needs_ocr`, and `unclassified` are invalid.
The graph manifest and runtime status must expose explicit graph, contract, KG,
and RAG identities; an absent or mismatched identity is a failed deployment.

## 5. PostgreSQL restart and replay

Create a real thread through the web UI, record its `thread_id`, `stream_id`,
and final event ID, then restart PostgreSQL:

```powershell
docker compose restart postgres
docker compose ps
```

After PostgreSQL is healthy, verify:

```powershell
Invoke-WebRequest http://localhost:8000/health/ready -UseBasicParsing
```

1. `GET /threads/{thread_id}/status` returns the same authoritative terminal
   resource or QA result.
2. `GET /streams/{stream_id}` with `Last-Event-ID` replays only later events
   while the journal retention window remains valid.
3. Reusing a request ID with a different payload returns an explicit conflict;
   it must not return the first request's result.
4. A browser refresh restores completed status and download cards for the same
   user/thread namespace.

## 6. Browser canary

Use the real web page and capture screenshots plus machine-readable SSE/status
evidence for exactly these six bounded scenarios:

1. Big-data MapReduce review document.
2. Computer-science data-structure quiz.
3. Machine-learning architecture video script.
4. Mathematics integration/series mind map.
5. Python code practice plus video animation.
6. Big-data and machine-learning study plan plus review document, followed by
   refresh and status recovery.

For every scenario, assert continuous SSE sequence numbers, one authoritative
terminal event, the matching `stream_done`, correct subject/resource identity,
Last-Event-ID replay, explicit request-drift conflict, and no fabricated
download for blocked resources. Ready and evidence-blocked resources are both
valid strict terminal outcomes and must be reported as observed; this smoke
authoring set does not predeclare that a request will have sufficient evidence,
enter repair, or finish ready. Refresh recovery is additionally exercised on
the final scenario. A scenario that cannot be made deterministic with a sealed
fixture remains observational and must not be reported as a stable CI pass.

Run the six authored requests through Chromium. The output directory must be
new or empty:

```powershell
python scripts/run_production_browser_canary.py `
  --project-root . `
  --dataset config/evaluation/private_authoring/evidence_rollout_smoke_dataset.authoring.json `
  --output-dir artifacts/browser_canary/production-close `
  --frontend-url http://localhost:3000 `
  --backend-url http://localhost:8000 `
  --timeout-seconds 1200 `
  --headless
```

The machine-readable report contains identity, sequence, terminal status,
replay, download, refresh, and conflict evidence only. It intentionally omits
generated bodies and Provider payloads. Screenshots are retained separately.

## 7. RAG activation and rollback

Do not activate generation 55 from this runbook. Its engineering comparison
regressed Recall@5 (`0.53 -> 0.41`), MRR, and latency; the top80 experiment made
P95 worse. It may be loaded only by an explicitly pinned inactive canary or
evaluation adapter.

Activation requires a future frozen Gold V3, two independent semantic reviews,
the required chunk review, formal benchmark gates, and a passing page canary.
If a later approved generation is activated, rollback must use the registry
control-plane command explicitly. Never convert a candidate failure into a
Flat success inside a request.

## 8. Shutdown and cleanup

```powershell
docker compose down
```

Do not pass `--volumes` during routine shutdown: it would remove PostgreSQL and
generated-artifact volumes. After all tests finish, ignored compiler/test
caches may be removed. Keep `chroma_store`, generation 55, Flat 53, the
registry, successful reports, and Gold authoring checkpoints.

RAG build and diagnostic details are in the
[Parent-Child RAG runbook](parent_child_rag_local_build.md).
