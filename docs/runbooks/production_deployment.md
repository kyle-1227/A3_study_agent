# A3 Study Agent production deployment

This runbook is the Docker and canary procedure for the active production
identity. Run commands from the repository root. Never print or commit `.env`
values. A procedure is not acceptance evidence: this documentation update did
not execute a real Provider, Docker, or browser canary.

## 1. Release state

- PostgreSQL is the required production checkpointer.
- The served graph uses strict user/thread identity, structured contracts,
  journal replay, status recovery, and explicit resource terminal states.
- Parent-Child generation `pc_20260715_98336c2_55` is sealed `READY` and is the
  active production registry primary. Registry previous and shadow pointers are
  unset, and startup rejects any generation or manifest mismatch.
- The repository-root Flat `chroma_store` and Flat generation 53 are retained
  recovery assets. Production never selects or silently falls back to them.
- Evidence evaluation and adapter binding are V2-only; V1 inputs are rejected.
  PGR is the served path with activation enabled and shadow disabled. The
  six-case dataset remains smoke authoring, not a human-sealed benchmark.
- The served course graph is strict `KnowledgeGraphV1` with five production
  subjects and source-backed topic/resource identity.
- The backend loads `config/rag/index.production.yaml`; the generation fixed by
  `PARENT_CHILD_GENERATION_ID` must match the registry primary. The tracked
  config contains environment-variable names, never secret values.
- The latest complete backend gate recorded `2871 passed / 7 skipped`.
  Semgrep and Gitleaks are not installed and were not run. The real active-PGR
  browser canary procedure is documented, but this update claims no live pass.
- This deployment is a trusted local demo. Public multi-tenant authentication,
  tenant isolation, and abuse controls are not closed; do not expose it as a
  public service.

## 2. Required local assets

The Compose deployment requires explicit host locations for separately
supplied, licensed source data and the immutable Parent-Child index. A clean Git
checkout is not self-contained. In `.env`, configure:

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
only when authorized assets were supplied inside this checkout; do not infer
that from Git files. Docker mounts the Parent-Child index read-only and persists
generated downloads in a named `artifacts` volume.
The sealed Chroma tree stays read-only; disposable runtime snapshots use the
separate writable `rag_runtime_chroma` volume mounted at its designated
subdirectory.

Set shell-level `A3_ENV_FILE` to the ignored env file's absolute path before
every Compose command. Compose intentionally has no implicit `.env` fallback.

## 3. Build and start the configured active release

Validate the fully interpolated Compose model without rendering it to logs.
Required-variable interpolation fails here without revealing the missing
values; Docker validates the two host directories when it creates mounts:

```powershell
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE config --quiet
```

Build both images without replacing the currently running containers:

```powershell
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE build
```

Generation 55 is already configured as the active primary. A routine rebuild or
restart must not call `manage_rag_generation.py`, mutate the registry SQLite
file, or repeat activation. Preserve verified registry/index backups and
recovery images. Startup fails closed if the registry, configured generation,
manifest, KnowledgeGraph, or evidence identity drifts.

Start only from the images already built, then wait for PostgreSQL, backend,
and frontend readiness:

```powershell
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE up --detach --no-build --wait --wait-timeout 900
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps
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
must return `health_ready_v3`, `status=ready`, `checkpointer_type=postgres`,
`deployment_mode=active`, `rollout_activation_enabled=true`, and
`rollout_shadow_enabled=false`, plus the graph version, KnowledgeGraph
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
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE restart postgres
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps
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

This section is an execution procedure, not a recorded pass. Historical,
partial, or failed canary directories are not acceptance evidence. Do not
report a live pass until a new complete report succeeds against the final
runtime identities.

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
  --expected-generation-id $ExpectedGenerationId `
  --expected-generation-manifest-fingerprint $ExpectedGenerationManifestFingerprint `
  --timeout-seconds 1200 `
  --headless
```

Set both expected-generation variables explicitly from the independently
validated sealed READY record; never infer them from `/health/ready`. The
machine-readable V3 report binds that generation, the dataset KnowledgeGraph
identity, active/activation-enabled/shadow-disabled state, and matching pre/post
readiness observations. It also contains sequence, terminal, replay, download,
refresh, and conflict evidence only. It intentionally omits generated bodies
and Provider payloads. Screenshots are retained separately.

## 7. Active deployment and recovery boundary

Generation 55 is the explicit production primary and shadow remains disabled.
PGR is the only served evidence path. The six-case suite is still production
smoke rather than formal Gold, and historical benchmark regressions must not be
rewritten as a benchmark pass.

Preserve verified external copies of the sealed generation, registry, Flat 53,
root `chroma_store`, and deployable images. A failed deployment is recovered
only while backend and frontend are stopped: restore the verified external
asset or image, run the same `up --no-build --wait` command, and repeat all
readiness checks. Never convert a request failure into a Flat success or
perform an automatic generation switch. Any registry change requires a backend
restart; the in-process readiness identity must not be treated as a live
registry watch. This runbook records no real-canary pass.

## 8. Shutdown and cleanup

```powershell
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE down
```

Do not pass `--volumes` during routine shutdown: it would remove PostgreSQL and
generated-artifact volumes. After all tests finish, ignored compiler/test
caches may be removed. Keep `chroma_store`, generation 55, Flat 53, the
registry, successful reports, and Gold authoring checkpoints.

RAG build and diagnostic details are in the
[Parent-Child RAG runbook](parent_child_rag_local_build.md).
