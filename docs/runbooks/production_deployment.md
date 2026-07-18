# A3 Study Agent production deployment

This runbook is the Docker and canary procedure for the active production
identity. Run commands from the repository root. Never print or commit `.env`
values. Two historical code-practice browser canaries executed against the
active Docker/Provider path and passed their machine-readable checks. They do
not prove the still-incomplete six-scenario, human-content, or post-integration
Docker acceptance.

## 1. Release state

- PostgreSQL is the required production checkpointer. A strictly configured,
  health-checked `AsyncConnectionPool` supplies `AsyncPostgresSaver`; dead
  connections are replaced within the reconnect budget, while setup failure
  still aborts startup and never selects an in-memory saver.
- The served graph uses strict user/thread identity, structured contracts,
  journal replay, status recovery, and explicit resource terminal states.
- Parent-Child generation `pc_20260715_98336c2_55` is sealed `READY` and is the
  active production registry primary. Registry previous and shadow pointers are
  unset, and startup rejects any generation or manifest mismatch.
- The published `main` baseline is `b8f9504`.
- Browser canaries at `707d79806364d95fd300b21d0cb93411f592d67a` are
  historical runtime evidence only.
- SSE `eed2139`, Evidence `4a91f68`, and RAG `f53a710` remain integration
  candidates. Governance and the final Docker rebuild must finish before a
  final integration SHA is declared.
- The sealed generation-manifest fingerprint is
  `db579d40d1f4b79882f495277026e8fccfbfb816fbb150998e47753eec470218`,
  the KnowledgeGraph artifact fingerprint is
  `c504e41ef2e481b30b940ac6cb04f661401f7907d1690efeafc1ed14680fa0b5`, and
  the Evidence orchestration fingerprint is
  `6274c8ac2b0e70828d7e5f64f72ed8f2b9ab36ae8683adcf0b274d60df277b01`.
- The repository-root Flat `chroma_store` and Flat generation 53 are retained
  recovery assets. Production never selects or silently falls back to them.
- Evidence evaluation and adapter binding are V2-only; V1 inputs are rejected.
  PGR is the served path with activation enabled and shadow disabled. Evidence
  gaps may use the initial retrieval plus at most three supplement rounds,
  bounded by 24 search tasks and 72 ledger entries. Required evidence must
  still be complete; partial evidence never becomes a successful resource.
  The six-case dataset remains smoke authoring, not a human-sealed benchmark.
- The served course graph is strict `KnowledgeGraphV1` with five production
  subjects and source-backed topic/resource identity.
- `code_practice` generation is streaming; its strict
  `code_practice_reviewer` has an independent non-streaming model identity.
  Pydantic and business validation remain mandatory in both paths.
- The backend loads `config/rag/index.production.yaml`; the generation fixed by
  `PARENT_CHILD_GENERATION_ID` must match the registry primary. The tracked
  config contains environment-variable names, never secret values.
- The complete backend gate recorded `2880 passed / 7 skipped`; frontend
  SSE `eed2139` recorded 36 files and `208 passed`, with ESLint, typecheck,
  and production build passing. Evidence `4a91f68` recorded 64 passed. RAG
  `f53a710` recorded 48 passed / 1 skipped in the controller gate and
  50 passed / 1 skipped in its lane. Import Linter kept all `3/3` contracts.
  Semgrep and Gitleaks are not installed and were not run.
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
that from Git files. Compose uses long-syntax binds for both course data and the
Parent-Child index with `read_only: true` and
`bind.create_host_path: false`; a missing D:/E: host path therefore fails
instead of creating an empty directory. Generated downloads use a named
`artifacts` volume. The sealed Chroma tree stays read-only; disposable runtime
snapshots use the separate writable `rag_runtime_chroma` volume mounted at its
designated subdirectory.

Set shell-level `A3_ENV_FILE` to the ignored env file's absolute path before
every Compose command. Compose intentionally has no implicit `.env` fallback.

## 3. Build and start the configured active release

Validate the fully interpolated Compose model without rendering it to logs.
Required-variable interpolation fails here without revealing the missing
values; Docker validates the two host directories when it creates mounts:

```powershell
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE config --quiet
```

For a release, build both Compose-default image tags from one clean revision
and attach the same OCI revision label. The local competition frontend must be
compiled for `http://localhost:8000`:

```powershell
$Revision = (git rev-parse HEAD).Trim()
if (git status --porcelain) { throw 'release worktree is dirty' }

docker build `
  --file Dockerfile `
  --target backend `
  --label "org.opencontainers.image.revision=$Revision" `
  --tag "a3_study_agent-backend:$Revision" `
  --tag "a3_study_agent-backend:latest" `
  .

docker build `
  --file Dockerfile `
  --target frontend `
  --build-arg "NEXT_PUBLIC_API_URL=http://localhost:8000" `
  --label "org.opencontainers.image.revision=$Revision" `
  --tag "a3_study_agent-frontend:$Revision" `
  --tag "a3_study_agent-frontend:latest" `
  .

docker image inspect "a3_study_agent-backend:$Revision"
docker image inspect "a3_study_agent-frontend:$Revision"
```

The two inspections must expose `org.opencontainers.image.revision=$Revision`.
Do not print image environment variables or expand the Compose model into logs.

Generation 55 is already configured as the active primary. A routine rebuild or
restart must not call `manage_rag_generation.py`, mutate the registry SQLite
file, or repeat activation. Preserve verified registry/index backups and
recovery images. Startup fails closed if the registry, configured generation,
manifest, KnowledgeGraph, or evidence identity drifts.

Mutable runtime state is separate from read-only course material. Compose mounts
`app_state` at `/app/.runtime_state`; profile and memory SQLite schemas are
initialized before startup succeeds. Legacy `/app/data/profile.db` and
`memory.db` are atomically migrated only when the new target is absent. A
migration or schema failure is fatal, and existing state is never overwritten.
After image replacement, create and read a profile and perform a memory/history
writer round trip before declaring the volume healthy.

Start only from the images already built, then wait for PostgreSQL, backend,
and frontend readiness:

```powershell
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE up --detach --no-build --wait --wait-timeout 900
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps
```

The backend image installs Chromium and ffmpeg because video animation is a
production resource type. Backend and frontend run as separate supervised
containers; either service failure is visible to Compose. The reconnecting
checkpointer pool is a backend runtime property, so a database-only restart
test is invalid if Compose also recreates or restarts backend/frontend.

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
print the DB URI. A recovered readiness probe proves that a newly borrowed pool
connection works; it does not by itself prove that historical checkpoint and
journal state remained readable.

The subject response must expose only the five production subjects. Internal
directories such as `evaluation`, `_needs_ocr`, and `unclassified` are invalid.
The graph manifest and runtime status must expose explicit graph, contract, KG,
and RAG identities; an absent or mismatched identity is a failed deployment.

### 4.1 Bounded recovery quality floors

- Evidence `4a91f68` may perform a bounded reask only for the failed resource+subject
  partition with the same Provider/model. Structured, business, inventory,
  topic, budget, and identity checks remain mandatory, and the reask does not
  itself decide `blocked`; its focused gate recorded 64 passed.
- RAG `f53a710` may split a rerank batch only against the same endpoint and
  must return a complete score for every candidate. RRF-only and partial-score
  results are forbidden; the controller gate recorded 48 passed / 1 skipped,
  and the RAG lane recorded 50 passed / 1 skipped.
- SSE `eed2139` may perform one same-request status read only after transport
  failure or HTTP 410. Only matching `completed`, `failed`, or `stopped`
  status is authoritative. Pending, legacy, sequence-gap, contract-drift, or
  identity mismatch remains a failure, and the client never resubmits the Graph.

No recovery may change provider, model, generation, KnowledgeGraph, or Flat RAG,
and no partial evidence, pending state, empty object, or unreviewed draft may be
reported as success.

## 5. PostgreSQL restart and replay

Use at least two completed real threads and record their `thread_id`,
`stream_id`, terminal payload hash, Context injection count, artifact URL set,
and final event ID. Capture the backend/frontend container IDs, then restart
PostgreSQL only:

```powershell
$BackendBefore = docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps --quiet backend
$FrontendBefore = docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps --quiet frontend
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE restart postgres
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps
$BackendAfter = docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps --quiet backend
$FrontendAfter = docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps --quiet frontend
if ($BackendBefore -ne $BackendAfter -or $FrontendBefore -ne $FrontendAfter) {
  throw 'PostgreSQL-only restart recreated an application container'
}
```

After PostgreSQL is healthy, verify:

```powershell
Invoke-WebRequest http://localhost:8000/health/ready -UseBasicParsing
```

1. `GET /threads/{thread_id}/status` for both historical threads returns the
   same authoritative terminal resource or QA result, payload hash, Context
   injection count, and artifact URL set.
2. `GET /streams/{stream_id}` with `Last-Event-ID` replays only later events
   while the journal retention window remains valid.
3. Reusing a request ID with a different payload returns an explicit conflict;
   it must not return the first request's result.
4. A browser refresh restores completed status and download cards for the same
   user/thread namespace.
5. Every referenced artifact is downloaded again with HTTP 200, an attachment
   disposition, and a non-empty body. For the two recorded code-practice
   canaries this means six files in total and `injection_count=15` per thread.

The test fails if readiness stays unavailable after the configured reconnect
budget, any application container ID changes, any historical identity drifts,
or the backend is manually restarted to recover. Do not diagnose a pass by
restarting the application after PostgreSQL.

## 6. Browser canary

Two consecutive code-practice runs against Evidence fingerprint
`6274c8ac2b0e70828d7e5f64f72ed8f2b9ab36ae8683adcf0b274d60df277b01`
recorded `production_success=true`:

- `artifacts/browser_canary/code-practice-707d798-1-20260717T155617Z/result.json`
- `artifacts/browser_canary/code-practice-707d798-2-20260717T155922Z/result.json`

Both reached `planner -> agent -> reviewer -> output`, produced downloadable
DOCX/Markdown/Python artifacts, passed replay, request-drift, and refresh
recovery checks, and injected 15 context items. Each also observed one transient
thread-status 404 before the new thread existed; the terminal request still
passed. These reports prove one repeated scenario only. Historical, partial, or
failed directories are not acceptance evidence, and the following six-scenario
suite remains incomplete.

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
registry watch. This runbook records two repeated code-practice passes, not a
complete six-scenario or human-content acceptance.

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
