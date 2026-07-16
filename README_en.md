# A3 Study Agent

[中文](README.md)

A3 Study Agent is a multi-agent learning system for university study. It combines strict learner profiles, learning paths, a curated course knowledge graph, Parent-Child RAG, web research, evidence judgement, and seven resource generators in a recoverable streaming experience.

## Current production-convergence state

| Area | State |
| --- | --- |
| Web/API | Next.js + FastAPI with `agent_stream_v2` SSE, status recovery, replay, and explicit terminal events |
| State and identity | PostgreSQL checkpoints; strict user, thread, request, dataset, and case binding |
| Course graph | `KnowledgeGraphV1`, five subjects, source-backed topic/resource identity |
| New RAG | this release config pins sealed `READY` generation `pc_20260715_98336c2_55` and the resource-aware PGR path; final runtime verification is authoritative |
| RAG deployment | registry primary is configured as generation 55 and previous / shadow are unset; activation, manifest, and served identity require final `health_ready_v3` / manifest verification |
| Evaluation | Evidence is V2-only and V1 is rejected; P0 / PG / PR / PGR real-node adapters are evaluation variants, while the six-case dataset remains smoke authoring rather than formal Gold |
| Quality gate | the latest complete backend gate recorded `2871 passed / 7 skipped`; Semgrep and Gitleaks are not installed and were not run |
| Live canary | the active-PGR browser canary is being rerun; final acceptance must not yet be claimed |
| Deployment boundary | this is a trusted local demo; public multi-tenant authentication and tenant isolation are not closed |
| Rollback | repository-root `chroma_store` and Flat 53 must remain in this release; later cleanup requires separate approval |

`READY` proves artifact integrity only. Production startup additionally requires the registry primary and `PARENT_CHILD_GENERATION_ID` to name the same generation, an empty shadow pointer, and the exact manifest identity. A request fails fast; it never switches to Flat RAG after an error. Flat 53 and the root `chroma_store` remain offline recovery assets, not request-time fallbacks.

## Capabilities

- Strict onboarding, learner-profile, learning-history, and assessment binding.
- Learning-path planning validated against source-backed KnowledgeGraph topics.
- Parallel single-subject, multi-subject, and multi-resource orchestration.
- Parent-Child Vector + BM25 + RRF + reranker + parent hydration.
- Strict local/web requirement, judgement, and bounded-repair evidence loops.
- P0 (no planning/no repair), PG (planning/no repair), PR (no planning/repair), and PGR (planning/repair) evaluation adapters; they are not four served traffic variants.
- Study plan, mind map, quiz, review document, code practice, video script, and video animation resources.
- SSE `EvidenceProgress`, Last-Event-ID replay, thread-status recovery, and persistent downloads.

## Architecture

```mermaid
flowchart LR
    UI[Next.js Web] -->|agent_stream_v2| API[FastAPI]
    API --> ID[Strict user/thread/request binding]
    ID --> SUP[Supervisor]
    SUP --> QA[QA path]
    SUP --> LP[Learner path planner]
    LP --> EP[Resource evidence planner]
    EP --> LR[Parent-Child local retrieval]
    EP --> WR[Web research]
    LR --> J[Requirement evidence judge]
    WR --> J
    J -->|bounded repair| EP
    J --> RG[Parallel resource generation]
    RG --> FINAL[Authoritative resource final]
    QA --> FINAL
    API <--> PG[(PostgreSQL checkpoints)]
    LR --> PC[(READY generation 55)]
```

Provider, model, base URL, API-key environment name, and retry policy come from strict configuration. Business nodes do not hardcode them and do not silently switch Provider, model, or RAG path after failure.

## One-command Docker deployment

Requirements: Docker Desktop / Docker Engine, Compose v2, local course data, and the sealed Parent-Child index.

```powershell
if (-not (Test-Path -LiteralPath '.env')) {
  Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
# Populate secrets, a strong DB password, and the two host asset paths.
$env:A3_ENV_FILE = (Resolve-Path '.env').Path

docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE config --quiet
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE up --detach --build --wait
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps
```

Required settings:

- shell selector `A3_ENV_FILE` (absolute path to the ignored env file)
- `DEEPSEEK_API_KEY`
- `RAG_EMBEDDING_API_KEY`
- `RAG_RERANKER_API_KEY`
- `TAVILY_API_KEY`
- `POSTGRES_PASSWORD`
- `NEXT_PUBLIC_API_URL`
- `COURSE_DATA_HOST_PATH`
- `PARENT_CHILD_INDEX_HOST_PATH`
- `PARENT_CHILD_GENERATION_ID`

Compose supervises backend, frontend, and PostgreSQL separately. The sealed Parent-Child index is mounted read-only, `.runtime_chroma` has a dedicated writable volume, and generated downloads use the persistent `artifacts` volume. Chromium and ffmpeg are included for real video-animation output.

Verify startup:

```powershell
Invoke-WebRequest http://localhost:8000/health/live -UseBasicParsing
Invoke-WebRequest http://localhost:8000/health/ready -UseBasicParsing
Invoke-WebRequest http://localhost:8000/graph/manifest -UseBasicParsing
Invoke-WebRequest http://localhost:8000/subjects -UseBasicParsing
Invoke-WebRequest http://localhost:3000 -UseBasicParsing
```

`/health/ready` must return `health_ready_v3`, `status=ready`, `checkpointer_type=postgres`, `deployment_mode=active`, `rollout_activation_enabled=true`, and `rollout_shadow_enabled=false`, together with the graph, KnowledgeGraph, generation-manifest, and evidence-orchestration identities. Any missing or mismatched identity is a failed deployment.

See the [production deployment runbook](docs/runbooks/production_deployment.md) for PostgreSQL restart/replay, the six-scenario Playwright canary, and rollback boundaries.

## Local development

Python 3.11+ and Node.js 20.12+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,quality]"
if (-not (Test-Path -LiteralPath '.env')) {
  Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
# Populate .env; strict local startup also requires PostgreSQL, secrets,
# course data, and the sealed index.

Push-Location frontend
npm ci
Pop-Location

python -m scripts.run_backend --no-reload --host 0.0.0.0 --port 8000
```

In another terminal:

```powershell
Push-Location frontend
npm run dev
```

Parent-Child builds, Gold authoring, diagnostics, and registry operations require explicit arguments. Follow the [Parent-Child RAG runbook](docs/runbooks/parent_child_rag_local_build.md).

## Quality gates

Run the complete matrix once after integration; use focused related tests while developing.

```powershell
python -m compileall -q src tests app.py
ruff check .
ruff format --check .
python -m pytest -q
lint-imports --config .importlinter
bandit -r src -x tests

Push-Location frontend
npm run test
npm run typecheck
npm run lint
npm run build
Pop-Location
```

The recorded complete backend result is `2871 passed / 7 skipped`. Semgrep and Gitleaks are not installed and were not run, so they must not be reported as passing. The real browser canary is still being rerun and cannot be replaced by unit-test evidence.

## Repository layout

```text
app.py                     FastAPI, SSE, status/replay, and artifact APIs
frontend/                  Next.js web client
src/graph/                 Served graph, evidence loop, and resource nodes
src/learning_guidance/     KnowledgeGraph, profile/history, and path contracts
src/rag/parent_child/      Generation, retrieval, hydration, and runtime
src/evaluation/            P0/PG/PR/PGR rollout evaluation
config/                    Strict runtime configuration and prompts
scripts/                   Build, diagnostics, evaluation, and deployment tools
tests/                     Backend, contract, security, and integration tests
docs/runbooks/             Production and RAG operations
```

## Important limits

- Do not present the six-case smoke dataset as formal Gold or completed human review.
- Do not delete the legacy RAG, Flat 53, generation 55, registry, successful reports, or Gold checkpoints.
- Do not expose API keys, Authorization, full DB URIs, or Provider bodies in reports, traces, screenshots, or commands.
- Do not turn a Candidate failure into a false legacy-RAG success; rollback is explicit only.
- This deployment is for a trusted local demo only. Do not expose it publicly until multi-tenant authentication, tenant isolation, and abuse controls are closed.

## License

See [LICENSE](LICENSE).
