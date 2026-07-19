# A3 Study Agent

[中文](README.md)

A3 Study Agent is a multi-agent learning system for university study. It combines strict learner profiles, learning paths, KnowledgeGraphV1, evidence-constrained answers, resource generation, and recoverable SSE web interaction.

## Active runtime

- Web: Next.js + FastAPI with agent_stream_v2 progress, replay, and thread recovery.
- Learning: learner profiles, paths, assessment, KnowledgeGraphV1, and seven resource types.
- Retrieval: one Parent--Child primary using vector search, BM25, RRF, reranking, and parent hydration.
- Evidence: local and web evidence pass a requirement/evidence judge with bounded repair.
- Persistence: PostgreSQL checkpoints and fail-closed startup/readiness.
- Deployment scope: trusted local Docker web interaction only. This repository does not claim that the real six-case browser Canary or manual teaching acceptance has passed.

## Parent--Child primary

The served backend reads only this layout:

~~~text
indexes/parent_child/
  primary/
    primary_state.json
    revisions/r<revision>/
      primary_metadata.json
      primary_validation.json
      chroma_children/
      parents.sqlite
      bm25/
      policy_manifest.json
      subject_manifest.json
~~~

primary_state.json is the only runtime pointer. It records the revision, update time, configuration fingerprint, and successful structural validation. It does not use a sealed marker, READY state, generation registry, manifest SHA, shadow pointer, previous pointer, or rollback pointer.

A build is written to primary/.staging/<build-id>. The state pointer changes only after strict checks of Chroma, BM25, the parent store, collection dimensions, provider identity, subjects, and chunk policy. A missing or damaged primary blocks backend startup; it never falls back to repository-root chroma_store.

Migrate the existing Parent--Child artifacts into revision 1:

~~~powershell
python scripts/migrate_parent_child_primary.py --project-root . --index-config config/rag/index.production.yaml --source-artifact-identity pc_20260715_98336c2_55 --build-id migrate-primary-r1
~~~

Build later revisions directly with the current chunk policy:

~~~powershell
python scripts/build_parent_child_primary.py --project-root . --index-config config/rag/index.production.yaml --build-id rebuild-20260719 --artifact-identity pc-primary-20260719
~~~

See [the primary local runbook](docs/runbooks/parent_child_primary_local.md). Do not delete historical chroma_store data, generations, or registry files until the real Docker/browser Canary and manual web interaction have passed.

## One-command Docker deployment

Prerequisites: Docker Desktop or Compose v2, an explicit uncommitted environment file, authorized course data, and a built primary index.

~~~powershell
Copy-Item .env.example .env
# Edit .env with secrets, database password, COURSE_DATA_HOST_PATH, and PARENT_CHILD_INDEX_HOST_PATH.
$env:A3_ENV_FILE = (Resolve-Path .env).Path
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE config --quiet
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE up --detach --build --wait --wait-timeout 420
Invoke-WebRequest http://localhost:8000/health/ready -UseBasicParsing
~~~

Required inputs include RAG_EMBEDDING_API_KEY, RAG_RERANKER_API_KEY, POSTGRES_PASSWORD, COURSE_DATA_HOST_PATH, and PARENT_CHILD_INDEX_HOST_PATH. Compose does not mount /app/chroma_store; mutable Chroma snapshots use the separate rag_runtime_chroma volume.

Readiness must return health_ready_v4 with parent_child_primary_revision, parent_child_primary_updated_at, and parent_child_primary_config_fingerprint. The browser Canary fetches readiness twice and rejects any primary identity drift.

The first start copies the read-only primary Chroma into the separate runtime volume and performs strict validation; allow up to 420 seconds for this cold path.

## Quality checks

~~~powershell
python -m py_compile app.py src/schemas.py
ruff check .
ruff format --check .
python -m pytest tests/test_primary_index.py tests/test_app_health.py tests/test_production_browser_canary.py -q
~~~

Run import-linter, a type checker, Semgrep, Bandit, and Gitleaks when available. Missing tools are not passes.

## Competition materials

- [Competition index](docs/competition/README.md)
- [System development](docs/competition/system_development.md)
- [Test report](docs/competition/test_report.md)
- [Deployment guide](docs/competition/deployment_guide.md)
- [Third-party notices](docs/competition/third_party_notices.md)
- [Primary RAG local runbook](docs/runbooks/parent_child_primary_local.md)

## License

The code is released under the [MIT License](LICENSE). Course content, model services, and third-party components remain subject to their own terms.
