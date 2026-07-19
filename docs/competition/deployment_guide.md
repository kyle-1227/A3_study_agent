# A3 Study Agent deployment guide

## Current deployment contract

The local Docker webpage serves one Parent--Child primary only. The backend
loads primary/primary_state.json under the mounted indexes/parent_child root,
then validates its metadata, validation result, Chroma children, BM25 corpora,
parent store, provider identity, vector shape, subjects, and chunk policy.

The deployment does not use READY, sealed manifests, generation registry
activation, shadow, previous, rollback, PARENT_CHILD_GENERATION_ID, or a
repository-root chroma_store mount. A damaged or absent primary blocks backend
startup; it does not trigger a legacy retrieval fallback.

This is a trusted local-demo deployment. Do not expose it as a public
multi-tenant service without separate authorization, tenancy, and abuse-control
work.

## Prerequisites

- Docker Desktop or Docker Engine with Compose v2.
- An explicit environment file that is not committed.
- Authorized course data.
- A built Parent--Child primary under the configured index root.

Required environment inputs include:

- RAG_EMBEDDING_API_KEY
- RAG_RERANKER_API_KEY
- POSTGRES_PASSWORD
- COURSE_DATA_HOST_PATH
- PARENT_CHILD_INDEX_HOST_PATH
- NEXT_PUBLIC_API_URL

## Create or migrate primary

To copy the existing Parent--Child artifact into primary revision 1:

~~~powershell
python scripts/migrate_parent_child_primary.py --project-root . --index-config config/rag/index.production.yaml --source-artifact-identity pc_20260715_98336c2_55 --build-id migrate-primary-r1
~~~

To build a new revision with the current chunk policy:

~~~powershell
python scripts/build_parent_child_primary.py --project-root . --index-config config/rag/index.production.yaml --build-id rebuild-20260719 --artifact-identity pc-primary-20260719
~~~

Both paths build into primary/.staging and atomically publish only after strict
structural validation. A failure leaves the previously active primary state
unchanged.

## Start

~~~powershell
Copy-Item .env.example .env
# Edit .env locally. Never commit or print secret values.
$env:A3_ENV_FILE = (Resolve-Path .env).Path

docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE config --quiet
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE up --detach --build --wait --wait-timeout 420
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps
~~~

Compose uses separate backend, frontend, and PostgreSQL services.
`PARENT_CHILD_INDEX_HOST_PATH` names the `indexes/parent_child` root that
contains `primary`; Compose mounts only its active `primary` directory
read-only, keeps runtime Chroma snapshots in a separate writable sibling volume,
and does not mount /app/chroma_store.

The first primary cold start copies and validates the byte-verified Chroma
snapshot in that writable volume; allow the documented 420-second wait window.

## Verify

~~~powershell
Invoke-WebRequest http://localhost:8000/health/live -UseBasicParsing
Invoke-WebRequest http://localhost:8000/health/ready -UseBasicParsing
Invoke-WebRequest http://localhost:3000 -UseBasicParsing
~~~

Readiness must return health_ready_v4 and identify the active primary through:

- parent_child_primary_revision
- parent_child_primary_updated_at
- parent_child_primary_config_fingerprint

## Browser Canary and cleanup

Run the six-case browser Canary with the revision and fingerprint from
primary_state.json:

~~~powershell
$canaryTimeoutSeconds = <explicit-per-case-budget-seconds>
python scripts/run_production_browser_canary.py --project-root . --dataset config/evaluation/private_authoring/evidence_rollout_smoke_dataset.authoring.json --output-dir artifacts/canary-primary --frontend-url http://localhost:3000 --backend-url http://localhost:8000 --expected-primary-revision 1 --expected-primary-config-fingerprint <fingerprint-from-primary-state> --timeout-seconds $canaryTimeoutSeconds --headed
~~~

The Canary reads readiness before and after interaction and rejects primary
identity drift. Then manually verify a question/answer, a learning path, and
resource generation in the webpage.

Only after both checks truly succeed may an authorized operator remove old
chroma_store data, historical generations, registry files, and legacy flat-RAG
implementation. Static tests and a healthy endpoint do not prove that Canary.
