# Parent--Child primary: local migration, rebuild, and Docker validation

This runbook applies to the local Docker webpage deployment. It removes sealed
generation publishing from the served path without weakening Pydantic,
business, provenance, provider-identity, or fail-closed checks.

## 1. Preconditions

- Use an explicit environment file outside Git. Do not print or commit it.
- The configured course data directory must be authorized and readable.
- The Parent--Child index root must be writable for a local build or migration.
- Do not delete chroma_store, generation directories, registry files, or
  manifests before the real Docker/browser Canary succeeds.

The backend reads only primary/primary_state.json below the mounted index root.
It fails closed if the state, metadata, validation result, Chroma collection,
BM25 corpora, parent database, policy inventory, subject inventory, vector
dimension, or provider configuration identity is invalid.

## 2. Migrate the existing artifact as primary revision 1

This is a non-destructive copy from the existing Parent--Child artifact
directory. It does not read the registry, READY state, or sealed manifest.

~~~powershell
python scripts/migrate_parent_child_primary.py --project-root . --index-config config/rag/index.production.yaml --source-artifact-identity pc_20260715_98336c2_55 --build-id migrate-primary-r1
~~~

Inspect only non-secret primary identity fields:

~~~powershell
Get-Content -Raw indexes/parent_child/primary/primary_state.json
Get-Content -Raw indexes/parent_child/primary/revisions/r1/primary_metadata.json
Get-Content -Raw indexes/parent_child/primary/revisions/r1/primary_validation.json
~~~

Do not edit any of these JSON files by hand. A missing or invalid file must be
treated as a failed deployment, not repaired by pointing the webpage to old
Chroma storage.

## 3. Rebuild with the new chunk strategy

Use this only after the checked-in chunk policy and course data are correct.
The command writes artifacts to a unique staging directory, validates them, then
atomically switches primary state.

~~~powershell
python scripts/build_parent_child_primary.py --project-root . --index-config config/rag/index.production.yaml --build-id rebuild-20260719 --artifact-identity pc-primary-20260719
~~~

A provider or structure failure leaves the active state unchanged. A failed
staging directory is not served and requires explicit inspection/cleanup; it is
not a fallback source.

## 4. Start the Docker webpage

~~~powershell
Copy-Item .env.example .env
# Fill required values locally; do not paste values into a terminal transcript.
$env:A3_ENV_FILE = (Resolve-Path .env).Path

docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE config --quiet
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE up --detach --build --wait --wait-timeout 420
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps
Invoke-WebRequest http://localhost:8000/health/ready -UseBasicParsing
~~~

`PARENT_CHILD_INDEX_HOST_PATH` must name the `indexes/parent_child` root that
contains `primary`; Compose mounts only that active
`indexes/parent_child/primary` directory read-only. It does not mount
`/app/chroma_store`. The backend creates disposable runtime Chroma snapshots in
the sibling dedicated `rag_runtime_chroma` volume.

The first primary cold start copies the byte-verified Chroma snapshot into that
writable volume and can take several minutes; the 420-second wait allowance is
intentional.

Readiness is valid only when schema_version is health_ready_v4 and it reports:

- parent_child_primary_revision
- parent_child_primary_updated_at
- parent_child_primary_config_fingerprint
- graph and KnowledgeGraph identities
- evidence orchestration identity
- PostgreSQL checkpointer readiness

## 5. Browser Canary and manual verification

Read the expected revision and configuration fingerprint from primary_state.json,
then run the six-case Canary with those non-secret values:

~~~powershell
$canaryTimeoutSeconds = <explicit-per-case-budget-seconds>
python scripts/run_production_browser_canary.py --project-root . --dataset config/evaluation/private_authoring/evidence_rollout_smoke_dataset.authoring.json --output-dir artifacts/canary-primary --frontend-url http://localhost:3000 --backend-url http://localhost:8000 --expected-primary-revision 1 --expected-primary-config-fingerprint <fingerprint-from-primary-state> --timeout-seconds $canaryTimeoutSeconds --headed
~~~

The Canary reads /health/ready both before and after browser interaction. It
rejects a changing revision, timestamp, configuration fingerprint, graph
identity, KnowledgeGraph identity, or evidence orchestration identity.

Set the per-case budget from the checked-in evidence and resource-generation
limits before running the Canary. In particular, supplement rounds, web timeout,
resource generation timeout, and permitted review rounds all contribute to one
case's wall-clock limit. Do not reuse `180` or `420` seconds as a passing proxy:
any timeout, stream error, or identity drift remains a failed Canary.

Then manually use the webpage for at least one:

1. evidence-grounded question and answer;
2. learning-path request;
3. resource-generation request.

Confirm that SSE progresses to a typed terminal event, citations/provenance
contain the primary revision, and local evidence is Parent--Child evidence.

## 6. Post-Canary cleanup gate

Only after the six-case Canary and manual web checks succeed may an authorized
operator remove the old chroma_store, registry, sealed generation artifacts, and
legacy flat-RAG source/tests. Record the exact validation evidence first.

Do not claim this cleanup or the real Canary has completed merely because
static tests, Compose configuration parsing, or /health/ready pass.
