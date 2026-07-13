# Local RAG Build Report Template

Use this template for a local, provider-backed experimental build. Do not put
API-key values, Authorization headers, raw provider bodies, or full course text
in the report. Preserve the real failed stage; never substitute baseline results
or an older generation.

## Identity and local paths

- Commit:
- Runtime config: `config/rag/index.runtime.yaml`
- Build ID / generation ID / run ID:
- Experimental only: `true`
- Activation prohibited: `true`

## Secret presence

For each variable, record only `present=true` or `present=false`:

- `RAG_EMBEDDING_API_KEY`
- `RAG_RERANKER_API_KEY`
- `OPENROUTER_API_KEY`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

## Evidence to report

- Embedding probe: provider/model/endpoint fingerprint, actual dimension,
  input-type and batch support, repeat consistency, latency, failure type.
- Reranker probe: provider/model/endpoint fingerprint, index coverage, score
  range, relevant-versus-irrelevant ordering, latency, failure type.
- Chat probe: model fingerprint, non-empty output SHA-256, latency, failure.
- Corpus/chunks: discovered subjects, source-group completeness, per-subject
  parent/child counts, total vectors, batches, empty/orphan/over-limit counts.
- Artifacts: Flat Chroma count/path; generation ID, `READY`, inactive state;
  manifest, BM25, Parent Store, Chroma, and hydration checks.
- Smoke: per-subject retrieval evidence and grounded QA citation/support review.
- Readiness: readiness status, `evaluation_eligible`, activation allowed=false,
  failed stage and all remaining blockers.
