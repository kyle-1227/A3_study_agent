# Parent–Child RAG engineering benchmark — 2026-07-15

This report records a provider-backed engineering comparison. It is not the
formal Gold V3 benchmark and cannot authorize Shadow, activation, rollout, or
removal of the legacy RAG path.

## Immutable identities

- Gold dataset: `local_gold_v2`
- Gold SHA-256: `7a023664869daadbe5950f810e31e1f0eee05682c6456cee0d94089f58214529`
- Flat build: `flat_20260715_98336c2_53`
- Flat manifest SHA-256: `c1486b889d8c70ac4d75d3d184e93cfb983f7a532d031bcbb124ab9fe6ac6105`
- Candidate generation: `pc_20260715_98336c2_55`
- Candidate manifest SHA-256: `db579d40d1f4b79882f495277026e8fccfbfb816fbb150998e47753eec470218`
- Candidate Chroma tree SHA-256: `5cfcbef996cdb47d030f4174aa1ef62e72505ba0fe86eb2d352143bb02194303`
- Candidate lifecycle state: `READY`, inactive
- Registry primary, previous, and shadow pointers: unset
- Build code revision: `98336c2339f85790dd2a2c3a1b7fb85f04078d11`

The Flat and Candidate arms used the same embedding fingerprint and all 100
queries. Runtime Chroma access used marker-owned disposable snapshots. The
canonical Flat and Candidate Chroma tree digests were unchanged after the run,
and no runtime snapshot files remained after close.

## Results

| Metric | Flat | Candidate | Candidate delta |
|---|---:|---:|---:|
| Evidence Recall@5 | 0.5300 | 0.4100 | -0.1200, 95% CI [-0.1900, -0.0600] |
| MRR | 0.3916 | 0.3442 | -0.0475, 95% CI [-0.0865, -0.0131] |
| Noise@5 | 0.8900 | 0.8967 | +0.0067, 95% CI [-0.0130, +0.0258] |
| P50 latency | 3348.4 ms | 2635.9 ms | -21.3% |
| P95 latency | 5390.2 ms | 3862.5 ms | -28.3% |
| Mean context tokens | 3031.1 | 1852.9 | ratio 0.611 |

Operational integrity was clean: zero retrieval errors, zero orphan children,
zero generation mismatches, and 447 of 447 requested parents hydrated. The
Candidate was faster and used less context, but its P95 still exceeded the
configured 3000 ms absolute budget. More importantly, Recall@5 and MRR both
regressed with confidence intervals entirely below zero.

## Typed failed attempts and projection fixes

Two non-success attempts were retained as typed, content-free failure
artifacts:

1. Attempt 01 stopped with `ValidationError` because the evaluation projection
   did not explicitly represent an empty domain `section_path`. Commit
   `9967505` maps that valid domain state to the evaluator's explicit `None`
   representation and adds negative/positive tests.
2. Attempt 02 stopped with `KeyError` because the projection attempted to rank
   reranked children whose parents were not selected for final hydration.
   Commit `35fe43a` projects only children that support the final ranked and
   hydrated parents, validates parent assignment, and adds regression tests.

Neither fix adds a fallback, substitutes Flat output for a Candidate failure,
or changes a retrieval score. Partial or failed attempts remain failures.

## Decision and blockers

- `production_recommendation_blocked=true`
- `evaluation_eligible=false`
- `activation_allowed=false`

The current Candidate must remain `READY` and inactive. The legacy RAG must not
be deleted. Before a production decision, Gold V3 still needs five Computer and
seven Python additions, two independent human semantic reviews, the 150-pair
parent/child review, and a checkpointed alternating-order formal benchmark.
The observed retrieval regression additionally requires query-level stage
diagnosis and a fresh retrieval policy/generation; gates and Gold evidence must
not be tuned downward to make this V2 engineering set pass.

The complete private run artifacts remain under
`reports/rag_benchmark/engineering_v2_flat53_pc55_20260715_03/`. They are
gitignored because they include high-cardinality retrieval coordinates. This
versioned report contains no query text, source body, parent body, secret,
authorization header, or provider response body.
