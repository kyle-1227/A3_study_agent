# Parent–Child RAG retrieval regression diagnosis — 2026-07-15

This is a real provider-backed engineering diagnosis on Gold V2. It is not a
formal Gold V3 benchmark and cannot authorize Shadow, activation, rollout, or
removal of the legacy RAG path.

## Fixed identities

- Gold dataset: `local_gold_v2`
- Gold SHA-256: `7a023664869daadbe5950f810e31e1f0eee05682c6456cee0d94089f58214529`
- Candidate generation: `pc_20260715_98336c2_55`
- Generation manifest SHA-256: `db579d40d1f4b79882f495277026e8fccfbfb816fbb150998e47753eec470218`
- Candidate lifecycle state: `READY`, inactive
- Deployment primary, previous, and shadow pointers: unset
- Query set: 17 fixed queries: 12 prior Recall@5 regressions and one stable
  control per subject
- Base OCR runtime config SHA-256:
  `0bd71d177b95a677c2df1f923acc51069c63689e081d589eac00c23fc8272b2c`
- Top80 runtime config SHA-256:
  `dd830ff96e0a65bf802e04b91e4ac7f11e6f35db68ab9b7bd231e1f0aaa4cc9f`

The candidate config changes exactly one validated runtime field:
`retrieval.reranker_top_n: 20 -> 80`. It does not change or rebuild the sealed
generation. Both runs loaded the exact generation ID and used marker-owned
disposable Chroma snapshots; no deployment pointer was read.

## Stage attribution

| First outcome for each Gold span | Top20 | Top80 |
|---|---:|---:|
| Hydrated window contains Gold | 6 | 8 |
| Excluded by source cap | 7 | 8 |
| Excluded by fusion cutoff | 3 | 0 |
| Child channels miss Gold | 1 | 1 |

Top80 recovered two fusion-cutoff cases. A third fusion-cutoff case moved only
as far as the source-cap exclusion. The five stable controls remained hydrated,
but the dominant source-cap loss was not solved. No additional parameter was
tested after this fixed A/B.

The private canonical reports are gitignored:

- `reports/rag_diagnostics/pc55_v2_17q_top20_20260715_02.json`, SHA-256
  `03f144a7676f560170c1b1bfd65c5772fc33382df01580d1d2a029e23ac01b23`
- `reports/rag_diagnostics/pc55_v2_17q_top80_20260715_01.json`, SHA-256
  `6f6745d3d974fa537454e92d6dc58427cb454cbedd553a99cce480aa9de04fbb`

They contain policy-independent coordinates, IDs, ranks, scores, outcome codes,
and timings. They contain no query text, child or parent body, provider response
body, API key, authorization header, or secret environment value.

## Latency

| Stage | Top20 P50 | Top20 P95 | Top80 P50 | Top80 P95 |
|---|---:|---:|---:|---:|
| Vector | 1539 ms | 2730 ms | 1936 ms | 3811 ms |
| BM25 | 171 ms | 963 ms | 358 ms | 1453 ms |
| Reranker | 625 ms | 1682 ms | 1703 ms | 2875 ms |
| Parent aggregation | 0.8 ms | 1.4 ms | 5.9 ms | 24.8 ms |
| Hydration | 2.4 ms | 3.9 ms | 4.7 ms | 11.0 ms |
| Total | 2679 ms | 4932 ms | 4122 ms | 7234 ms |

Top80 increased total P50 by about 54% and P95 by about 47%. Both arms exceed
the configured 3000 ms absolute P95 budget, and Top80 is materially worse.
Provider timing was not alternated or warm-up corrected, so these values remain
engineering diagnostics; the regression is large enough that this caveat does
not make Top80 production-eligible.

## Decision

- `production_recommendation_blocked=true`
- `evaluation_eligible=false`
- `activation_allowed=false`
- `continue_parameter_tuning=false`

The two additional hydrated matches do not compensate for the latency cost or
the unresolved source-cap and channel losses. No 100-query Top80 run will be
performed, no generation will be rebuilt, and no deployment pointer will be
changed. Generation 55 remains `READY` and inactive. The current legacy RAG is
retained as the explicit rollback path until a future, separately approved
candidate passes frozen Gold V3, two-person review, formal benchmark, and page
canary gates.
