# RAG production data request

Date: 2026-07-11

The read-only audit was run with explicit `config/rag/benchmark.yaml`, `config/rag/source_groups.json`, and `data/`. It produced `reports/rag_readiness_current.json` in the ignored runtime-report directory. All five primary subjects are blocked.

| subject | active files | independent sources | pages | valid chars | extractable page ratio | human/history gold | additional sources required | additional gold required |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| big_data | 2 | 2 | 592 | 561,426 | 97.47% | 0 | 1 | 20 |
| computer | 1 | 1 | 93 | 52,395 | 96.77% | 0 | 2 | 20 |
| machine_learning | 1 | 1 | 441 | 368,658 | 97.05% | 0 | 2 | 20 |
| math | 2 | 1 | 581 | 422,451 | 96.21% | 0 | 2 | 20 |
| python | 2 | 2 | 1,327 | 907,596 | 96.53% | 0 | 1 | 20 |

The two mathematics volumes are explicitly grouped as one source series. `_needs_ocr` is excluded from active sources. The configured human, historical-annotated, and synthetic-smoke dataset paths are currently all absent.

Required before a production recommendation:

1. At least three independent sources for every primary subject, with source-series grouping reviewed by a data owner.
2. At least 20 rollout-eligible human or annotated historical queries per subject and at least 50 globally.
3. Gold evidence expressed as source/doc/page/cleaned-character spans plus relevance grades, not generation-specific child IDs.
4. A safely exported, redacted, manually annotated historical query set if one exists.
5. An explicit valid `config/rag/index.yaml` with approved non-secret provider/model/protocol identities and corresponding secret environment variables.

Automatic section/title questions may be supplied only as `synthetic_smoke` with `eligible_for_rollout=false`. Network downloads, fabricated gold, lowered gates, or inferred independent-source groupings are not acceptable substitutes.
