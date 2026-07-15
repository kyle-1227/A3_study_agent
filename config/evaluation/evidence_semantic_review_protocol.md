# Evidence semantic review protocol v1

This protocol applies to every case and every P0/PG/PR/PGR output in one
evidence-rollout evaluation run. Reviewers inspect the private evaluation
material, while the submitted review bundle remains content-free.

For each exact output fingerprint, record:

1. `claim_count`: every independently checkable answer claim.
2. `supported_claim_count`: claims fully supported by the selected evidence.
3. `fact_count`: every independently checkable factual statement.
4. `ungrounded_fact_count`: factual statements not supported by the selected
   evidence or contradicted by it.

Do not infer scores for missing, truncated, inaccessible, or failed outputs.
Do not copy query text, URLs, evidence bodies, provider bodies, credentials, or
private review material into the review bundle. Use the exact canonical output
fingerprint supplied by the execution record and a one-way reviewer identity
hash. Every review must include a timezone-aware canonical ISO 8601 timestamp
and `assessment_source: human`.

One complete review is required for every expected case/variant slot. A missing
review, a fingerprint mismatch, or a non-human assessment blocks evaluation.
