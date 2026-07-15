# Recommendation Final V1 implementation status

## Scope

This slice implements the strict domain core selected by D8-A. It does not yet
switch the served graph or claim that the explicit-recommendation product path is
live. The active new-RAG task owns the shared graph, stream, and frontend files;
those integration points remain a separate single-writer step.

## Contract

`recommendation_final_v1` is an authoritative recommendation-only terminal. It
does not extend Resource Final V3 and therefore does not weaken the invariant
that Resource Final success contains a real generated resource.

An available final must:

- use `mode=explicit_request`;
- match the current request and authenticated learner identity;
- retain the learning-guidance runtime fingerprint that produced the result;
- contain at least one strictly validated recommendation;
- bind every target to an exact resource/topic/type/title record in the injected
  `KnowledgeGraphV1` artifact;
- carry the knowledge-graph artifact fingerprint and data version;
- carry a digest and count for the full subject catalog inventory;
- carry a content-addressed candidate snapshot, payload hash, and final ID for
  consistency checking (not a cryptographic authenticity signature); and
- omit private profile-signal and history-event identifiers from the public
  payload.

The public JSON contract accepts JSON arrays through two narrowly scoped tuple
fields and stores them as immutable tuples. The public Mapping validator rejects
Python tuples before JSON encoding can normalize them; scalar fields, nested
models, extra fields, and canonical ISO 8601 timestamps remain strict. Existing
model instances are serialized and revalidated rather than trusted. The builder
also derives public summaries deterministically and rejects private profile or
history evidence IDs in public recommendation text.

An unavailable final carries an explicit typed reason and no fake candidate,
generated timestamp, neutral score, or empty-success batch.

## Integration boundary

The following work is intentionally not claimed by this commit:

- a strict Supervisor recommendation action;
- a served explicit-recommendation graph branch;
- `recommendation_final` in `agent_stream_v2`, its journal/replay state machine,
  thread status, or OpenAPI;
- the frontend parser, reducer, and committed recommendation UI; and
- real Provider/PostgreSQL end-to-end proof.

Those pieces must be integrated after the new-RAG single writer stabilizes. The
integration must continue to reject Resource Final V3 recommendation-only
success and must not add a compatibility fallback.

## Verification

The isolated contract slice passed:

- 13 dedicated positive/negative `recommendation_final_v1` tests;
- 101 combined recommendation, learning-guidance, curated-KG, and Resource
  Final V3 contract tests;
- `python -m compileall -q src tests app.py`;
- Ruff check and format for every touched Python file;
- dependency-following scoped mypy for the new public module and its dedicated
  test: 2 files, 0 issues;
- import-linter with the pending D1-A configuration: 331 files, 2,008
  dependencies, 3 contracts kept and 0 broken;
- the repository no-fallback/no-hardcode Semgrep rules under isolated Semgrep
  1.127.0: 6 rules, 2 files, 0 findings;
- scoped Bandit and report-only Vulture: 0 findings;
- Gitleaks 8.24.2 over all changed content: 0 findings; and
- 8 security tests.

The repository-wide pytest run reached `2481 passed, 7 skipped`; its only
failure was the pre-existing launcher test that hardcodes the checkout directory
name `A3_study_agent` while this isolated worktree is named
`A3_study_agent-d8`. Pending cleanup commit `544566a` contains the already
verified cross-worktree correction. No D8 runtime or contract test failed.

The D5-A integration also closes the two prior `arg-type` gaps in
`src/learning_guidance/contracts.py`: validators now reject an absent
`ValidationInfo.field_name` explicitly before calling helpers that require a
string. No cast, default field name, or validation bypass was introduced.

The integrated D5-A source contract and recommendation engine can produce
`no_eligible_candidates`. The explicit-final reason union accepts that state and
derives a fixed public summary for it. The automatic-only
`generated_resources_unavailable` reason remains excluded and is covered by a
negative test.

Whole-repository Ruff remains historical debt on this baseline: 58 lint
findings and 61 files requiring formatting. None is in the touched D8 Python
files. The tooling installation initially exposed an incompatible global
Semgrep/OpenTelemetry dependency set; the global environment was restored and
`pip check` passed, then Semgrep was installed and run in an isolated virtual
environment.
