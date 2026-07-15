# Recommendation Final V1 implementation status

## Scope

Decision D8-A is now integrated through the currently served graph, the two
Parent-Child candidate factories, `agent_stream_v2`, thread status, OpenAPI, and
the main frontend. The implementation remains isolated on
`codex/streaming-context-v3-integration`; it has not been pushed or merged to
main and does not activate the new-RAG rollout.

Implementation commits:

- `47a9e77` and `a462963`: strict Recommendation Final V1 domain contract and
  `no_eligible_candidates` alignment;
- `c8e20f8`: source-backed production learning-guidance catalog;
- `225b084`: strict frontend parser, LiveTurn commit/replay, status restore, and
  recommendation card; and
- `9d6d69f`: Supervisor/graph/runtime, strict terminal arbitration,
  SSE/journal/status/OpenAPI, and checkpoint fail-closed integration.

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

## Integrated runtime behavior

- Supervisor accepts only the explicit recommendation shape:
  `intent=academic`, `response_mode=recommendation`, empty QA/resource fields,
  and no live-verification flag.
- The route is physically
  `resource_recommendation_explicit -> recommendation_final_output -> END`; it
  does not enter RAG, resource generation, or Resource Final V3.
- Missing learner identity takes precedence over subject diagnostics. A missing
  subject and a multi-subject request remain distinct typed unavailable states.
  A same-thread workspace continuation may supply one already-bound subject,
  but an unbound or mismatching continuation fails closed.
- `recommendation_final` is a first-class authoritative terminal in the stream
  sequencer, capacity reservation, journal replay, reconnect, active-run state,
  checkpoint status, and `stream_done.terminal_type`.
- The frontend revalidates the final ID, payload hash, runtime identity, catalog
  snapshot, and terminal branch before committing. Provisional text is never
  committed as a recommendation.
- QA final arbitration was hardened at the same boundary: current-request QA
  now revalidates strict JSON shape, runtime thread/request, business rules,
  payload hash, and QA ID. A malformed QA cannot disappear and allow another
  terminal to win.
- A completed run is no longer published when the checkpoint completion write
  fails. QA, Resource, and Recommendation finals all produce the typed
  non-completed terminal `terminal_checkpoint_persist_failed` instead.
- All five SSE endpoints document only `text/event-stream` for HTTP 200.

Resource Final V3 still rejects recommendation-only success; no compatibility
adapter, fallback model, neutral score, generated-resource substitute, or
silent default was added.

## Remaining boundary

The recommendation route is live at the code level, but production `available`
results are not yet claimed. The D5 profile/history adapters are strict readers,
while the current application still lacks authoritative writers for
`profile.extra.learning_guidance_v1` and episodic
`metadata.learning_guidance_v1`. The existing episodic writer also identifies
records by `thread_id` before `user_id`, which does not match guidance history
lookup. Until those writer contracts and a real user E2E exist, the route will
correctly return typed `profile_unavailable` or `history_unavailable` for users
without source-backed guidance data.

The following remain owned by the new-RAG production gate rather than D8:
production Parent-Child index/gold/provider evidence, rollout activation,
served-graph replacement, checkpoint clear cutover, and deletion of the old
served graph/node IDs.

## Verification

Current integrated results:

- D8 graph/QA/stream/status/recovery focused suite: `654 passed`;
- full backend: `2590 passed, 7 skipped`; warnings are third-party deprecations,
  one existing aiosqlite closed-loop test-thread warning, and two AsyncMock
  warnings that were subsequently removed and rechecked in the focused module;
- frontend: `31 files / 142 tests`, TypeScript typecheck, full ESLint, and Next
  production build all passed; the production route list has no `/volunteer`;
- `python -m compileall -q src tests app.py`, `git diff --check`, and scoped Ruff
  check/format for all 36 touched Python files passed;
- scoped mypy passed 12 touched source files with no issue;
- import-linter analyzed 333 files and 2,051 dependencies: 3 contracts kept,
  0 broken;
- Gitleaks 8.24.2 scanned the staged 83.39 KB diff and found no leak;
- actual no-checkpointer application startup compiled 24 physical nodes and a
  non-empty graph version with the learning-guidance runtime injected; and
- the OpenAPI gate confirmed the five agent-stream routes expose only SSE for
  their success response.

Repository-wide Ruff remains historical debt: 45 lint findings and 57 files
requiring formatting, none in this touched set. Scoped Bandit reports two
pre-existing findings (`0.0.0.0` development binding and an old broad-exception
continue in run-control), so it is not recorded as passing. Isolated Semgrep
1.127.0 ran 8 rules over 13 touched source files and reported 55 broad lexical
findings, all from pre-existing `fallback` text/sanitizer parameters or comments;
the added diff contains no fallback/provider-hardcode token, but this scan is
also not recorded as a clean pass.

The D5-A integration also closes the two prior `arg-type` gaps in
`src/learning_guidance/contracts.py`: validators now reject an absent
`ValidationInfo.field_name` explicitly before calling helpers that require a
string. No cast, default field name, or validation bypass was introduced.

The integrated D5-A source contract and recommendation engine can produce
`no_eligible_candidates`. The explicit-final reason union accepts that state and
derives a fixed public summary for it. The automatic-only
`generated_resources_unavailable` reason remains excluded and is covered by a
negative test. Semgrep was installed in an isolated tooling environment with
`setuptools<81`; `pip check` reports no broken dependency, and the
project/global Python environments were not modified.
