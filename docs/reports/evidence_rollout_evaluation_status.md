# Evidence rollout evaluation control-plane status

Date: 2026-07-15
Branch: `codex/evidence-rollout-eval`
Base: `5c40f20a366b6def0fccc484ff33263577ad4418`

## Outcome

This lane implements the strict evaluation control plane for the canonical
P0/PG/PR/PGR experiment. It does not activate rollout and does not claim a live
four-variant result.

The machine decision is fail-closed. A pass requires all expected case/variant
slots, exact dataset/config/runtime/generation/executor fingerprints, successful
Provider/Parent-Child/Web component states, a human review bound to every exact
output fingerprint, an eligible benchmark result, live execution, and
`activation_enabled: true`. Hermetic execution can compute benchmark metrics but
can never authorize activation.

## Canonical variant contract

| Variant | Resource-aware planning | Bounded evidence repair |
| --- | ---: | ---: |
| P0 | false | false |
| PG | true | false |
| PR | false | true |
| PGR | true | true |

`LiveEvidenceVariantExecutor` accepts explicit adapters in any order, validates
their strict identities against this matrix, and computes its own fingerprint.
It does not alias, infer, copy, or synthesize a missing adapter. An incomplete
inventory returns a blocked decision before any adapter or Provider call, with a
specific reason such as `live_variant_adapter_missing_pg` or
`live_variant_adapter_missing_pr`.

Each production adapter must provide:

1. An `EvidenceLiveAdapterIdentityV2` containing its exact variant factors,
   adapter fingerprint, dataset/KG identities, and the complete ordered
   case/target binding inventory.
2. An async `execute(case, binding)` implementation for only that variant.
3. An `EvidenceVariantAttemptV2` containing either a strict content-free
   observation or typed safe failure metadata.
4. Observations bound to the query digest, dataset, curated KG, current case and
   target fingerprints, initial-evidence scenario identity,
   execution/benchmark/rollout configs, runtime, READY generation manifest,
   executor, and exact variant definition.
5. Counts and weights derived from the authored gold targets, expected source
   routes, and evidence requirements. Raw query text, URLs, evidence bodies,
   Provider bodies, headers, DB URIs, and secrets are forbidden from the public
   observation and report contracts.

## Dedicated evidence dataset

The existing 100-query retrieval QA gold dataset was not relabelled or treated
as evidence-orchestration gold. The new authoring contract requires:

- simple and multi-resource or multi-subject cases;
- an initially-sufficient case partition;
- explicit resource/subject targets;
- an exact KG topic and non-empty ordered KG resource inventory per target;
- target, case, KG artifact, and dataset fingerprints;
- an explicit initial-evidence state, source inventory, and scenario identity;
- expected `parent_child` and/or `web` routes per target;
- human-authored weighted evidence requirements for every target.

The authoring template is
`config/evaluation/evidence_rollout_dataset.template.json`. It deliberately has
no dataset fingerprint and contains replacement markers, so it cannot be loaded
as a sealed production dataset. `seal_evidence_evaluation_dataset()` seals only
an already validated authoring model.

The private six-case smoke draft covers five subjects, all seven generated
resource types, both initial-evidence states, and Parent-Child/Web route shapes.
It is explicitly smoke-only, unsealed, and human-unapproved. Its
initial-evidence fingerprint binds only the declared scenario identity; it does
not prove captured evidence content or semantic sufficiency and cannot satisfy
the activation gate.

## Artifact and report behavior

The hermetic CLI requires explicit paths for every input, including the curated
`KnowledgeGraphV1` artifact. It accepts only an
`execution_mode: hermetic` attempt bundle; a JSON bundle cannot masquerade as
live proof. Inputs must be canonical JSON and all embedded fingerprints are
revalidated. The READY generation manifest and human-review protocol are also
supplied explicitly and digest-checked. The review protocol fingerprint uses
strict UTF-8 text with canonical LF line endings, so Git's Windows CRLF checkout
policy cannot change its identity; lone carriage returns are rejected.

Successful publication atomically creates:

- `activation_decision.json`: complete machine-readable decision;
- `safe_report.json`: closed, content-free summary contract;
- `safe_report.md`: summary rendered only from the safe contract.

Exit code `0` means pass, `1` means a valid fail/blocked decision was published,
and `2` means no trustworthy decision could be formed. Pre-decision failures
produce a typed, content-free `.failure.json`; publication failures are surfaced
on stderr and are not silently swallowed.

## Current live readiness

| Prerequisite | Status | Evidence |
| --- | --- | --- |
| READY generation | ready | `pc_20260715_98336c2_55`; manifest validates as `ready`, `validation_passed=true`, SHA-256 `db579d40d1f4b79882f495277026e8fccfbfb816fbb150998e47753eec470218` |
| Dedicated RAG embedding key | present | `RAG_EMBEDDING_API_KEY=true`, verified externally without reading the value in this worktree |
| Dedicated RAG reranker key | present | `RAG_RERANKER_API_KEY=true`, verified externally without reading the value in this worktree |
| P0 live adapter | missing | Existing graph factory is not yet wrapped in the evaluation adapter contract |
| PG live adapter | missing | No independent production graph/adapter exists |
| PR live adapter | missing | No independent production graph/adapter exists |
| PGR live adapter | missing | Existing graph factory is not yet wrapped in the evaluation adapter contract |
| Curated evidence dataset | blocked | Six-case v2 smoke authoring draft exists but is unsealed, below activation-scale resolution, and lacks human approval |
| Complete human semantic review bundle | missing | No reviewed output bundle exists |
| Rollout activation | disabled | `config/rag/rollout.yaml` remains `activation_enabled: false` |

The secret-bearing `.env` remains only in the main worktree and is not copied,
read, printed, or committed by this lane. A future live launcher must inject the
already configured variables into its child process without persisting them in
the evaluation worktree. The launcher and reports may expose only required
variable names and boolean presence.

The real P0/PG/PR/PGR Provider run is therefore blocked. No Provider call was
attempted in this lane, and no mock, QA-gold substitution, empty result, or old
graph output is counted as live evidence.

## Integration handoff

The service integration lane should implement four concrete adapters against
the protocol above, compute each adapter fingerprint from its exact graph/runtime
composition, and construct the runtime binding only after the executor computes
its fingerprint. It must then:

1. curate and seal the dedicated dataset;
2. execute the exact READY generation with variables injected by the safe
   launcher;
3. obtain complete human semantic reviews bound to exact output fingerprints;
4. rerun/finalize the complete matrix and publish the decision bundle;
5. keep rollout disabled until the live decision is complete and benchmark
   eligible; activation remains a separate explicit control-plane action.

The downstream activation control should require the matching decision
fingerprint in conjunction with the existing candidate validation artifact. It
must not infer eligibility from the Markdown report.
