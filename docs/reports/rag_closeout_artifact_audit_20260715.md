# RAG closeout artifact and dead-code audit — 2026-07-15

This was a read-only ownership/reference audit except for the explicitly listed
current-task temporary files. It did not open canonical Chroma, read `.env`, or
change registry deployment pointers.

## Protected assets

Do not delete:

- root `chroma_store`: current legacy rollback asset;
- `indexes/parent_child/generation_registry.sqlite`;
- `indexes/parent_child/pc_20260715_98336c2_55`: current READY, inactive
  Candidate;
- `artifacts/rag/flat_20260715_98336c2_53`: retained Flat comparator;
- the successful engineering benchmark and final Top20/Top80 diagnostic;
- Gold authoring checkpoints, inspections, and review material;
- `embedding_cache`, which remains useful for a future approved build;
- READY generations `pc_20260714_3a41bf4_51` and
  `pc_20260714_f5adeb8_47` until a separate retention decision.

The registry had three READY and six FAILED rows. Primary, previous, and shadow
were all unset, revision was zero, and activation history was empty.

## Deleted current-task temporaries

The closeout removed only files/directories created by the current diagnostic
and proven to be disposable:

- one interrupted Candidate Chroma snapshot whose ownership marker and source
  digest matched Candidate 55 canonical Chroma;
- one interrupted Flat Chroma snapshot whose ownership marker and source digest
  matched Flat 53 canonical Chroma;
- the abandoned source-cap private diagnostic JSON;
- the wrong-config, content-free diagnostic failure marker.

Both runtime roots were empty after cleanup. Canonical Chroma digests were
verified before deletion; neither canonical directory was removed or edited.

## High-confidence cleanup candidates not deleted

Approximately 4.59 GB remains eligible for a separately authorized cleanup:

- six ownership-marked FAILED generation staging directories, about 1.90 GB;
- thirteen Flat directories with no strict manifest or repository reference,
  about 1.67 GB;
- `flat_20260715_98336c2_53_validation_snapshot`, a byte-identical disposable
  validation copy, about 505 MB;
- `flat_20260715_98336c2_52`, identity-equivalent to Flat 53 except for its
  collection name, about 505 MB;
- two RAG-specific pytest temporary directories, about 4.5 MB.

FAILED generation directories must be removed only through
`manage_rag_generation.py --operation cleanup`; manual recursive deletion would
leave registry rows inconsistent. Flat and pytest candidates are not deleted in
this commit because some were produced by earlier tasks and the current branch
does not own their lifecycle decision.

Two older READY generations, a complete-but-failed Flat build, and private LLM
smoke logs are report-only candidates. They require a separate explicit
retention/privacy decision.

## Dead code, scripts, docs, and prompts

- All fourteen RAG CLIs have a test, runbook reference, or explicit operational
  purpose. No RAG CLI was deleted.
- `diagnose_parent_child_regressions.py` was the only new unreferenced entrypoint;
  the parent-child README and runbook now document it.
- `probe_rag_providers.py` has direct CLI tests and shares the provider-probe
  contract used by the local build; it is not dead code.
- No RAG-specific prompt file or concrete generation ID was found in business
  runtime code.
- Vulture exited 3 after reporting only context-manager `__exit__` parameters
  and Pydantic validator `cls` parameters. These are protocol/framework
  signatures, not deletion evidence.
- `rg` was unavailable with Access denied during this audit; repository
  references were cross-checked with Git and PowerShell searches instead.

No legacy RAG logic, fallback path, Graph node, service code, evaluation code,
prompt, or READY generation was deleted.
