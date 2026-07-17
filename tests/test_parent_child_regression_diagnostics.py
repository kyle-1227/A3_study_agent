from __future__ import annotations

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from scripts.diagnose_parent_child_regressions import (
    RegressionDiagnosisCliError,
    _load_subset,
    _runtime_policy,
    _select_ready_generation,
    _select_queries,
    _write_failure,
    main,
)
from src.rag.parent_child._storage_io import model_json_bytes
from src.rag.parent_child.evaluation import GoldDataset, GoldEvidenceSpan, GoldQuery
from src.rag.parent_child.regression_diagnostics import (
    RegressionQuerySubset,
    build_regression_report,
    diagnose_gold_query,
)
from src.rag.parent_child.retrieval import (
    HybridRetrievalDiagnosticTrace,
    HybridRetrievalPolicy,
    RetrievalDiagnosticChildCoordinate,
    RetrievalDiagnosticHydrationCoordinate,
    RetrievalDiagnosticParentCoordinate,
    RetrievalDiagnosticTimings,
    RetrievalDiagnosticWindowCoordinate,
    compute_retrieval_fingerprint,
)


DOC_ID = "doc_" + "1" * 40
OTHER_DOC_ID = "doc_" + "2" * 40
SOURCE = "math/source.md"
FINGERPRINT = "a" * 64
MANIFEST_SHA = "b" * 64


def _policy(*, reranker_top_n: int) -> HybridRetrievalPolicy:
    return HybridRetrievalPolicy(
        schema_version="hybrid_retrieval_policy_v1",
        generation_manifest_sha256=MANIFEST_SHA,
        embedding_fingerprint="c" * 64,
        bm25_tokenizer_fingerprint="d" * 64,
        reranker_fingerprint="e" * 64,
        vector_top_k=50,
        bm25_top_k=50,
        vector_rrf_weight=1.0,
        bm25_rrf_weight=1.0,
        rrf_k=20,
        reranker_top_n=reranker_top_n,
        reranker_transport_fallback_mode="disabled",
        unique_parent_top_k=5,
        max_children_per_parent=1,
        max_parents_per_source=2,
        parent_support_lambda=0.5,
        full_parent_max_chars=2400,
        hit_window_chars_per_side=200,
        multi_subject_per_subject_top_k=5,
        multi_subject_max_parents=10,
        subject_coverage_quota=1,
    )


def _gold() -> GoldQuery:
    return GoldQuery(
        schema_version="gold_query_v1",
        query_id="q-diagnostic",
        subject="math",
        query="private query text must not be persisted",
        dataset_kind="human_gold",
        eligible_for_rollout=False,
        gold_spans=(
            GoldEvidenceSpan(
                schema_version="gold_evidence_span_v1",
                gold_span_id="gold_diagnostic",
                source_group_id="math_group",
                source_relpath=SOURCE,
                doc_id=DOC_ID,
                pagination_kind="logical",
                page_start=1,
                page_end=1,
                start_char=100,
                end_char=120,
                section_path=("Section",),
                relevance_grade=3,
            ),
        ),
    )


def _dataset(query_count: int) -> GoldDataset:
    return GoldDataset(
        schema_version="gold_dataset_v1",
        dataset_id="diagnostic-dataset",
        queries=tuple(
            GoldQuery(
                schema_version="gold_query_v1",
                query_id=f"q-{index:02d}",
                subject="math",
                query=f"private query {index}",
                dataset_kind="human_gold",
                eligible_for_rollout=False,
                gold_spans=(
                    GoldEvidenceSpan(
                        schema_version="gold_evidence_span_v1",
                        gold_span_id=f"gold_{index:02d}",
                        source_group_id="math_group",
                        source_relpath=SOURCE,
                        doc_id=DOC_ID,
                        pagination_kind="logical",
                        page_start=1,
                        page_end=1,
                        start_char=index * 20,
                        end_char=index * 20 + 10,
                        section_path=("Section",),
                        relevance_grade=3,
                    ),
                ),
            )
            for index in range(query_count)
        ),
    )


def _child(
    *,
    child_id: str,
    parent_id: str,
    doc_id: str,
    start_char: int,
    end_char: int,
    fusion_rank: int,
    reranker_rank: int | None,
) -> RetrievalDiagnosticChildCoordinate:
    return RetrievalDiagnosticChildCoordinate(
        schema_version="retrieval_diagnostic_child_coordinate_v1",
        child_id=child_id,
        parent_id=parent_id,
        doc_id=doc_id,
        source_relpath=SOURCE,
        pagination_kind="logical",
        page_start=1,
        page_end=1,
        start_char=start_char,
        end_char=end_char,
        section_path=("Section",),
        vector_rank=fusion_rank,
        bm25_rank=None,
        fusion_rank=fusion_rank,
        submitted_to_reranker=reranker_rank is not None,
        reranker_rank=reranker_rank,
        vector_raw_score=0.9,
        bm25_raw_score=None,
        rrf_score=0.05,
        reranker_score=None if reranker_rank is None else 0.8,
    )


def _parent(
    *,
    parent_id: str,
    pre_cap_rank: int,
    selected_rank: int | None,
    selection_outcome: str,
    all_child_ids: tuple[str, ...],
    supporting_child_ids: tuple[str, ...],
) -> RetrievalDiagnosticParentCoordinate:
    return RetrievalDiagnosticParentCoordinate.model_validate(
        {
            "schema_version": "retrieval_diagnostic_parent_coordinate_v1",
            "parent_id": parent_id,
            "subject": "math",
            "source_relpath": SOURCE,
            "pre_cap_rank": pre_cap_rank,
            "selected_rank": selected_rank,
            "selection_outcome": selection_outcome,
            "parent_score": 0.8,
            "best_child_rank": 1,
            "all_child_ids": all_child_ids,
            "supporting_child_ids": supporting_child_ids,
        }
    )


def _hydrated(
    *,
    parent_id: str,
    selected_rank: int,
    doc_id: str,
    window_start: int,
    window_end: int,
) -> RetrievalDiagnosticHydrationCoordinate:
    return RetrievalDiagnosticHydrationCoordinate(
        schema_version="retrieval_diagnostic_hydration_coordinate_v1",
        parent_id=parent_id,
        selected_rank=selected_rank,
        doc_id=doc_id,
        source_relpath=SOURCE,
        pagination_kind="logical",
        page_start=1,
        page_end=1,
        start_char=0,
        end_char=500,
        expansion_mode="hit_window",
        windows=(
            RetrievalDiagnosticWindowCoordinate(
                schema_version="retrieval_diagnostic_window_coordinate_v1",
                start_char=window_start,
                end_char=window_end,
            ),
        ),
    )


def _trace(
    *,
    children: tuple[RetrievalDiagnosticChildCoordinate, ...],
    parents: tuple[RetrievalDiagnosticParentCoordinate, ...],
    hydrated: tuple[RetrievalDiagnosticHydrationCoordinate, ...],
    retrieval_fingerprint: str = FINGERPRINT,
) -> HybridRetrievalDiagnosticTrace:
    return HybridRetrievalDiagnosticTrace(
        schema_version="hybrid_retrieval_diagnostic_trace_v1",
        status="ok",
        request_id="q-diagnostic",
        ranking_mode="reranked",
        fallback_reason_code=None,
        subject="math",
        generation_id="generation-ready",
        retrieval_fingerprint=retrieval_fingerprint,
        children=children,
        parents=parents,
        hydrated_parents=hydrated,
        timings=RetrievalDiagnosticTimings(
            schema_version="retrieval_diagnostic_timings_v1",
            vector_ms=1.0,
            bm25_ms=2.0,
            fusion_ms=0.1,
            reranker_ms=3.0,
            parent_aggregation_ms=0.2,
            hydration_ms=0.3,
            total_ms=6.6,
        ),
    )


def test_diagnostics_distinguish_channel_fusion_and_reranker_losses() -> None:
    unrelated = _child(
        child_id="unrelated",
        parent_id="parent-unrelated",
        doc_id=OTHER_DOC_ID,
        start_char=10,
        end_char=20,
        fusion_rank=1,
        reranker_rank=1,
    )
    unrelated_parent = _parent(
        parent_id="parent-unrelated",
        pre_cap_rank=1,
        selected_rank=1,
        selection_outcome="selected",
        all_child_ids=("unrelated",),
        supporting_child_ids=("unrelated",),
    )
    unrelated_hydrated = _hydrated(
        parent_id="parent-unrelated",
        selected_rank=1,
        doc_id=OTHER_DOC_ID,
        window_start=10,
        window_end=20,
    )

    miss = diagnose_gold_query(
        query=_gold(),
        trace=_trace(
            children=(unrelated,),
            parents=(unrelated_parent,),
            hydrated=(unrelated_hydrated,),
        ),
    )
    assert miss.gold_spans[0].outcome == "child_retrieval_miss"

    fused_out = _child(
        child_id="target",
        parent_id="parent-target",
        doc_id=DOC_ID,
        start_char=100,
        end_char=120,
        fusion_rank=2,
        reranker_rank=None,
    )
    cutoff = diagnose_gold_query(
        query=_gold(),
        trace=_trace(
            children=(unrelated, fused_out),
            parents=(unrelated_parent,),
            hydrated=(unrelated_hydrated,),
        ),
    )
    assert cutoff.gold_spans[0].outcome == "fusion_cutoff"

    demoted = _child(
        child_id="target",
        parent_id="parent-target",
        doc_id=DOC_ID,
        start_char=100,
        end_char=120,
        fusion_rank=1,
        reranker_rank=2,
    )
    sibling = _child(
        child_id="sibling",
        parent_id="parent-target",
        doc_id=DOC_ID,
        start_char=200,
        end_char=220,
        fusion_rank=2,
        reranker_rank=1,
    )
    target_parent = _parent(
        parent_id="parent-target",
        pre_cap_rank=1,
        selected_rank=1,
        selection_outcome="selected",
        all_child_ids=("sibling", "target"),
        supporting_child_ids=("sibling",),
    )
    reranker_loss = diagnose_gold_query(
        query=_gold(),
        trace=_trace(
            children=(demoted, sibling),
            parents=(target_parent,),
            hydrated=(
                _hydrated(
                    parent_id="parent-target",
                    selected_rank=1,
                    doc_id=DOC_ID,
                    window_start=200,
                    window_end=220,
                ),
            ),
        ),
    )
    assert reranker_loss.gold_spans[0].outcome == "reranker_demotion"
    assert reranker_loss.gold_spans[0].reranker_rank_worsened is True

    target_without_rank_loss = _child(
        child_id="target",
        parent_id="parent-target",
        doc_id=DOC_ID,
        start_char=100,
        end_char=120,
        fusion_rank=2,
        reranker_rank=2,
    )
    leading_sibling = _child(
        child_id="sibling",
        parent_id="parent-target",
        doc_id=DOC_ID,
        start_char=200,
        end_char=220,
        fusion_rank=1,
        reranker_rank=1,
    )
    aggregation_loss = diagnose_gold_query(
        query=_gold(),
        trace=_trace(
            children=(leading_sibling, target_without_rank_loss),
            parents=(target_parent,),
            hydrated=(
                _hydrated(
                    parent_id="parent-target",
                    selected_rank=1,
                    doc_id=DOC_ID,
                    window_start=200,
                    window_end=220,
                ),
            ),
        ),
    )
    assert aggregation_loss.gold_spans[0].outcome == "parent_aggregation"
    assert aggregation_loss.gold_spans[0].reranker_rank_worsened is False


@pytest.mark.parametrize(
    ("selection_outcome", "selected_rank", "expected"),
    [
        ("source_cap", None, "source_cap"),
        ("unique_parent_cap", None, "unique_parent_cap"),
        ("selected", 1, "hydration_omission"),
    ],
)
def test_diagnostics_distinguish_parent_caps_and_hydration_omission(
    selection_outcome: str,
    selected_rank: int | None,
    expected: str,
) -> None:
    target = _child(
        child_id="target",
        parent_id="parent-target",
        doc_id=DOC_ID,
        start_char=100,
        end_char=120,
        fusion_rank=1,
        reranker_rank=1,
    )
    parent = _parent(
        parent_id="parent-target",
        pre_cap_rank=1,
        selected_rank=selected_rank,
        selection_outcome=selection_outcome,
        all_child_ids=("target",),
        supporting_child_ids=("target",),
    )

    diagnostic = diagnose_gold_query(
        query=_gold(),
        trace=_trace(children=(target,), parents=(parent,), hydrated=()),
    )

    assert diagnostic.gold_spans[0].outcome == expected


def test_diagnostics_distinguish_window_omission_and_final_match() -> None:
    target = _child(
        child_id="target",
        parent_id="parent-target",
        doc_id=DOC_ID,
        start_char=100,
        end_char=120,
        fusion_rank=1,
        reranker_rank=1,
    )
    parent = _parent(
        parent_id="parent-target",
        pre_cap_rank=1,
        selected_rank=1,
        selection_outcome="selected",
        all_child_ids=("target",),
        supporting_child_ids=("target",),
    )
    omitted = diagnose_gold_query(
        query=_gold(),
        trace=_trace(
            children=(target,),
            parents=(parent,),
            hydrated=(
                _hydrated(
                    parent_id="parent-target",
                    selected_rank=1,
                    doc_id=DOC_ID,
                    window_start=200,
                    window_end=220,
                ),
            ),
        ),
    )
    matched = diagnose_gold_query(
        query=_gold(),
        trace=_trace(
            children=(target,),
            parents=(parent,),
            hydrated=(
                _hydrated(
                    parent_id="parent-target",
                    selected_rank=1,
                    doc_id=DOC_ID,
                    window_start=95,
                    window_end=125,
                ),
            ),
        ),
    )

    assert omitted.gold_spans[0].outcome == "window_omission"
    assert matched.gold_spans[0].outcome == "hydrated_match"


def test_report_is_query_body_free_and_policy_fingerprint_bound() -> None:
    target = _child(
        child_id="target",
        parent_id="parent-target",
        doc_id=DOC_ID,
        start_char=100,
        end_char=120,
        fusion_rank=1,
        reranker_rank=1,
    )
    parent = _parent(
        parent_id="parent-target",
        pre_cap_rank=1,
        selected_rank=1,
        selection_outcome="selected",
        all_child_ids=("target",),
        supporting_child_ids=("target",),
    )
    policy = _policy(reranker_top_n=20)
    trace = _trace(
        children=(target,),
        parents=(parent,),
        hydrated=(
            _hydrated(
                parent_id="parent-target",
                selected_rank=1,
                doc_id=DOC_ID,
                window_start=95,
                window_end=125,
            ),
        ),
        retrieval_fingerprint=compute_retrieval_fingerprint(policy),
    )
    diagnostic = diagnose_gold_query(query=_gold(), trace=trace)

    report = build_regression_report(
        dataset_id="gold-v2-engineering",
        gold_dataset_sha256="f" * 64,
        generation_id="generation-ready",
        generation_manifest_sha256=MANIFEST_SHA,
        retrieval_policy=policy,
        diagnostics=(diagnostic,),
    )

    serialized = report.model_dump_json()
    assert _gold().query not in serialized
    assert "private query text" not in serialized
    assert report.stage_counts[0].outcome == "hydrated_match"


def test_runtime_only_reranker_candidate_changes_fingerprint_and_revalidates() -> None:
    sealed = _policy(reranker_top_n=20)

    candidate = _runtime_policy(sealed_policy=sealed, reranker_top_n=80)

    assert candidate.reranker_top_n == 80
    assert compute_retrieval_fingerprint(candidate) != compute_retrieval_fingerprint(
        sealed
    )
    with pytest.raises(RegressionDiagnosisCliError, match="policy validation"):
        _runtime_policy(sealed_policy=sealed, reranker_top_n=101)


def test_exact_generation_selection_never_reads_deployment_pointer() -> None:
    expected = Mock(
        generation_id="generation-ready",
        state="READY",
        manifest_sha256=MANIFEST_SHA,
    )
    registry = Mock()
    registry.get_generation.return_value = expected
    registry.deployment.side_effect = AssertionError(
        "deployment pointer must not be read"
    )

    selected = _select_ready_generation(
        registry=registry,
        generation_id="generation-ready",
    )

    assert selected is expected
    registry.get_generation.assert_called_once_with("generation-ready")
    registry.deployment.assert_not_called()


def test_query_selection_is_digest_bound_and_limited_to_twenty(tmp_path) -> None:
    dataset = _dataset(21)
    selected_ids = [query.query_id for query in dataset.queries[:10]]

    selected = _select_queries(
        root=tmp_path,
        dataset=dataset,
        gold_sha256="a" * 64,
        query_ids=selected_ids,
        subset_path=None,
    )

    assert tuple(query.query_id for query in selected) == tuple(selected_ids)
    with pytest.raises(RegressionDiagnosisCliError, match="between 10 and 20"):
        _select_queries(
            root=tmp_path,
            dataset=dataset,
            gold_sha256="a" * 64,
            query_ids=[query.query_id for query in dataset.queries],
            subset_path=None,
        )

    subset_path = tmp_path / "subset.json"
    subset_path.write_bytes(
        model_json_bytes(
            RegressionQuerySubset(
                schema_version="regression_query_subset_v1",
                gold_dataset_sha256="b" * 64,
                query_ids=tuple(selected_ids),
            )
        )
    )
    with pytest.raises(RegressionDiagnosisCliError, match="digest mismatch"):
        _load_subset(subset_path, gold_sha256="a" * 64)


def test_failure_artifact_does_not_persist_exception_message(tmp_path) -> None:
    output = tmp_path / "diagnostic.json"

    _write_failure(
        root=tmp_path.resolve(),
        output=output,
        error=RuntimeError("secret provider response body"),
    )

    payload = output.with_name("diagnostic.json.failure.json").read_text(
        encoding="utf-8"
    )
    assert "secret provider response body" not in payload
    assert "RuntimeError" in payload


def test_strict_subset_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        RegressionQuerySubset.model_validate(
            {
                "schema_version": "regression_query_subset_v1",
                "gold_dataset_sha256": "a" * 64,
                "query_ids": tuple(f"q-{index:02d}" for index in range(10)),
                "query": "must never enter the safe subset contract",
            }
        )


def test_cli_missing_required_artifact_fails_nonzero_without_provider_start(
    tmp_path,
) -> None:
    arguments = [
        "--project-root",
        str(tmp_path),
        "--index-config",
        "missing-index.yaml",
        "--gold-dataset",
        "missing-gold.json",
        "--candidate-generation-id",
        "generation-ready",
        "--reranker-top-n",
        "80",
        "--output",
        "diagnostic.json",
    ]
    for index in range(10):
        arguments.extend(("--query-id", f"q-{index:02d}"))

    exit_code = main(arguments)

    assert exit_code == 1
    failure = tmp_path / "diagnostic.json.failure.json"
    assert failure.is_file()
    assert "query" not in failure.read_text(encoding="utf-8")
