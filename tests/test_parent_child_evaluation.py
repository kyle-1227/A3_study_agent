from __future__ import annotations

from pydantic import ValidationError
import pytest

from src.rag.parent_child.evaluation import (
    BootstrapConfig,
    ComparisonMetricSpec,
    EvaluationContractError,
    EvaluationGateConfig,
    GoldDataset,
    GoldEvidenceSpan,
    GoldQuery,
    QueryRetrievalResult,
    RetrievalEvaluationInput,
    RetrievedEvidenceHit,
    compare_paired_reports,
    evaluate_retrieval_run,
)


DOC_A = "doc_" + "a" * 40
DOC_B = "doc_" + "b" * 40
DOC_NOISE = "doc_" + "f" * 40


def _span(
    suffix: str,
    *,
    doc_id: str = DOC_A,
    source_relpath: str = "math/book-a.pdf",
    source_group_id: str = "math-source-a",
    start_char: int = 100,
    end_char: int = 150,
    section_path: tuple[str, ...] | None = ("Limits",),
) -> GoldEvidenceSpan:
    return GoldEvidenceSpan(
        schema_version="gold_evidence_span_v1",
        gold_span_id=f"gold_{suffix}",
        source_group_id=source_group_id,
        source_relpath=source_relpath,
        doc_id=doc_id,
        pagination_kind="physical",
        page_start=1,
        page_end=1,
        start_char=start_char,
        end_char=end_char,
        section_path=section_path,
        relevance_grade=3,
    )


def _query(
    query_id: str,
    *,
    subject: str = "math",
    dataset_kind: str = "human_gold",
    eligible: bool = True,
    spans: tuple[GoldEvidenceSpan, ...] | None = None,
) -> GoldQuery:
    active_spans = spans if spans is not None else (_span(query_id),)
    return GoldQuery(
        schema_version="gold_query_v1",
        query_id=query_id,
        subject=subject,
        query=f"Question {query_id}",
        dataset_kind=dataset_kind,
        eligible_for_rollout=eligible,
        gold_spans=active_spans,
    )


def _hit(
    hit_id: str,
    *,
    rank: int,
    doc_id: str,
    source_relpath: str,
    start_char: int,
    end_char: int,
    section_path: tuple[str, ...] | None,
    parent_rank: int | None,
) -> RetrievedEvidenceHit:
    return RetrievedEvidenceHit(
        schema_version="retrieved_evidence_hit_v1",
        hit_id=hit_id,
        rank=rank,
        parent_id=None if parent_rank is None else f"parent-{hit_id}",
        parent_rank=parent_rank,
        doc_id=doc_id,
        source_relpath=source_relpath,
        pagination_kind="physical",
        page_start=1,
        page_end=1,
        start_char=start_char,
        end_char=end_char,
        section_path=section_path,
    )


def _result(
    query: GoldQuery,
    hits: tuple[RetrievedEvidenceHit, ...],
) -> QueryRetrievalResult:
    return QueryRetrievalResult(
        schema_version="query_retrieval_result_v1",
        query_id=query.query_id,
        subject=query.subject,
        hits=hits,
    )


def _input(
    run_id: str,
    dataset: GoldDataset,
    results: tuple[QueryRetrievalResult, ...],
    *,
    parent_aware: bool,
) -> RetrievalEvaluationInput:
    return RetrievalEvaluationInput(
        schema_version="retrieval_evaluation_input_v1",
        run_id=run_id,
        dataset_id=dataset.dataset_id,
        parent_aware=parent_aware,
        results=results,
    )


def _gate(
    subjects: tuple[str, ...],
    *,
    global_queries: int,
    subject_queries: int,
    sources: int,
) -> EvaluationGateConfig:
    return EvaluationGateConfig(
        schema_version="evaluation_gate_config_v1",
        primary_subjects=subjects,
        min_global_rollout_queries=global_queries,
        min_subject_rollout_queries=subject_queries,
        min_independent_sources_per_subject=sources,
    )


def _dataset(*queries: GoldQuery, dataset_id: str = "gold-v1") -> GoldDataset:
    return GoldDataset(
        schema_version="gold_dataset_v1",
        dataset_id=dataset_id,
        queries=tuple(queries),
    )


def _metric_value(metrics: object, field_name: str, k: int) -> float | None:
    values = getattr(metrics, field_name)
    return next(value.value for value in values if value.k == k)


def test_gold_contract_forbids_child_ids_extras_and_synthetic_rollout() -> None:
    payload = _span("one").model_dump(mode="python")
    payload["gold_span_id"] = "child_" + "0" * 40
    with pytest.raises(ValidationError, match="gold_ namespace"):
        GoldEvidenceSpan.model_validate(payload)

    query_payload = _query("q1").model_dump(mode="python")
    query_payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GoldQuery.model_validate(query_payload)

    with pytest.raises(ValidationError, match="never rollout eligible"):
        _query(
            "smoke",
            dataset_kind="synthetic_smoke",
            eligible=True,
        )


def test_metrics_use_exact_cleaned_spans_and_parent_ranks() -> None:
    spans = (
        _span("a", start_char=100, end_char=150),
        _span(
            "b",
            doc_id=DOC_B,
            source_relpath="math/book-b.pdf",
            source_group_id="math-source-b",
            start_char=500,
            end_char=550,
            section_path=("Derivatives",),
        ),
    )
    query = _query("two-spans", spans=spans)
    dataset = _dataset(query)
    hits = (
        _hit(
            "noise",
            rank=1,
            doc_id=DOC_NOISE,
            source_relpath="math/noise.pdf",
            start_char=1,
            end_char=20,
            section_path=("Other",),
            parent_rank=1,
        ),
        _hit(
            "match-a",
            rank=2,
            doc_id=DOC_A,
            source_relpath="math/book-a.pdf",
            start_char=120,
            end_char=160,
            section_path=("Limits",),
            parent_rank=2,
        ),
        _hit(
            "match-b",
            rank=3,
            doc_id=DOC_B,
            source_relpath="math/book-b.pdf",
            start_char=490,
            end_char=510,
            section_path=("Derivatives",),
            parent_rank=3,
        ),
    )
    report = evaluate_retrieval_run(
        dataset,
        _input(
            "candidate",
            dataset,
            (_result(query, hits),),
            parent_aware=True,
        ),
        top_ks=(1, 2, 3),
        parent_top_ks=(1, 2, 3),
        gate_config=_gate(("math",), global_queries=1, subject_queries=1, sources=2),
    )

    metrics = report.per_query[0]
    assert [_metric_value(metrics, "evidence_recall", k) for k in (1, 2, 3)] == [
        0.0,
        0.5,
        1.0,
    ]
    assert [_metric_value(metrics, "parent_recall", k) for k in (1, 2, 3)] == [
        0.0,
        0.5,
        1.0,
    ]
    assert [_metric_value(metrics, "source_recall", k) for k in (1, 2, 3)] == [
        0.0,
        0.5,
        1.0,
    ]
    assert [_metric_value(metrics, "section_recall", k) for k in (1, 2, 3)] == [
        0.0,
        0.5,
        1.0,
    ]
    assert _metric_value(metrics, "noise", 1) == 1.0
    assert _metric_value(metrics, "noise", 2) == 0.5
    assert _metric_value(metrics, "noise", 3) == pytest.approx(1 / 3)
    assert metrics.mrr == 0.5
    assert report.rollout_data_gate.passed is True
    assert report.rollout_data_gate.production_recommendation_blocked is False


def test_zero_hits_and_missing_section_are_explicitly_undefined_where_needed() -> None:
    query = _query("empty", spans=(_span("empty", section_path=None),))
    dataset = _dataset(query)
    report = evaluate_retrieval_run(
        dataset,
        _input(
            "empty-run",
            dataset,
            (_result(query, ()),),
            parent_aware=True,
        ),
        top_ks=(1, 5),
        parent_top_ks=(1, 3),
        gate_config=_gate(("math",), global_queries=1, subject_queries=1, sources=1),
    )

    metrics = report.per_query[0]
    assert _metric_value(metrics, "evidence_recall", 5) == 0.0
    assert _metric_value(metrics, "parent_recall", 3) == 0.0
    assert _metric_value(metrics, "noise", 5) is None
    assert _metric_value(metrics, "section_recall", 5) is None
    assert metrics.mrr == 0.0
    assert report.global_metrics.noise[1].defined_query_count == 0
    assert report.global_metrics.noise[1].value is None
    assert report.global_metrics.section_recall[1].defined_query_count == 0


def test_low_data_and_synthetic_smoke_block_rollout() -> None:
    smoke = _query(
        "smoke",
        dataset_kind="synthetic_smoke",
        eligible=False,
    )
    dataset = _dataset(smoke)
    result = _result(smoke, ())
    report = evaluate_retrieval_run(
        dataset,
        _input("smoke-run", dataset, (result,), parent_aware=False),
        top_ks=(1,),
        parent_top_ks=(),
        gate_config=_gate(("math",), global_queries=1, subject_queries=1, sources=1),
    )

    assert report.rollout_eligible_global_metrics.query_count == 0
    assert report.rollout_data_gate.passed is False
    assert report.rollout_data_gate.production_recommendation_blocked is True
    assert "global_rollout_gold_below_minimum" in report.rollout_data_gate.blocker_codes
    assert "math:rollout_gold_below_minimum" in report.rollout_data_gate.blocker_codes
    assert (
        "math:independent_sources_below_minimum"
        in report.rollout_data_gate.blocker_codes
    )


def test_query_set_and_parent_contract_mismatches_fail_fast() -> None:
    first = _query("first")
    second = _query("second")
    dataset = _dataset(first, second)
    with pytest.raises(EvaluationContractError, match="query set mismatch"):
        evaluate_retrieval_run(
            dataset,
            _input(
                "missing",
                dataset,
                (_result(first, ()),),
                parent_aware=False,
            ),
            top_ks=(1,),
            parent_top_ks=(),
            gate_config=_gate(
                ("math",), global_queries=1, subject_queries=1, sources=1
            ),
        )

    relevant = _hit(
        "has-parent",
        rank=1,
        doc_id=DOC_A,
        source_relpath="math/book-a.pdf",
        start_char=100,
        end_char=120,
        section_path=("Limits",),
        parent_rank=1,
    )
    single_dataset = _dataset(first)
    with pytest.raises(EvaluationContractError, match="parent fields"):
        evaluate_retrieval_run(
            single_dataset,
            _input(
                "wrong-parent-contract",
                single_dataset,
                (_result(first, (relevant,)),),
                parent_aware=False,
            ),
            top_ks=(1,),
            parent_top_ks=(),
            gate_config=_gate(
                ("math",), global_queries=1, subject_queries=1, sources=1
            ),
        )


def _binary_hit(query: GoldQuery, *, relevant: bool) -> RetrievedEvidenceHit:
    gold = query.gold_spans[0]
    return _hit(
        f"hit-{query.query_id}",
        rank=1,
        doc_id=gold.doc_id if relevant else DOC_NOISE,
        source_relpath=gold.source_relpath
        if relevant
        else f"{query.subject}/noise.pdf",
        start_char=gold.start_char if relevant else 1,
        end_char=gold.end_char if relevant else 20,
        section_path=gold.section_path if relevant else ("Noise",),
        parent_rank=None,
    )


def test_subject_stratified_paired_bootstrap_is_reproducible() -> None:
    queries = (
        _query("math-1"),
        _query(
            "math-2",
            spans=(
                _span(
                    "math-2",
                    source_group_id="math-source-b",
                    start_char=200,
                    end_char=250,
                ),
            ),
        ),
        _query(
            "python-1",
            subject="python",
            spans=(
                _span(
                    "python-1",
                    doc_id=DOC_B,
                    source_relpath="python/book.pdf",
                    source_group_id="python-source-a",
                    start_char=300,
                    end_char=350,
                    section_path=("Functions",),
                ),
            ),
        ),
        _query(
            "python-2",
            subject="python",
            spans=(
                _span(
                    "python-2",
                    doc_id=DOC_B,
                    source_relpath="python/book.pdf",
                    source_group_id="python-source-a",
                    start_char=400,
                    end_char=450,
                    section_path=("Classes",),
                ),
            ),
        ),
    )
    dataset = _dataset(*queries, dataset_id="paired-gold")
    gate = _gate(("math", "python"), global_queries=4, subject_queries=2, sources=1)
    baseline_results = tuple(
        _result(query, (_binary_hit(query, relevant=index == 0),))
        for index, query in enumerate(queries)
    )
    candidate_results = tuple(
        _result(query, (_binary_hit(query, relevant=index != 3),))
        for index, query in enumerate(queries)
    )
    baseline = evaluate_retrieval_run(
        dataset,
        _input("baseline", dataset, baseline_results, parent_aware=False),
        top_ks=(1,),
        parent_top_ks=(),
        gate_config=gate,
    )
    candidate = evaluate_retrieval_run(
        dataset,
        _input("candidate", dataset, candidate_results, parent_aware=False),
        top_ks=(1,),
        parent_top_ks=(),
        gate_config=gate,
    )
    specs = (
        ComparisonMetricSpec(
            schema_version="comparison_metric_spec_v1",
            metric_name="evidence_recall_at_k",
            k=1,
        ),
        ComparisonMetricSpec(
            schema_version="comparison_metric_spec_v1",
            metric_name="mrr",
            k=None,
        ),
    )
    bootstrap = BootstrapConfig(
        schema_version="paired_bootstrap_config_v1",
        iterations=500,
        seed=1729,
        confidence=0.95,
    )

    first = compare_paired_reports(
        baseline,
        candidate,
        metric_specs=specs,
        bootstrap=bootstrap,
        eligible_queries_only=True,
    )
    second = compare_paired_reports(
        baseline,
        candidate,
        metric_specs=specs,
        bootstrap=bootstrap,
        eligible_queries_only=True,
    )

    assert first == second
    assert first.global_metrics[0].baseline_mean == 0.25
    assert first.global_metrics[0].candidate_mean == 0.75
    assert first.global_metrics[0].mean_delta == 0.5
    assert first.global_metrics[0].sample_count == 4
    assert tuple(item.subject for item in first.subjects) == ("math", "python")
    assert first.rollout_data_gate_passed is True
    assert first.production_recommendation_blocked is False


def test_smoke_only_comparison_cannot_become_rollout_eligible() -> None:
    query = _query(
        "smoke",
        dataset_kind="synthetic_smoke",
        eligible=False,
    )
    dataset = _dataset(query, dataset_id="smoke-dataset")
    result = (_result(query, ()),)
    gate = _gate(("math",), global_queries=1, subject_queries=1, sources=1)
    baseline = evaluate_retrieval_run(
        dataset,
        _input("baseline", dataset, result, parent_aware=False),
        top_ks=(1,),
        parent_top_ks=(),
        gate_config=gate,
    )
    candidate = evaluate_retrieval_run(
        dataset,
        _input("candidate", dataset, result, parent_aware=False),
        top_ks=(1,),
        parent_top_ks=(),
        gate_config=gate,
    )
    spec = ComparisonMetricSpec(
        schema_version="comparison_metric_spec_v1",
        metric_name="mrr",
        k=None,
    )
    bootstrap = BootstrapConfig(
        schema_version="paired_bootstrap_config_v1",
        iterations=10,
        seed=1,
        confidence=0.9,
    )
    comparison = compare_paired_reports(
        baseline,
        candidate,
        metric_specs=(spec,),
        bootstrap=bootstrap,
        eligible_queries_only=False,
    )
    assert comparison.rollout_data_gate_passed is False
    assert comparison.production_recommendation_blocked is True

    with pytest.raises(EvaluationContractError, match="no queries remain"):
        compare_paired_reports(
            baseline,
            candidate,
            metric_specs=(spec,),
            bootstrap=bootstrap,
            eligible_queries_only=True,
        )
