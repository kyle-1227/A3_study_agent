from __future__ import annotations

import hashlib

import pytest

from src.rag.parent_child.benchmarking import (
    BenchmarkExecutionError,
    BenchmarkRunBinding,
    run_paired_benchmark,
)
from src.rag.parent_child.evaluation import GoldDataset, GoldEvidenceSpan, GoldQuery
from src.rag.parent_child.flat_baseline import (
    FlatBaselineChunkMetadata,
    FlatBaselineDocument,
    FlatBaselineHit,
    FlatBaselineRetrievalResult,
    make_flat_chunk_id,
)
from src.rag.parent_child.models import ChildDocument, ChildMetadata, ParentRecord
from src.rag.parent_child.retrieval import (
    ChildEvidenceHit,
    HybridRetrievalRequest,
    HybridRetrievalResult,
    HydratedParentContext,
    ParentAggregate,
    ParentContextWindow,
    RetrievalTimings,
)


POLICY_ID = "a" * 64
SOURCE_SHA1 = "b" * 40
DOC_ID = "doc_" + "c" * 40
GENERATION_ID = "candidate-1"
GOLD_SHA = "d" * 64
EMBEDDING_SHA = "e" * 64
BASELINE_RETRIEVAL_SHA = "f" * 64
CANDIDATE_RETRIEVAL_SHA = "1" * 64
BASELINE_MANIFEST_SHA = "2" * 64
CANDIDATE_MANIFEST_SHA = "3" * 64


def _dataset() -> GoldDataset:
    span = GoldEvidenceSpan(
        schema_version="gold_evidence_span_v1",
        gold_span_id="gold_alpha",
        source_group_id="source-a",
        source_relpath="math/source.md",
        doc_id=DOC_ID,
        pagination_kind="logical",
        page_start=1,
        page_end=1,
        start_char=0,
        end_char=5,
        section_path=("Alpha",),
        relevance_grade=3,
    )
    return GoldDataset(
        schema_version="gold_dataset_v1",
        dataset_id="gold-v1",
        queries=(
            GoldQuery(
                schema_version="gold_query_v1",
                query_id="query-alpha",
                subject="math",
                query="alpha?",
                dataset_kind="human_gold",
                eligible_for_rollout=True,
                gold_spans=(span,),
            ),
        ),
    )


def _flat_result(
    *, section_path: tuple[str, ...] = ("Alpha",)
) -> FlatBaselineRetrievalResult:
    content = "alpha"
    chunk_id = make_flat_chunk_id(
        doc_id=DOC_ID,
        policy_id=POLICY_ID,
        start_char=0,
        end_char=len(content),
        content_sha1=hashlib.sha1(content.encode("utf-8")).hexdigest(),
    )
    document = FlatBaselineDocument(
        schema_version="flat_baseline_document_v1",
        content=content,
        metadata=FlatBaselineChunkMetadata(
            schema_version="flat_baseline_chunk_metadata_v1",
            chunk_id=chunk_id,
            doc_id=DOC_ID,
            subject="math",
            policy_id=POLICY_ID,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            chunk_chars=len(content),
            content_sha1=hashlib.sha1(content.encode("utf-8")).hexdigest(),
            source_file="source.md",
            source_relpath="math/source.md",
            source_file_sha1=SOURCE_SHA1,
            doc_type="markdown",
            section_path=section_path,
            pagination_kind="logical",
            page_start=1,
            page_end=1,
        ),
    )
    return FlatBaselineRetrievalResult(
        hits=(FlatBaselineHit(document=document, rank=1, rerank_score=0.9),),
        vector_ms=1.0,
        bm25_ms=2.0,
        reranker_ms=3.0,
        total_ms=6.0,
    )


def _candidate_result(
    *, section_path: tuple[str, ...] = ("Alpha",)
) -> HybridRetrievalResult:
    content = "alpha"
    section_title = section_path[-1] if section_path else ""
    parent = ParentRecord(
        schema_version="parent_record_v1",
        parent_id="parent_" + "4" * 40,
        doc_id=DOC_ID,
        subject="math",
        generation_id=GENERATION_ID,
        policy_id=POLICY_ID,
        parent_index=0,
        source_file="source.md",
        source_relpath="math/source.md",
        source_file_sha1=SOURCE_SHA1,
        doc_type="markdown",
        extraction_method="fixture_v1",
        cleaning_policy_id="clean_v1",
        section_id="section_" + "5" * 40,
        section_title=section_title,
        section_path=section_path,
        pagination_kind="logical",
        page_start=1,
        page_end=1,
        start_char=0,
        end_char=len(content),
        parent_chars=len(content),
        content_sha1=hashlib.sha1(content.encode("utf-8")).hexdigest(),
        content=content,
    )
    child = ChildDocument(
        schema_version="child_document_v1",
        content=content,
        metadata=ChildMetadata(
            schema_version="child_metadata_v1",
            child_id="child_" + "6" * 40,
            parent_id=parent.parent_id,
            doc_id=DOC_ID,
            subject="math",
            generation_id=GENERATION_ID,
            policy_id=POLICY_ID,
            child_index=0,
            child_start_in_parent=0,
            child_end_in_parent=len(content),
            start_char=0,
            end_char=len(content),
            child_chars=len(content),
            content_sha1=hashlib.sha1(content.encode("utf-8")).hexdigest(),
            source_file="source.md",
            source_relpath="math/source.md",
            source_file_sha1=SOURCE_SHA1,
            doc_type="markdown",
            section_id=parent.section_id,
            section_title=section_title,
            section_path=section_path,
            pagination_kind="logical",
            page_start=1,
            page_end=1,
        ),
    )
    request = HybridRetrievalRequest(
        schema_version="hybrid_retrieval_request_v1",
        request_id="candidate-query-alpha",
        query="alpha?",
        subject="math",
        generation_id=GENERATION_ID,
    )
    return HybridRetrievalResult(
        schema_version="hybrid_retrieval_result_v1",
        status="ok",
        request=request,
        retrieval_fingerprint=CANDIDATE_RETRIEVAL_SHA,
        ranked_children=(
            ChildEvidenceHit(
                schema_version="child_evidence_hit_v1",
                final_rank=1,
                document=child,
                vector_rank=1,
                bm25_rank=None,
                vector_raw_score=0.1,
                bm25_raw_score=None,
                rrf_score=0.1,
                rerank_score=0.9,
            ),
        ),
        ranked_parents=(
            ParentAggregate(
                schema_version="parent_aggregate_v1",
                rank=1,
                parent_id=parent.parent_id,
                subject="math",
                source_relpath="math/source.md",
                parent_score=0.9,
                best_child_rank=1,
                supporting_child_ids=(child.metadata.child_id,),
            ),
        ),
        hydrated_parents=(
            HydratedParentContext(
                schema_version="hydrated_parent_context_v1",
                rank=1,
                parent=parent,
                supporting_child_ids=(child.metadata.child_id,),
                expansion_mode="full_parent",
                heading="Alpha",
                windows=(
                    ParentContextWindow(
                        schema_version="parent_context_window_v1",
                        start_in_parent=0,
                        end_in_parent=len(content),
                        content=content,
                    ),
                ),
            ),
        ),
        timings=RetrievalTimings(
            schema_version="retrieval_timings_v1",
            vector_ms=1.0,
            bm25_ms=2.0,
            reranker_ms=3.0,
            hydrate_ms=4.0,
            total_ms=10.0,
        ),
    )


def _bindings() -> tuple[BenchmarkRunBinding, BenchmarkRunBinding]:
    baseline = BenchmarkRunBinding(
        schema_version="benchmark_run_binding_v1",
        run_id="baseline-run",
        dataset_id="gold-v1",
        gold_dataset_sha256=GOLD_SHA,
        embedding_fingerprint=EMBEDDING_SHA,
        retrieval_fingerprint=BASELINE_RETRIEVAL_SHA,
        implementation_kind="flat_baseline",
        artifact_manifest_sha256=BASELINE_MANIFEST_SHA,
        generation_id=None,
    )
    candidate = BenchmarkRunBinding(
        schema_version="benchmark_run_binding_v1",
        run_id="candidate-run",
        dataset_id="gold-v1",
        gold_dataset_sha256=GOLD_SHA,
        embedding_fingerprint=EMBEDDING_SHA,
        retrieval_fingerprint=CANDIDATE_RETRIEVAL_SHA,
        implementation_kind="parent_child_candidate",
        artifact_manifest_sha256=CANDIDATE_MANIFEST_SHA,
        generation_id=GENERATION_ID,
    )
    return baseline, candidate


def test_paired_benchmark_binds_same_gold_and_projects_policy_independent_spans() -> (
    None
):
    dataset = _dataset()
    baseline, candidate = _bindings()

    execution = run_paired_benchmark(
        dataset=dataset,
        baseline_binding=baseline,
        candidate_binding=candidate,
        baseline_retrieve=lambda query: _flat_result(),
        candidate_retrieve=lambda query: _candidate_result(),
        token_counter=len,
    )

    assert execution.baseline_input.gold_dataset_sha256 == GOLD_SHA
    assert execution.candidate_input.gold_dataset_sha256 == GOLD_SHA
    assert execution.baseline_input.parent_aware is False
    assert execution.candidate_input.parent_aware is True
    assert execution.candidate_input.results[0].hits[0].start_char == 0
    assert execution.candidate_input.results[0].hits[0].end_char == 5
    assert len(execution.diagnostics) == 2


def test_paired_benchmark_projects_explicit_empty_section_paths_as_none() -> None:
    dataset = _dataset()
    baseline, candidate = _bindings()

    execution = run_paired_benchmark(
        dataset=dataset,
        baseline_binding=baseline,
        candidate_binding=candidate,
        baseline_retrieve=lambda query: _flat_result(section_path=()),
        candidate_retrieve=lambda query: _candidate_result(section_path=()),
        token_counter=len,
    )

    assert execution.baseline_input.results[0].hits[0].section_path is None
    assert execution.candidate_input.results[0].hits[0].section_path is None


def test_candidate_failure_never_returns_baseline_success() -> None:
    dataset = _dataset()
    baseline, candidate = _bindings()
    calls: list[str] = []

    def baseline_retrieve(query: GoldQuery) -> FlatBaselineRetrievalResult:
        calls.append(query.query_id)
        return _flat_result()

    def candidate_retrieve(query: GoldQuery) -> HybridRetrievalResult:
        del query
        raise RuntimeError("provider response body must not become a result")

    with pytest.raises(BenchmarkExecutionError, match="candidate retrieval failed"):
        run_paired_benchmark(
            dataset=dataset,
            baseline_binding=baseline,
            candidate_binding=candidate,
            baseline_retrieve=baseline_retrieve,
            candidate_retrieve=candidate_retrieve,
            token_counter=len,
        )
    assert calls == ["query-alpha"]


def test_flat_chunk_metadata_rejects_identity_tampering() -> None:
    result = _flat_result()
    payload = result.hits[0].document.metadata.model_dump(mode="python")
    payload["policy_id"] = "9" * 64
    with pytest.raises(ValueError, match="flat chunk ID"):
        FlatBaselineChunkMetadata.model_validate(payload)
