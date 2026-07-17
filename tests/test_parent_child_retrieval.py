from __future__ import annotations

import hashlib
from typing import Callable, Literal

import pytest

from src.rag.parent_child.handoff import (
    EvidenceHandoffError,
    JudgeKeepDecision,
    build_local_evidence_refs,
    build_multi_local_evidence_refs,
    kept_child_ids_from_decisions,
    parent_context_items,
)
from src.rag.parent_child.models import ChildDocument, ChildMetadata, ParentRecord
from src.rag.parent_child.retrieval import (
    ChildSearchCandidate,
    HybridRetrievalPolicy,
    HybridRetrievalRequest,
    MultiBranchHybridRequest,
    ParentChildHybridRetriever,
    ParentHydrationError,
    RetrievalChannelError,
    RetrievalInvariantError,
    RetrievalProtocolError,
    RerankerTransportExhaustedError,
    RerankCandidate,
    RerankScore,
    WeightedHybridBranch,
    aggregate_parent_score,
)


GENERATION_ID = "gen-a"
POLICY_ID = "a" * 64
SOURCE_SHA1 = "b" * 40


def _parent(
    number: int,
    *,
    content: str,
    source_relpath: str,
    section_title: str,
    subject: str = "math",
) -> ParentRecord:
    parent_start = number * 1000
    source_file = source_relpath.rsplit("/", maxsplit=1)[-1]
    return ParentRecord(
        schema_version="parent_record_v1",
        parent_id="parent_" + f"{number:040x}",
        doc_id="doc_" + f"{number:040x}",
        subject=subject,
        generation_id=GENERATION_ID,
        policy_id=POLICY_ID,
        parent_index=0,
        source_file=source_file,
        source_relpath=source_relpath,
        source_file_sha1=SOURCE_SHA1,
        doc_type="notes",
        extraction_method="fixture_v1",
        cleaning_policy_id="clean_v1",
        section_id="section_" + f"{number:040x}",
        section_title=section_title,
        section_path=(section_title,),
        pagination_kind="logical",
        page_start=1,
        page_end=1,
        start_char=parent_start,
        end_char=parent_start + len(content),
        parent_chars=len(content),
        content_sha1=hashlib.sha1(content.encode("utf-8")).hexdigest(),
        content=content,
    )


def _child(
    number: int,
    parent: ParentRecord,
    start: int,
    end: int,
    *,
    child_id: str | None,
) -> ChildDocument:
    content = parent.content[start:end]
    exact_child_id = child_id or "child_" + f"{number:040x}"
    return ChildDocument(
        schema_version="child_document_v1",
        content=content,
        metadata=ChildMetadata(
            schema_version="child_metadata_v1",
            child_id=exact_child_id,
            parent_id=parent.parent_id,
            doc_id=parent.doc_id,
            subject=parent.subject,
            generation_id=parent.generation_id,
            policy_id=parent.policy_id,
            child_index=number,
            child_start_in_parent=start,
            child_end_in_parent=end,
            start_char=parent.start_char + start,
            end_char=parent.start_char + end,
            child_chars=end - start,
            content_sha1=hashlib.sha1(content.encode("utf-8")).hexdigest(),
            source_file=parent.source_file,
            source_relpath=parent.source_relpath,
            source_file_sha1=parent.source_file_sha1,
            doc_type=parent.doc_type,
            section_id=parent.section_id,
            section_title=parent.section_title,
            section_path=parent.section_path,
            pagination_kind=parent.pagination_kind,
            page_start=1,
            page_end=1,
        ),
    )


def _search_candidate(child: ChildDocument, raw_score: float) -> ChildSearchCandidate:
    return ChildSearchCandidate(
        schema_version="child_search_candidate_v1",
        document=child,
        raw_score=raw_score,
    )


def _policy(
    *, fallback_mode: Literal["disabled", "rrf_only"] = "rrf_only"
) -> HybridRetrievalPolicy:
    return HybridRetrievalPolicy(
        schema_version="hybrid_retrieval_policy_v1",
        generation_manifest_sha256="a" * 64,
        embedding_fingerprint="b" * 64,
        bm25_tokenizer_fingerprint="c" * 64,
        reranker_fingerprint="d" * 64,
        vector_top_k=5,
        bm25_top_k=5,
        vector_rrf_weight=1.0,
        bm25_rrf_weight=1.0,
        rrf_k=20,
        reranker_top_n=5,
        reranker_transport_fallback_mode=fallback_mode,
        unique_parent_top_k=2,
        max_children_per_parent=2,
        max_parents_per_source=1,
        parent_support_lambda=0.5,
        full_parent_max_chars=50,
        hit_window_chars_per_side=3,
        multi_subject_per_subject_top_k=2,
        multi_subject_max_parents=4,
        subject_coverage_quota=1,
    )


def _request() -> HybridRetrievalRequest:
    return HybridRetrievalRequest(
        schema_version="hybrid_retrieval_request_v1",
        request_id="request-1",
        query="What is alpha?",
        subject="math",
        generation_id=GENERATION_ID,
    )


class _Search:
    def __init__(
        self,
        results: tuple[ChildSearchCandidate, ...],
        error: Exception | None,
    ) -> None:
        self.results = results
        self.error = error
        self.calls: list[tuple[str, str, str, int]] = []

    def search(
        self,
        *,
        query: str,
        subject: str,
        generation_id: str,
        top_k: int,
    ) -> tuple[ChildSearchCandidate, ...]:
        self.calls.append((query, subject, generation_id, top_k))
        if self.error is not None:
            raise self.error
        return self.results


class _ScoreReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[RerankCandidate, ...]]] = []

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankScore, ...]:
        self.calls.append((query, candidates))
        return tuple(
            RerankScore(
                schema_version="rerank_score_v1",
                child_id=candidate.child_id,
                score=self.scores[candidate.child_id],
            )
            for candidate in candidates
        )


class _RawReranker:
    def __init__(
        self,
        output_factory: Callable[[tuple[RerankCandidate, ...]], object],
    ) -> None:
        self.output_factory = output_factory
        self.calls = 0

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> object:
        del query
        self.calls += 1
        return self.output_factory(candidates)


class _UnavailableReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankScore, ...]:
        del query, candidates
        self.calls += 1
        raise RerankerTransportExhaustedError("fixture transport exhausted")


class _Hydrator:
    def __init__(
        self,
        parents: dict[str, ParentRecord],
        returned_ids: tuple[str, ...] | None,
    ) -> None:
        self.parents = parents
        self.returned_ids = returned_ids
        self.calls: list[tuple[str, ...]] = []

    def get_many(self, parent_ids: tuple[str, ...]) -> tuple[ParentRecord, ...]:
        requested = tuple(parent_ids)
        self.calls.append(requested)
        identities = self.returned_ids if self.returned_ids is not None else requested
        return tuple(self.parents[parent_id] for parent_id in identities)


class _MustNotRerank:
    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankScore, ...]:
        del query, candidates
        raise AssertionError("reranker must not be called")


class _MustNotHydrate:
    def get_many(self, parent_ids: tuple[str, ...]) -> tuple[ParentRecord, ...]:
        del parent_ids
        raise AssertionError("parent hydrator must not be called")


class _QuerySearch:
    def __init__(
        self,
        results_by_query: dict[str, tuple[ChildSearchCandidate, ...]],
    ) -> None:
        self.results_by_query = results_by_query

    def search(
        self,
        *,
        query: str,
        subject: str,
        generation_id: str,
        top_k: int,
    ) -> tuple[ChildSearchCandidate, ...]:
        assert subject == "math"
        assert generation_id == GENERATION_ID
        results = self.results_by_query[query]
        assert len(results) <= top_k
        return results


class _FlexibleQuerySearch:
    def __init__(
        self,
        results_by_query: dict[str, tuple[ChildSearchCandidate, ...]],
    ) -> None:
        self.results_by_query = results_by_query

    def search(
        self,
        *,
        query: str,
        subject: str,
        generation_id: str,
        top_k: int,
    ) -> tuple[ChildSearchCandidate, ...]:
        assert generation_id == GENERATION_ID
        results = self.results_by_query[query]
        assert all(item.document.metadata.subject == subject for item in results)
        assert len(results) <= top_k
        return results


def test_hybrid_retrieval_merges_reranks_caps_and_hydrates_exact_windows() -> None:
    parent_one = _parent(
        1,
        content="# Alpha\n" + "0123456789" * 10,
        source_relpath="math/source-a.md",
        section_title="Alpha",
    )
    parent_two = _parent(
        2,
        content="# Beta\n" + "abcdefghij" * 8,
        source_relpath="math/source-a.md",
        section_title="Beta",
    )
    parent_three = _parent(
        3,
        content="# Gamma\nshort context",
        source_relpath="math/source-b.md",
        section_title="Gamma",
    )
    child_one = _child(1, parent_one, 10, 15, child_id=None)
    child_two = _child(2, parent_one, 17, 22, child_id=None)
    child_three = _child(3, parent_two, 10, 15, child_id=None)
    child_four = _child(4, parent_three, 8, 13, child_id=None)

    vector = _Search(
        (
            _search_candidate(child_three, 0.95),
            _search_candidate(child_one, 0.90),
            _search_candidate(child_four, 0.80),
        ),
        None,
    )
    bm25 = _Search(
        (
            _search_candidate(child_two, 9.0),
            _search_candidate(child_one, 8.0),
        ),
        None,
    )
    reranker = _ScoreReranker(
        {
            child_one.metadata.child_id: 0.80,
            child_two.metadata.child_id: 0.50,
            child_three.metadata.child_id: 0.79,
            child_four.metadata.child_id: 0.70,
        }
    )
    hydrator = _Hydrator(
        {
            parent_one.parent_id: parent_one,
            parent_two.parent_id: parent_two,
            parent_three.parent_id: parent_three,
        },
        None,
    )
    result = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=vector,
        bm25_search=bm25,
        reranker=reranker,
        parent_hydrator=hydrator,
    ).retrieve(_request())

    assert result.status == "ok"
    by_child_id = {
        hit.document.metadata.child_id: hit for hit in result.ranked_children
    }
    merged_hit = by_child_id[child_one.metadata.child_id]
    assert merged_hit.vector_rank == 2
    assert merged_hit.bm25_rank == 2
    assert len(result.ranked_children) == 4

    assert tuple(parent.parent_id for parent in result.ranked_parents) == (
        parent_one.parent_id,
        parent_three.parent_id,
    )
    assert result.ranked_parents[0].parent_score == pytest.approx(0.85)
    assert result.ranked_parents[0].supporting_child_ids == (
        child_one.metadata.child_id,
        child_two.metadata.child_id,
    )
    assert hydrator.calls == [(parent_one.parent_id, parent_three.parent_id)]

    first_context, second_context = result.hydrated_parents
    assert first_context.expansion_mode == "hit_window"
    assert first_context.heading == "Alpha"
    assert len(first_context.windows) == 1
    assert (
        first_context.windows[0].start_in_parent,
        first_context.windows[0].end_in_parent,
    ) == (7, 25)
    assert first_context.windows[0].content == parent_one.content[7:25]
    assert second_context.expansion_mode == "full_parent"
    assert second_context.windows[0].content == parent_three.content


def test_reranker_transport_exhaustion_uses_explicit_rrf_only_ranking() -> None:
    parent = _parent(
        40,
        content="# RRF\n0123456789",
        source_relpath="math/rrf.md",
        section_title="RRF",
    )
    child = _child(40, parent, 6, 11, child_id=None)
    reranker = _UnavailableReranker()
    result, trace = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=_Search((_search_candidate(child, 0.9),), None),
        bm25_search=_Search((_search_candidate(child, 3.0),), None),
        reranker=reranker,
        parent_hydrator=_Hydrator({parent.parent_id: parent}, None),
    ).retrieve_with_diagnostics(_request())

    assert reranker.calls == 1
    assert result.ranking_mode == "rrf_only"
    assert result.fallback_reason_code == "reranker_transport_exhausted"
    assert trace.ranking_mode == "rrf_only"
    assert trace.fallback_reason_code == "reranker_transport_exhausted"
    hit = result.ranked_children[0]
    assert hit.ranking_mode == "rrf_only"
    assert hit.ranking_score == 1.0
    assert hit.rerank_score is None
    assert trace.children[0].submitted_to_reranker is False
    assert trace.children[0].reranker_score is None


def test_reranker_transport_exhaustion_stays_typed_when_fallback_disabled() -> None:
    parent = _parent(
        44,
        content="# Disabled\n0123456789",
        source_relpath="math/disabled.md",
        section_title="Disabled",
    )
    child = _child(44, parent, 11, 16, child_id=None)
    with pytest.raises(RetrievalChannelError, match="transport exhausted"):
        ParentChildHybridRetriever(
            policy=_policy(fallback_mode="disabled"),
            vector_search=_Search((_search_candidate(child, 0.9),), None),
            bm25_search=_Search((), None),
            reranker=_UnavailableReranker(),
            parent_hydrator=_MustNotHydrate(),
        ).retrieve_children(_request())


def test_diagnostic_retrieval_reuses_pipeline_without_query_or_body_leakage() -> None:
    selected_parent = _parent(
        41,
        content="# Selected\n" + "0123456789" * 8,
        source_relpath="math/source-a.md",
        section_title="Selected",
    )
    source_capped_parent = _parent(
        42,
        content="# Capped\n" + "abcdefghij" * 8,
        source_relpath="math/source-a.md",
        section_title="Capped",
    )
    second_selected_parent = _parent(
        43,
        content="# Other\n" + "klmnopqrst" * 8,
        source_relpath="math/source-b.md",
        section_title="Other",
    )
    selected_child = _child(41, selected_parent, 12, 20, child_id=None)
    capped_child = _child(42, source_capped_parent, 10, 18, child_id=None)
    other_child = _child(43, second_selected_parent, 9, 17, child_id=None)
    retriever = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=_Search(
            (
                _search_candidate(selected_child, 0.9),
                _search_candidate(capped_child, 0.8),
                _search_candidate(other_child, 0.7),
            ),
            None,
        ),
        bm25_search=_Search((_search_candidate(selected_child, 3.0),), None),
        reranker=_ScoreReranker(
            {
                selected_child.metadata.child_id: 0.9,
                capped_child.metadata.child_id: 0.8,
                other_child.metadata.child_id: 0.7,
            }
        ),
        parent_hydrator=_Hydrator(
            {
                selected_parent.parent_id: selected_parent,
                source_capped_parent.parent_id: source_capped_parent,
                second_selected_parent.parent_id: second_selected_parent,
            },
            None,
        ),
    )

    result, trace = retriever.retrieve_with_diagnostics(_request())

    assert result.status == "ok"
    by_child = {child.child_id: child for child in trace.children}
    assert by_child[selected_child.metadata.child_id].vector_rank == 1
    assert by_child[selected_child.metadata.child_id].bm25_rank == 1
    assert by_child[selected_child.metadata.child_id].fusion_rank == 1
    assert by_child[selected_child.metadata.child_id].reranker_rank == 1
    by_parent = {parent.parent_id: parent for parent in trace.parents}
    assert by_parent[source_capped_parent.parent_id].selection_outcome == "source_cap"
    assert tuple(parent.parent_id for parent in trace.hydrated_parents) == (
        selected_parent.parent_id,
        second_selected_parent.parent_id,
    )
    first_window = trace.hydrated_parents[0].windows[0]
    assert first_window.start_char >= selected_parent.start_char
    assert first_window.end_char <= selected_parent.end_char
    serialized = trace.model_dump_json()
    assert _request().query not in serialized
    assert selected_parent.content not in serialized
    assert trace.timings.fusion_ms >= 0
    assert trace.timings.parent_aggregation_ms >= 0


def test_normal_zero_hit_is_explicit_and_skips_rerank_and_hydration() -> None:
    vector = _Search((), None)
    bm25 = _Search((), None)
    result = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=vector,
        bm25_search=bm25,
        reranker=_MustNotRerank(),
        parent_hydrator=_MustNotHydrate(),
    ).retrieve(_request())

    assert result.status == "empty"
    assert result.ranked_children == ()
    assert len(vector.calls) == 1
    assert len(bm25.calls) == 1


def test_judge_flow_hydrates_only_kept_children_after_child_preview_stage() -> None:
    parent = _parent(
        8,
        content="# Alpha\n" + "0123456789" * 8,
        source_relpath="math/source-a.md",
        section_title="Alpha",
    )
    child_one = _child(81, parent, 10, 18, child_id=None)
    child_two = _child(82, parent, 20, 28, child_id=None)
    vector = _Search(
        (
            _search_candidate(child_one, 0.9),
            _search_candidate(child_two, 0.8),
        ),
        None,
    )
    bm25 = _Search((_search_candidate(child_two, 3.0),), None)
    hydrator = _Hydrator({parent.parent_id: parent}, None)
    retriever = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=vector,
        bm25_search=bm25,
        reranker=_ScoreReranker(
            {
                child_one.metadata.child_id: 0.9,
                child_two.metadata.child_id: 0.8,
            }
        ),
        parent_hydrator=hydrator,
    )

    child_result = retriever.retrieve_children(_request())

    assert child_result.status == "ok"
    assert hydrator.calls == []
    refs = build_local_evidence_refs(child_result, preview_max_chars=5)
    assert all(len(ref.content_preview) <= 5 for ref in refs)
    assert parent.content not in "".join(ref.model_dump_json() for ref in refs)
    decisions = tuple(
        JudgeKeepDecision(
            schema_version="judge_keep_decision_v1",
            evidence_id=ref.evidence_id,
            keep=ref.evidence_id == child_two.metadata.child_id,
        )
        for ref in refs
    )
    kept = kept_child_ids_from_decisions(refs, decisions)
    assert kept == (child_two.metadata.child_id,)
    contexts = retriever.hydrate_kept_parents(
        child_result,
        kept,
    )
    assert hydrator.calls == [(parent.parent_id,)]
    assert len(contexts) == 1
    assert contexts[0].supporting_child_ids == (child_two.metadata.child_id,)
    context_items = parent_context_items(contexts)
    assert context_items[0].window_spans
    assert child_two.content in context_items[0].content

    with pytest.raises(EvidenceHandoffError, match="identity set"):
        kept_child_ids_from_decisions(refs, decisions[:-1])


def test_required_channel_failure_is_typed_and_does_not_run_other_channels() -> None:
    vector = _Search((), RuntimeError("provider body must not be copied"))
    bm25 = _Search((), None)
    with pytest.raises(RetrievalChannelError, match="RuntimeError") as error:
        ParentChildHybridRetriever(
            policy=_policy(),
            vector_search=vector,
            bm25_search=bm25,
            reranker=_MustNotRerank(),
            parent_hydrator=_MustNotHydrate(),
        ).retrieve(_request())

    assert "provider body" not in str(error.value)
    assert bm25.calls == []


def test_candidate_generation_or_subject_mismatch_fails_before_reranker() -> None:
    parent = _parent(
        1,
        content="# Alpha\n0123456789",
        source_relpath="math/source-a.md",
        section_title="Alpha",
    )
    child = _child(1, parent, 8, 12, child_id=None)
    drifted_metadata = ChildMetadata.model_validate(
        {**child.metadata.model_dump(mode="python"), "generation_id": "gen-b"}
    )
    drifted = ChildDocument(
        schema_version="child_document_v1",
        content=child.content,
        metadata=drifted_metadata,
    )
    with pytest.raises(RetrievalInvariantError, match="generation"):
        ParentChildHybridRetriever(
            policy=_policy(),
            vector_search=_Search((_search_candidate(drifted, 1.0),), None),
            bm25_search=_Search((), None),
            reranker=_MustNotRerank(),
            parent_hydrator=_MustNotHydrate(),
        ).retrieve(_request())


@pytest.mark.parametrize(
    "output_factory",
    [
        lambda candidates: (),
        lambda candidates: (
            {
                "schema_version": "rerank_score_v1",
                "child_id": candidates[0].child_id,
                "score": 0.5,
            },
            {
                "schema_version": "rerank_score_v1",
                "child_id": candidates[0].child_id,
                "score": 0.4,
            },
        ),
        lambda candidates: (
            {
                "schema_version": "rerank_score_v1",
                "child_id": "child_" + "f" * 40,
                "score": 0.5,
            },
        ),
        lambda candidates: (
            {
                "schema_version": "rerank_score_v1",
                "child_id": candidates[0].child_id,
                "score": float("nan"),
            },
        ),
        lambda candidates: (
            {
                "schema_version": "rerank_score_v1",
                "child_id": candidates[0].child_id,
                "score": 1.1,
            },
        ),
    ],
    ids=["missing", "duplicate", "unknown", "non-finite", "out-of-range"],
)
def test_reranker_identity_and_score_protocol_is_fail_fast(
    output_factory: Callable[[tuple[RerankCandidate, ...]], object],
) -> None:
    parent = _parent(
        1,
        content="# Alpha\n0123456789",
        source_relpath="math/source-a.md",
        section_title="Alpha",
    )
    child = _child(1, parent, 8, 12, child_id=None)
    reranker = _RawReranker(output_factory)
    with pytest.raises(RetrievalProtocolError):
        ParentChildHybridRetriever(
            policy=_policy(),
            vector_search=_Search((_search_candidate(child, 1.0),), None),
            bm25_search=_Search((), None),
            reranker=reranker,
            parent_hydrator=_MustNotHydrate(),
        ).retrieve(_request())
    assert reranker.calls == 1


def test_same_child_id_with_conflicting_channel_payload_fails() -> None:
    first_parent = _parent(
        1,
        content="# Alpha\n0123456789",
        source_relpath="math/source-a.md",
        section_title="Alpha",
    )
    second_parent = _parent(
        2,
        content="# Beta\nabcdefghij",
        source_relpath="math/source-b.md",
        section_title="Beta",
    )
    vector_child = _child(1, first_parent, 8, 12, child_id=None)
    bm25_child = _child(
        2,
        second_parent,
        7,
        11,
        child_id=vector_child.metadata.child_id,
    )
    with pytest.raises(RetrievalInvariantError, match="disagree"):
        ParentChildHybridRetriever(
            policy=_policy(),
            vector_search=_Search((_search_candidate(vector_child, 1.0),), None),
            bm25_search=_Search((_search_candidate(bm25_child, 2.0),), None),
            reranker=_MustNotRerank(),
            parent_hydrator=_MustNotHydrate(),
        ).retrieve(_request())


def test_parent_hydrator_must_return_exact_requested_order() -> None:
    first_parent = _parent(
        1,
        content="# Alpha\n0123456789",
        source_relpath="math/source-a.md",
        section_title="Alpha",
    )
    second_parent = _parent(
        2,
        content="# Beta\nabcdefghij",
        source_relpath="math/source-b.md",
        section_title="Beta",
    )
    first_child = _child(1, first_parent, 8, 12, child_id=None)
    second_child = _child(2, second_parent, 7, 11, child_id=None)
    hydrator = _Hydrator(
        {
            first_parent.parent_id: first_parent,
            second_parent.parent_id: second_parent,
        },
        (second_parent.parent_id, first_parent.parent_id),
    )
    with pytest.raises(ParentHydrationError, match="exact requested order"):
        ParentChildHybridRetriever(
            policy=_policy(),
            vector_search=_Search(
                (
                    _search_candidate(first_child, 1.0),
                    _search_candidate(second_child, 0.9),
                ),
                None,
            ),
            bm25_search=_Search((), None),
            reranker=_ScoreReranker(
                {
                    first_child.metadata.child_id: 0.9,
                    second_child.metadata.child_id: 0.8,
                }
            ),
            parent_hydrator=hydrator,
        ).retrieve(_request())


def test_parent_support_formula_is_bounded_and_exact() -> None:
    assert aggregate_parent_score((0.8, 0.5), support_lambda=0.5) == pytest.approx(0.85)
    assert aggregate_parent_score((1.0, 1.0), support_lambda=1.0) == 1.0
    with pytest.raises(ValueError, match="at least one"):
        aggregate_parent_score((), support_lambda=0.5)


def test_multi_branch_parent_fusion_uses_weighted_cross_rrf() -> None:
    first_parent = _parent(
        1,
        content="# Alpha\n0123456789",
        source_relpath="math/source-a.md",
        section_title="Alpha",
    )
    second_parent = _parent(
        2,
        content="# Beta\nabcdefghij",
        source_relpath="math/source-b.md",
        section_title="Beta",
    )
    first_child = _child(1, first_parent, 8, 12, child_id=None)
    second_child = _child(2, second_parent, 7, 11, child_id=None)
    first_query = "alpha and beta"
    second_query = "alpha"
    vector = _QuerySearch(
        {
            first_query: (
                _search_candidate(first_child, 1.0),
                _search_candidate(second_child, 0.9),
            ),
            second_query: (_search_candidate(first_child, 1.0),),
        }
    )
    bm25 = _QuerySearch({first_query: (), second_query: ()})
    retriever = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=vector,
        bm25_search=bm25,
        reranker=_ScoreReranker(
            {
                first_child.metadata.child_id: 0.9,
                second_child.metadata.child_id: 0.8,
            }
        ),
        parent_hydrator=_Hydrator(
            {
                first_parent.parent_id: first_parent,
                second_parent.parent_id: second_parent,
            },
            None,
        ),
    )
    first_request = HybridRetrievalRequest(
        schema_version="hybrid_retrieval_request_v1",
        request_id="request-branch-a",
        query=first_query,
        subject="math",
        generation_id=GENERATION_ID,
    )
    second_request = HybridRetrievalRequest(
        schema_version="hybrid_retrieval_request_v1",
        request_id="request-branch-b",
        query=second_query,
        subject="math",
        generation_id=GENERATION_ID,
    )
    result = retriever.retrieve_multi(
        MultiBranchHybridRequest(
            schema_version="multi_branch_hybrid_request_v1",
            request_id="multi-request",
            generation_id=GENERATION_ID,
            branches=(
                WeightedHybridBranch(
                    schema_version="weighted_hybrid_branch_v1",
                    branch_id="expanded",
                    weight=2.0,
                    request=first_request,
                ),
                WeightedHybridBranch(
                    schema_version="weighted_hybrid_branch_v1",
                    branch_id="focused",
                    weight=1.0,
                    request=second_request,
                ),
            ),
            cross_branch_rrf_k=20,
            parent_top_k=2,
        )
    )

    assert result.status == "ok"
    assert len(result.branch_results) == 2
    assert result.ranked_parents[0].parent_id == first_parent.parent_id
    assert result.ranked_parents[0].cross_branch_rrf_score == pytest.approx(3 / 21)
    assert tuple(
        provenance.branch_id for provenance in result.ranked_parents[0].provenance
    ) == ("expanded", "focused")


def test_multi_branch_judge_flow_defers_parent_hydration_until_kept_ids() -> None:
    parent = _parent(
        21,
        content="# Deferred\n0123456789",
        source_relpath="math/deferred.md",
        section_title="Deferred",
    )
    child = _child(21, parent, 11, 16, child_id=None)
    first_query = "deferred context"
    second_query = "focused context"
    vector = _QuerySearch(
        {
            first_query: (_search_candidate(child, 1.0),),
            second_query: (_search_candidate(child, 0.9),),
        }
    )
    bm25 = _QuerySearch({first_query: (), second_query: ()})
    reranker = _ScoreReranker({child.metadata.child_id: 0.9})
    child_only_retriever = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=vector,
        bm25_search=bm25,
        reranker=reranker,
        parent_hydrator=_MustNotHydrate(),
    )
    requests = tuple(
        WeightedHybridBranch(
            schema_version="weighted_hybrid_branch_v1",
            branch_id=f"branch-{index}",
            weight=1.0,
            request=HybridRetrievalRequest(
                schema_version="hybrid_retrieval_request_v1",
                request_id=f"request-{index}",
                query=query,
                subject="math",
                generation_id=GENERATION_ID,
            ),
        )
        for index, query in enumerate((first_query, second_query), start=1)
    )
    child_result = child_only_retriever.retrieve_children_multi(
        MultiBranchHybridRequest(
            schema_version="multi_branch_hybrid_request_v1",
            request_id="deferred-request",
            generation_id=GENERATION_ID,
            branches=requests,
            cross_branch_rrf_k=20,
            parent_top_k=1,
        )
    )

    assert child_result.status == "ok"
    assert child_result.ranked_parents[0].parent_id == parent.parent_id
    hydrator = _Hydrator({parent.parent_id: parent}, None)
    hydration_retriever = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=vector,
        bm25_search=bm25,
        reranker=reranker,
        parent_hydrator=hydrator,
    )
    contexts = hydration_retriever.hydrate_kept_multi(
        child_result,
        (child.metadata.child_id,),
    )

    assert hydrator.calls == [(parent.parent_id,)]
    assert tuple(context.parent.parent_id for context in contexts) == (
        parent.parent_id,
    )
    assert contexts[0].supporting_child_ids == (child.metadata.child_id,)


def test_multi_branch_judge_refs_respect_supporting_child_cap() -> None:
    parent = _parent(
        30,
        content="# Capped\nalpha beta gamma delta epsilon zeta",
        source_relpath="math/capped.md",
        section_title="Capped",
    )
    children = (
        _child(31, parent, 9, 14, child_id=None),
        _child(32, parent, 15, 19, child_id=None),
        _child(33, parent, 20, 25, child_id=None),
    )
    query = "capped support"
    vector = _QuerySearch(
        {query: tuple(_search_candidate(child, 1.0) for child in children)}
    )
    retriever = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=vector,
        bm25_search=_QuerySearch({query: ()}),
        reranker=_ScoreReranker(
            {
                child.metadata.child_id: score
                for child, score in zip(children, (0.9, 0.8, 0.7), strict=True)
            }
        ),
        parent_hydrator=_MustNotHydrate(),
    )
    request = HybridRetrievalRequest(
        schema_version="hybrid_retrieval_request_v1",
        request_id="request-cap",
        query=query,
        subject="math",
        generation_id=GENERATION_ID,
    )
    result = retriever.retrieve_children_multi(
        MultiBranchHybridRequest(
            schema_version="multi_branch_hybrid_request_v1",
            request_id="multi-cap",
            generation_id=GENERATION_ID,
            branches=(
                WeightedHybridBranch(
                    schema_version="weighted_hybrid_branch_v1",
                    branch_id="cap",
                    weight=1.0,
                    request=request,
                ),
            ),
            cross_branch_rrf_k=20,
            parent_top_k=1,
        )
    )

    refs = build_multi_local_evidence_refs(result, preview_max_chars=20)

    assert tuple(ref.child_id for ref in refs) == tuple(
        child.metadata.child_id for child in children[:2]
    )
    with pytest.raises(ParentHydrationError, match="outside selected parent support"):
        retriever.hydrate_kept_multi(result, (children[2].metadata.child_id,))


def test_multi_branch_fusion_reserves_configured_subject_coverage() -> None:
    math_parent = _parent(
        11,
        content="# Math\nstrong math context",
        source_relpath="math/source-a.md",
        section_title="Math",
    )
    other_math_parent = _parent(
        12,
        content="# Math 2\nanother strong context",
        source_relpath="math/source-b.md",
        section_title="Math 2",
    )
    python_parent = _parent(
        13,
        content="# Python\nrelevant python context",
        source_relpath="python/source.md",
        section_title="Python",
        subject="python",
    )
    math_child = _child(11, math_parent, 7, 13, child_id=None)
    other_math_child = _child(12, other_math_parent, 9, 16, child_id=None)
    python_child = _child(13, python_parent, 9, 17, child_id=None)
    vector = _FlexibleQuerySearch(
        {
            "math query": (
                _search_candidate(math_child, 1.0),
                _search_candidate(other_math_child, 0.9),
            ),
            "python query": (_search_candidate(python_child, 0.1),),
        }
    )
    bm25 = _FlexibleQuerySearch({"math query": (), "python query": ()})
    retriever = ParentChildHybridRetriever(
        policy=_policy(),
        vector_search=vector,
        bm25_search=bm25,
        reranker=_ScoreReranker(
            {
                math_child.metadata.child_id: 0.99,
                other_math_child.metadata.child_id: 0.98,
                python_child.metadata.child_id: 0.2,
            }
        ),
        parent_hydrator=_Hydrator(
            {
                math_parent.parent_id: math_parent,
                other_math_parent.parent_id: other_math_parent,
                python_parent.parent_id: python_parent,
            },
            None,
        ),
    )
    branches = tuple(
        WeightedHybridBranch(
            schema_version="weighted_hybrid_branch_v1",
            branch_id=subject,
            weight=1.0,
            request=HybridRetrievalRequest(
                schema_version="hybrid_retrieval_request_v1",
                request_id=f"request-{subject}",
                query=f"{subject} query",
                subject=subject,
                generation_id=GENERATION_ID,
            ),
        )
        for subject in ("math", "python")
    )

    result = retriever.retrieve_multi(
        MultiBranchHybridRequest(
            schema_version="multi_branch_hybrid_request_v1",
            request_id="coverage-request",
            generation_id=GENERATION_ID,
            branches=branches,
            cross_branch_rrf_k=20,
            parent_top_k=2,
        )
    )

    assert {parent.subject for parent in result.ranked_parents} == {"math", "python"}
