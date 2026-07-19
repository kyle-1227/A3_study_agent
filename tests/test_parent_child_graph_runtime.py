from __future__ import annotations

import asyncio
import hashlib
import json
from typing import get_type_hints
from typing import cast

import pytest

from src.graph.builder import (
    build_graph,
    build_parent_child_graph,
    get_compiled_parent_child_graph,
)
from src.graph.evidence import EvidenceJudgeItem, EvidenceJudgeOutput
from src.graph.parent_child_nodes import (
    ParentChildGraphContractError,
    ParentChildGraphRuntime,
    make_parent_child_hydration_node,
    make_parent_child_rag_node,
)
from src.graph.state import LearningState, context_reducer
from src.rag.parent_child.models import ChildDocument, ChildMetadata, ParentRecord
from src.rag.parent_child.retrieval import (
    BranchParentProvenance,
    ChildEvidenceHit,
    CrossBranchParentHit,
    HybridChildRetrievalResult,
    HydratedParentContext,
    MultiBranchHybridChildResult,
    MultiBranchHybridRequest,
    ParentAggregate,
    ParentContextWindow,
    RetrievalTimings,
)


_GENERATION_ID = "generation-a"
_POLICY_ID = "a" * 64
_PARENT_ID = "parent_" + "1" * 40
_DOC_ID = "doc_" + "2" * 40
_SECTION_ID = "section_" + "3" * 40
_CHILD_ID = "child_" + "4" * 40


def _parent_and_child() -> tuple[ParentRecord, ChildDocument]:
    content = "# Topic\nalpha child text and authoritative parent tail"
    child_start = content.index("alpha")
    child_end = child_start + len("alpha child text")
    parent = ParentRecord(
        schema_version="parent_record_v1",
        parent_id=_PARENT_ID,
        doc_id=_DOC_ID,
        subject="math",
        generation_id=_GENERATION_ID,
        policy_id=_POLICY_ID,
        parent_index=0,
        source_file="notes.md",
        source_relpath="math/notes.md",
        source_file_sha1="b" * 40,
        doc_type="notes",
        extraction_method="fixture_v1",
        cleaning_policy_id="clean_v1",
        section_id=_SECTION_ID,
        section_title="Topic",
        section_path=("Topic",),
        pagination_kind="logical",
        page_start=1,
        page_end=1,
        start_char=100,
        end_char=100 + len(content),
        parent_chars=len(content),
        content_sha1=hashlib.sha1(content.encode("utf-8")).hexdigest(),
        content=content,
    )
    child_content = content[child_start:child_end]
    child = ChildDocument(
        schema_version="child_document_v1",
        content=child_content,
        metadata=ChildMetadata(
            schema_version="child_metadata_v1",
            child_id=_CHILD_ID,
            parent_id=parent.parent_id,
            doc_id=parent.doc_id,
            subject=parent.subject,
            generation_id=parent.generation_id,
            policy_id=parent.policy_id,
            child_index=0,
            child_start_in_parent=child_start,
            child_end_in_parent=child_end,
            start_char=parent.start_char + child_start,
            end_char=parent.start_char + child_end,
            child_chars=len(child_content),
            content_sha1=hashlib.sha1(child_content.encode("utf-8")).hexdigest(),
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
    return parent, child


def _child_for_phrase(
    parent: ParentRecord,
    *,
    child_id: str,
    phrase: str,
    child_index: int,
) -> ChildDocument:
    child_start = parent.content.index(phrase)
    child_end = child_start + len(phrase)
    return ChildDocument(
        schema_version="child_document_v1",
        content=phrase,
        metadata=ChildMetadata(
            schema_version="child_metadata_v1",
            child_id=child_id,
            parent_id=parent.parent_id,
            doc_id=parent.doc_id,
            subject=parent.subject,
            generation_id=parent.generation_id,
            policy_id=parent.policy_id,
            child_index=child_index,
            child_start_in_parent=child_start,
            child_end_in_parent=child_end,
            start_char=parent.start_char + child_start,
            end_char=parent.start_char + child_end,
            child_chars=len(phrase),
            content_sha1=hashlib.sha1(phrase.encode("utf-8")).hexdigest(),
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


class _FakeGraphRetriever:
    def __init__(self, parent: ParentRecord, child: ChildDocument) -> None:
        self.parent = parent
        self.child = child
        self.retrieval_requests: list[MultiBranchHybridRequest] = []
        self.hydration_calls: list[tuple[str, ...]] = []
        self.last_result: MultiBranchHybridChildResult | None = None

    def retrieve_children_multi(
        self,
        request: MultiBranchHybridRequest,
    ) -> MultiBranchHybridChildResult:
        self.retrieval_requests.append(request)
        hit = ChildEvidenceHit(
            schema_version="child_evidence_hit_v1",
            final_rank=1,
            document=self.child,
            vector_rank=1,
            bm25_rank=None,
            vector_raw_score=0.1,
            bm25_raw_score=None,
            rrf_score=0.05,
            rerank_score=0.9,
        )
        branch_results = tuple(
            HybridChildRetrievalResult(
                schema_version="hybrid_child_retrieval_result_v1",
                status="ok",
                request=branch.request,
                retrieval_fingerprint="f" * 64,
                ranked_children=(hit,),
                ranked_parents=(
                    ParentAggregate(
                        schema_version="parent_aggregate_v1",
                        rank=1,
                        parent_id=self.parent.parent_id,
                        subject=self.parent.subject,
                        source_relpath=self.parent.source_relpath,
                        parent_score=0.9,
                        best_child_rank=1,
                        supporting_child_ids=(self.child.metadata.child_id,),
                    ),
                ),
                timings=RetrievalTimings(
                    schema_version="retrieval_timings_v1",
                    vector_ms=1.0,
                    bm25_ms=1.0,
                    reranker_ms=1.0,
                    hydrate_ms=0.0,
                    total_ms=3.0,
                ),
            )
            for branch in request.branches
        )
        result = MultiBranchHybridChildResult(
            schema_version="multi_branch_hybrid_child_result_v1",
            status="ok",
            request=request,
            branch_results=branch_results,
            ranked_parents=(
                CrossBranchParentHit(
                    schema_version="cross_branch_parent_hit_v1",
                    rank=1,
                    parent_id=self.parent.parent_id,
                    subject=self.parent.subject,
                    source_relpath=self.parent.source_relpath,
                    cross_branch_rrf_score=0.1,
                    best_branch_parent_rank=1,
                    provenance=(
                        BranchParentProvenance(
                            schema_version="branch_parent_provenance_v1",
                            branch_id=request.branches[0].branch_id,
                            branch_parent_rank=1,
                            branch_weight=request.branches[0].weight,
                        ),
                    ),
                ),
            ),
        )
        self.last_result = result
        return result

    def hydrate_kept_multi(
        self,
        result: MultiBranchHybridChildResult,
        kept_child_ids: tuple[str, ...],
    ) -> tuple[HydratedParentContext, ...]:
        assert result == self.last_result
        self.hydration_calls.append(tuple(kept_child_ids))
        if not kept_child_ids:
            return ()
        return (
            HydratedParentContext(
                schema_version="hydrated_parent_context_v1",
                rank=1,
                parent=self.parent,
                supporting_child_ids=tuple(kept_child_ids),
                expansion_mode="full_parent",
                heading=self.parent.section_title,
                windows=(
                    ParentContextWindow(
                        schema_version="parent_context_window_v1",
                        start_in_parent=0,
                        end_in_parent=self.parent.parent_chars,
                        content=self.parent.content,
                    ),
                ),
            ),
        )


class _TwoBranchGraphRetriever:
    def __init__(
        self, parent: ParentRecord, children: tuple[ChildDocument, ...]
    ) -> None:
        self.parent = parent
        self.children = children

    def retrieve_children_multi(
        self,
        request: MultiBranchHybridRequest,
    ) -> MultiBranchHybridChildResult:
        assert len(request.branches) == len(self.children) == 2
        branch_results: list[HybridChildRetrievalResult] = []
        provenance: list[BranchParentProvenance] = []
        for branch, child in zip(request.branches, self.children, strict=True):
            hit = ChildEvidenceHit(
                schema_version="child_evidence_hit_v1",
                final_rank=1,
                document=child,
                vector_rank=1,
                bm25_rank=None,
                vector_raw_score=0.1,
                bm25_raw_score=None,
                rrf_score=0.05,
                rerank_score=0.9,
            )
            branch_results.append(
                HybridChildRetrievalResult(
                    schema_version="hybrid_child_retrieval_result_v1",
                    status="ok",
                    request=branch.request,
                    retrieval_fingerprint="f" * 64,
                    ranked_children=(hit,),
                    ranked_parents=(
                        ParentAggregate(
                            schema_version="parent_aggregate_v1",
                            rank=1,
                            parent_id=self.parent.parent_id,
                            subject=self.parent.subject,
                            source_relpath=self.parent.source_relpath,
                            parent_score=0.9,
                            best_child_rank=1,
                            supporting_child_ids=(child.metadata.child_id,),
                        ),
                    ),
                    timings=RetrievalTimings(
                        schema_version="retrieval_timings_v1",
                        vector_ms=1.0,
                        bm25_ms=1.0,
                        reranker_ms=1.0,
                        hydrate_ms=0.0,
                        total_ms=3.0,
                    ),
                )
            )
            provenance.append(
                BranchParentProvenance(
                    schema_version="branch_parent_provenance_v1",
                    branch_id=branch.branch_id,
                    branch_parent_rank=1,
                    branch_weight=branch.weight,
                )
            )
        return MultiBranchHybridChildResult(
            schema_version="multi_branch_hybrid_child_result_v1",
            status="ok",
            request=request,
            branch_results=tuple(branch_results),
            ranked_parents=(
                CrossBranchParentHit(
                    schema_version="cross_branch_parent_hit_v1",
                    rank=1,
                    parent_id=self.parent.parent_id,
                    subject=self.parent.subject,
                    source_relpath=self.parent.source_relpath,
                    cross_branch_rrf_score=0.1,
                    best_branch_parent_rank=1,
                    provenance=tuple(provenance),
                ),
            ),
        )

    def hydrate_kept_multi(
        self,
        result: MultiBranchHybridChildResult,
        kept_child_ids: tuple[str, ...],
    ) -> tuple[HydratedParentContext, ...]:
        raise AssertionError("role mapping test must not hydrate parents")


def _runtime() -> tuple[ParentChildGraphRuntime, _FakeGraphRetriever, ParentRecord]:
    parent, child = _parent_and_child()
    retriever = _FakeGraphRetriever(parent, child)
    return (
        ParentChildGraphRuntime(
            generation_id=_GENERATION_ID,
            primary_revision=1,
            primary_config_fingerprint="e" * 64,
            available_subjects=("math",),
            retriever=retriever,
            retrieval_fingerprint="f" * 64,
            cross_branch_rrf_k=20,
            parent_top_k=1,
            preview_max_chars=100,
        ),
        retriever,
        parent,
    )


def _retrieval_state() -> LearningState:
    return cast(
        LearningState,
        {
            "request_id": "request-a",
            "retrieval_plan": [
                {
                    "subject": "math",
                    "role": "core_concept",
                    "local_retrieval_query": "explain alpha",
                    "purpose": "Answer with local course evidence.",
                    "priority": 1.0,
                    "_parent_child_priority_explicit": True,
                }
            ],
            "rewritten_query": "",
            "retry_count": 0,
        },
    )


def test_candidate_graph_judges_child_then_replaces_it_with_parent_context() -> None:
    runtime, retriever, parent = _runtime()
    rag_output = asyncio.run(make_parent_child_rag_node(runtime)(_retrieval_state()))

    assert retriever.hydration_calls == []
    candidate = rag_output["local_evidence_candidates"][0]
    assert candidate["evidence_id"] == _CHILD_ID
    assert parent.content not in json.dumps(candidate, ensure_ascii=False)
    assert "content" not in candidate["metadata"]

    judged = EvidenceJudgeOutput(
        overall_evidence_state="sufficient",
        need_more_web_research=False,
        judged_evidence=[
            EvidenceJudgeItem(
                evidence_id=_CHILD_ID,
                keep=True,
                final_quality="high",
                relevance="high",
                authority="high",
                usefulness="high",
                risk="low",
                evidence_score=0.95,
                score_reason="The child directly answers the query.",
                evidence_type="local_textbook_chunk",
                use_case="core_evidence",
                coverage_contribution="It explains the requested alpha concept.",
                reason="Keep the precise local evidence.",
            )
        ],
        coverage_gaps=[],
        decision_summary="Sufficient local evidence.",
    )
    child_context = [
        {
            **rag_output["local_evidence_originals"][_CHILD_ID],
            "evidence_id": _CHILD_ID,
            "judge_keep": True,
            "evidence_score": 0.95,
        }
    ]
    hydration_state = cast(
        LearningState,
        {
            **_retrieval_state(),
            **rag_output,
            "evidence_candidates": rag_output["local_evidence_candidates"],
            "evidence_judge_output": judged.model_dump(mode="json"),
            "context": child_context,
        },
    )
    hydration_output = asyncio.run(
        make_parent_child_hydration_node(runtime)(hydration_state)
    )
    final_context = context_reducer(child_context, hydration_output["context"])

    assert retriever.hydration_calls == [(_CHILD_ID,)]
    assert len(final_context) == 1
    assert final_context[0]["parent_id"] == parent.parent_id
    assert final_context[0]["supporting_child_ids"] == [_CHILD_ID]
    assert final_context[0]["supporting_children"] == [
        {
            "child_id": _CHILD_ID,
            "page_start": 1,
            "page_end": 1,
            "start_char": 108,
            "end_char": 124,
            "child_start_in_parent": 8,
            "child_end_in_parent": 24,
        }
    ]
    assert (
        final_context[0]["retrieval_fingerprint"] == runtime.graph_handoff_fingerprint
    )
    assert "authoritative parent tail" in final_context[0]["content"]
    assert hydration_output["graded_evidence"] == final_context


def test_candidate_graph_factory_adds_hydration_without_changing_legacy_graph(
    learning_guidance_runtime,
) -> None:
    runtime, _retriever, _parent = _runtime()

    assert (
        "parent_child_parent_hydration"
        not in build_graph(learning_guidance_runtime).nodes
    )
    candidate_graph = build_parent_child_graph(
        runtime,
        learning_guidance_runtime=learning_guidance_runtime,
    )
    assert "parent_child_parent_hydration" in candidate_graph.nodes
    assert candidate_graph.compile() is not None

    assert "runtime" in get_type_hints(build_parent_child_graph)
    assert "learning_guidance_runtime" in get_type_hints(build_parent_child_graph)
    assert "runtime" in get_type_hints(get_compiled_parent_child_graph)
    assert "learning_guidance_runtime" in get_type_hints(
        get_compiled_parent_child_graph
    )


def test_candidate_graph_rejects_legacy_defaulted_priority() -> None:
    runtime, _retriever, _parent = _runtime()
    state = dict(_retrieval_state())
    state["retrieval_plan"] = [dict(state["retrieval_plan"][0])]
    state["retrieval_plan"][0].pop("_parent_child_priority_explicit")

    with pytest.raises(
        ParentChildGraphContractError,
        match="violates the candidate contract",
    ):
        asyncio.run(make_parent_child_rag_node(runtime)(cast(LearningState, state)))


def test_candidate_hydration_rejects_checkpoint_policy_mismatch() -> None:
    runtime, _retriever, _parent = _runtime()
    rag_output = asyncio.run(make_parent_child_rag_node(runtime)(_retrieval_state()))
    state = cast(
        LearningState,
        {
            **_retrieval_state(),
            **rag_output,
            "parent_child_retrieval_fingerprint": "0" * 64,
        },
    )

    with pytest.raises(
        ParentChildGraphContractError,
        match="handoff policy changed in-flight",
    ):
        asyncio.run(make_parent_child_hydration_node(runtime)(state))


def test_multi_branch_children_keep_their_own_role_and_purpose() -> None:
    parent, first_child = _parent_and_child()
    second_child = _child_for_phrase(
        parent,
        child_id="child_" + "5" * 40,
        phrase="authoritative parent tail",
        child_index=1,
    )
    retriever = _TwoBranchGraphRetriever(parent, (first_child, second_child))
    runtime = ParentChildGraphRuntime(
        generation_id=_GENERATION_ID,
        primary_revision=1,
        primary_config_fingerprint="e" * 64,
        available_subjects=("math",),
        retriever=retriever,
        retrieval_fingerprint="f" * 64,
        cross_branch_rrf_k=20,
        parent_top_k=1,
        preview_max_chars=100,
    )
    state = cast(
        LearningState,
        {
            "request_id": "request-multi",
            "retrieval_plan": [
                {
                    "subject": "math",
                    "role": "core_concept",
                    "local_retrieval_query": "explain alpha",
                    "purpose": "Explain the definition.",
                    "priority": 1.0,
                    "_parent_child_priority_explicit": True,
                },
                {
                    "subject": "math",
                    "role": "practice",
                    "local_retrieval_query": "apply the parent tail",
                    "purpose": "Support the practice step.",
                    "priority": 0.8,
                    "_parent_child_priority_explicit": True,
                },
            ],
            "rewritten_query": "",
            "retry_count": 0,
        },
    )

    output = asyncio.run(make_parent_child_rag_node(runtime)(state))
    candidate_by_id = {
        candidate["evidence_id"]: candidate
        for candidate in output["local_evidence_candidates"]
    }

    assert candidate_by_id[first_child.metadata.child_id]["role"] == "core_concept"
    assert candidate_by_id[first_child.metadata.child_id]["purpose"] == (
        "Explain the definition."
    )
    assert candidate_by_id[second_child.metadata.child_id]["role"] == "practice"
    assert candidate_by_id[second_child.metadata.child_id]["purpose"] == (
        "Support the practice step."
    )
