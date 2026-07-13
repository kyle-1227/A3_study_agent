from __future__ import annotations

import hashlib
from pathlib import Path

from src.rag.parent_child.flat_baseline import (
    FlatBaselineChunkMetadata,
    FlatBaselineDocument,
    FlatBaselineError,
    FlatBaselineRuntime,
    build_flat_baseline_manifest,
    make_flat_chunk_id,
    read_flat_collection_ids,
    write_flat_baseline_collection,
)
from src.rag.parent_child.manifests import EmbeddingManifestIdentity
from src.rag.parent_child.retrieval import RerankCandidate, RerankScore


class _EmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        assert text == "calculus"
        return [1.0, 0.0]


class _Reranker:
    def rerank(
        self, *, query: str, candidates: tuple[RerankCandidate, ...]
    ) -> tuple[RerankScore, ...]:
        assert query == "calculus"
        return tuple(
            RerankScore(
                schema_version="rerank_score_v1",
                child_id=candidate.child_id,
                score=1.0 - index * 0.1,
            )
            for index, candidate in enumerate(candidates)
        )


class _PagedIdentifierCollection:
    def __init__(self, identifiers: tuple[str, ...], *, page_size: int) -> None:
        self._identifiers = identifiers
        self._page_size = page_size
        self.calls: list[tuple[int, int, tuple[object, ...]]] = []

    def get(
        self, *, limit: int | None = None, offset: int | None = None, include: object
    ) -> dict[str, object]:
        if limit is None or offset is None:
            raise AssertionError("bounded collection reads require limit and offset")
        if limit > self._page_size:
            raise AssertionError("collection read exceeded its declared page size")
        if not isinstance(include, list):
            raise AssertionError("collection include must be an explicit list")
        self.calls.append((limit, offset, tuple(include)))
        return {"ids": list(self._identifiers[offset : offset + limit])}


class _LateDuplicateIdentifierCollection(_PagedIdentifierCollection):
    def get(
        self, *, limit: int | None = None, offset: int | None = None, include: object
    ) -> dict[str, object]:
        payload = super().get(limit=limit, offset=offset, include=include)
        if offset == 4:
            payload["ids"] = [self._identifiers[0]]
        return payload


def _document(*, name: str, content: str, chunk_index: int) -> FlatBaselineDocument:
    doc_id = "doc_" + name * 40
    policy_id = "a" * 64
    content_sha1 = hashlib.sha1(content.encode("utf-8")).hexdigest()
    metadata = FlatBaselineChunkMetadata(
        schema_version="flat_baseline_chunk_metadata_v1",
        chunk_id=make_flat_chunk_id(
            doc_id=doc_id,
            policy_id=policy_id,
            start_char=0,
            end_char=len(content),
            content_sha1=content_sha1,
        ),
        doc_id=doc_id,
        subject="math",
        policy_id=policy_id,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(content),
        chunk_chars=len(content),
        content_sha1=content_sha1,
        source_file=f"{name}.txt",
        source_relpath=f"math/{name}.txt",
        source_file_sha1="b" * 40,
        doc_type="text",
        section_path=("Limits",),
        pagination_kind="logical",
        page_start=1,
        page_end=1,
    )
    return FlatBaselineDocument(
        schema_version="flat_baseline_document_v1",
        content=content,
        metadata=metadata,
    )


def test_flat_baseline_chroma_round_trip_pages_full_collection_and_runs_all_required_channels(
    tmp_path: Path,
) -> None:
    documents = (
        _document(name="c", content="calculus alpha", chunk_index=0),
        _document(name="d", content="calculus beta", chunk_index=1),
        _document(name="e", content="calculus gamma", chunk_index=2),
        _document(name="f", content="calculus delta", chunk_index=3),
        _document(name="a", content="calculus epsilon", chunk_index=4),
    )
    manifest = build_flat_baseline_manifest(
        collection_name="flat_test",
        embedding=EmbeddingManifestIdentity(
            provider="test-provider",
            model="test-model",
            base_url_identity="https://example.invalid/v1/embeddings",
            input_types=("document", "query"),
            fingerprint="c" * 64,
            dimension=2,
            distance_metric="cosine",
        ),
        bm25_tokenizer_fingerprint="d" * 64,
        flat_policy_fingerprint="e" * 64,
        source_fingerprint="f" * 64,
        documents=documents,
    )
    persist_directory = tmp_path / "flat-chroma"
    embedding = _EmbeddingProvider()
    write_flat_baseline_collection(
        documents=documents,
        persist_directory=persist_directory,
        manifest=manifest,
        embedding_provider=embedding,
        batch_size=2,
        max_in_flight_batches=1,
    )

    runtime = FlatBaselineRuntime(
        persist_directory=persist_directory,
        manifest=manifest,
        query_embedding_provider=embedding,
        reranker=_Reranker(),
        tokenizer=lambda text: tuple(text.split()),
        read_page_size=2,
    )
    try:
        result = runtime.retrieve(
            query="calculus",
            subject="math",
            vector_top_k=5,
            bm25_top_k=5,
            reranker_top_n=5,
        )
    finally:
        runtime.close()

    assert tuple(hit.rank for hit in result.hits) == (1, 2, 3, 4, 5)
    assert {hit.document.metadata.source_relpath for hit in result.hits} == {
        "math/c.txt",
        "math/d.txt",
        "math/e.txt",
        "math/f.txt",
        "math/a.txt",
    }
    assert result.vector_ms >= 0.0
    assert result.bm25_ms >= 0.0
    assert result.reranker_ms >= 0.0


def test_flat_baseline_full_read_requires_explicit_bounded_pages() -> None:
    collection = _PagedIdentifierCollection(
        ("flat_a", "flat_b", "flat_c", "flat_d", "flat_e"), page_size=2
    )

    identifiers = read_flat_collection_ids(
        collection=collection,
        expected_count=5,
        page_size=2,
    )

    assert identifiers == ("flat_a", "flat_b", "flat_c", "flat_d", "flat_e")
    assert collection.calls == [(2, 0, ()), (2, 2, ()), (1, 4, ())]


def test_flat_baseline_full_read_rejects_late_page_duplicate_identifier() -> None:
    collection = _LateDuplicateIdentifierCollection(
        ("flat_a", "flat_b", "flat_c", "flat_d", "flat_e"), page_size=2
    )

    try:
        read_flat_collection_ids(
            collection=collection,
            expected_count=5,
            page_size=2,
        )
    except FlatBaselineError as exc:
        assert str(exc) == "flat Chroma paged get returned duplicate ID"
    else:
        raise AssertionError("late-page duplicate IDs must fail the full read")
