from __future__ import annotations

import hashlib
from pathlib import Path

from src.rag.parent_child.flat_baseline import (
    FlatBaselineChunkMetadata,
    FlatBaselineDocument,
    FlatBaselineRuntime,
    build_flat_baseline_manifest,
    make_flat_chunk_id,
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


def test_flat_baseline_chroma_round_trip_runs_all_required_channels(
    tmp_path: Path,
) -> None:
    documents = (
        _document(name="c", content="calculus alpha", chunk_index=0),
        _document(name="d", content="calculus beta", chunk_index=1),
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
    )

    runtime = FlatBaselineRuntime(
        persist_directory=persist_directory,
        manifest=manifest,
        query_embedding_provider=embedding,
        reranker=_Reranker(),
        tokenizer=lambda text: tuple(text.split()),
    )
    try:
        result = runtime.retrieve(
            query="calculus",
            subject="math",
            vector_top_k=2,
            bm25_top_k=2,
            reranker_top_n=2,
        )
    finally:
        runtime.close()

    assert tuple(hit.rank for hit in result.hits) == (1, 2)
    assert {hit.document.metadata.source_relpath for hit in result.hits} == {
        "math/c.txt",
        "math/d.txt",
    }
    assert result.vector_ms >= 0.0
    assert result.bm25_ms >= 0.0
    assert result.reranker_ms >= 0.0
