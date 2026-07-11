from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.rag.parent_child.bm25_artifact import Bm25CorpusRow
from src.rag.parent_child.chroma_children import write_child_chroma_artifact
from src.rag.parent_child.models import ChildDocument, ChildMetadata
from src.rag.parent_child.retrieval import RetrievalInvariantError
from src.rag.parent_child.runtime_resources import (
    ChromaChildSearchChannel,
    SubjectBm25Router,
    SubjectBm25SearchChannel,
)


def _child(child_hex: str, content: str, *, section_title: str) -> ChildDocument:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
    return ChildDocument(
        schema_version="child_document_v1",
        content=content,
        metadata=ChildMetadata(
            schema_version="child_metadata_v1",
            child_id="child_" + child_hex * 40,
            parent_id="parent_" + child_hex * 40,
            doc_id="doc_" + child_hex * 40,
            subject="math",
            generation_id="gen-a",
            policy_id="d" * 64,
            child_index=0,
            child_start_in_parent=0,
            child_end_in_parent=len(content),
            start_char=0,
            end_char=len(content),
            child_chars=len(content),
            content_sha1=digest,
            source_file=f"{child_hex}.md",
            source_relpath=f"math/{child_hex}.md",
            source_file_sha1="e" * 40,
            doc_type="notes",
            section_id="section_" + child_hex * 40,
            section_title=section_title,
            section_path=(section_title,) if section_title else (),
            pagination_kind="logical",
            page_start=1,
            page_end=1,
        ),
    )


class _DocumentEmbedding:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        mapping = {"alpha theorem": [1.0, 0.0], "beta proof": [0.0, 1.0]}
        return [mapping[text] for text in texts]


class _QueryEmbedding:
    def embed_query(self, text: str) -> list[float]:
        assert text == "alpha"
        return [1.0, 0.0]


def test_sealed_chroma_vector_and_subject_bm25_resources_round_trip(
    tmp_path: Path,
) -> None:
    staging_root = tmp_path / "indexes" / ".staging" / "gen-a"
    staging_root.mkdir(parents=True)
    alpha = _child("a", "alpha theorem", section_title="")
    beta = _child("b", "beta proof", section_title="Proof")
    write_child_chroma_artifact(
        (alpha, beta),
        generation_staging_root=staging_root,
        persist_directory=staging_root / "chroma_children",
        generation_id="gen-a",
        collection_name="children-v1",
        distance_metric="cosine",
        expected_dimension=2,
        batch_size=2,
        embedding_provider=_DocumentEmbedding(),
    )
    vector = ChromaChildSearchChannel(
        persist_directory=(staging_root / "chroma_children").resolve(),
        collection_name="children-v1",
        generation_id="gen-a",
        expected_dimension=2,
        distance_metric="cosine",
        query_embedding_provider=_QueryEmbedding(),
    )
    try:
        vector_hits = vector.search(
            query="alpha",
            subject="math",
            generation_id="gen-a",
            top_k=2,
        )
        assert vector_hits[0].document == alpha

        rows = (
            Bm25CorpusRow(
                schema_version="bm25_row_v1",
                generation_id="gen-a",
                subject="math",
                child_id=alpha.metadata.child_id,
                tokens=("alpha", "theorem"),
            ),
            Bm25CorpusRow(
                schema_version="bm25_row_v1",
                generation_id="gen-a",
                subject="math",
                child_id=beta.metadata.child_id,
                tokens=("beta", "proof"),
            ),
        )
        bm25 = SubjectBm25SearchChannel(
            generation_id="gen-a",
            subject="math",
            rows=rows,
            tokenizer=lambda query: tuple(query.split()),
            child_lookup=vector,
        )
        bm25_hits = bm25.search(
            query="alpha",
            subject="math",
            generation_id="gen-a",
            top_k=2,
        )
        assert tuple(hit.document for hit in bm25_hits) == (alpha,)
    finally:
        vector.close()


def test_bm25_router_rejects_unknown_subject() -> None:
    alpha = _child("a", "alpha theorem", section_title="Limits")

    class _Lookup:
        def get_children(self, child_ids: tuple[str, ...]) -> tuple[ChildDocument, ...]:
            assert child_ids == (alpha.metadata.child_id,)
            return (alpha,)

    channel = SubjectBm25SearchChannel(
        generation_id="gen-a",
        subject="math",
        rows=(
            Bm25CorpusRow(
                schema_version="bm25_row_v1",
                generation_id="gen-a",
                subject="math",
                child_id=alpha.metadata.child_id,
                tokens=("alpha",),
            ),
        ),
        tokenizer=lambda query: (query,),
        child_lookup=_Lookup(),
    )
    router = SubjectBm25Router({"math": channel})

    with pytest.raises(RetrievalInvariantError, match="unknown subject"):
        router.search(
            query="alpha",
            subject="python",
            generation_id="gen-a",
            top_k=1,
        )
