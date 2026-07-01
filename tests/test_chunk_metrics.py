from __future__ import annotations

from langchain_core.documents import Document

from src.rag.eval.chunk_metrics import (
    ChunkMetricsConfig,
    chunk_hash,
    duplicate_flags,
    evaluate_documents,
)
from src.rag.ids import normalize_for_hash, sha1_text
from src.rag.metadata_schema import REQUIRED_EVALUATION_METADATA_FIELDS


def _metadata(**overrides):
    metadata = {
        "doc_id": "doc_1",
        "chunk_id": "chunk_1",
        "source_relpath": "data/test/source.txt",
        "source_file": "source.txt",
        "source_file_sha1": "file_sha1",
        "source_file_size": 123,
        "subject": "test",
        "doc_type": "course_material",
        "chunk_index": 0,
        "chunk_policy_version": "recursive_v1",
        "index_version": "a3_rag_v1",
        "content_sha1": "content_1",
        "chunk_chars": 10,
    }
    metadata.update(overrides)
    return metadata


def test_duplicate_flags_prefer_content_sha1_and_mark_non_first():
    docs = [
        Document(page_content="first text", metadata=_metadata(content_sha1="same")),
        Document(
            page_content="different text same hash",
            metadata=_metadata(chunk_id="chunk_2", chunk_index=1, content_sha1="same"),
        ),
        Document(
            page_content="fallback hash text",
            metadata={
                key: value
                for key, value in _metadata(chunk_id="chunk_3", chunk_index=2).items()
                if key != "content_sha1"
            },
        ),
        Document(
            page_content="fallback   hash text",
            metadata={
                key: value
                for key, value in _metadata(chunk_id="chunk_4", chunk_index=3).items()
                if key != "content_sha1"
            },
        ),
    ]

    assert chunk_hash(docs[0]) == "same"
    assert chunk_hash(docs[2]) == sha1_text(normalize_for_hash(docs[2].page_content))
    assert duplicate_flags(docs) == [False, True, False, True]


def test_evaluate_documents_computes_summary_and_metadata_coverage():
    docs = [
        Document(page_content="short", metadata=_metadata(content_sha1="a")),
        Document(
            page_content="A useful chunk body " * 5,
            metadata=_metadata(chunk_id="chunk_2", chunk_index=1, content_sha1="b"),
        ),
        Document(
            page_content="A useful chunk body " * 5,
            metadata=_metadata(chunk_id="chunk_3", chunk_index=2, content_sha1="b"),
        ),
        Document(
            page_content="",
            metadata={
                key: value
                for key, value in _metadata(chunk_id="chunk_4", chunk_index=3).items()
                if key != "doc_id"
            },
        ),
    ]

    payload = evaluate_documents(
        docs,
        config=ChunkMetricsConfig(too_short_chars=10, too_long_chars=200),
    )

    assert payload["summary"]["total_chunks"] == 4
    assert payload["summary"]["too_short_count"] == 2
    assert payload["summary"]["empty_chunk_count"] == 1
    assert payload["summary"]["duplicate_chunk_count"] == 1
    assert payload["summary"]["p10_chars"] == 0
    assert payload["summary"]["p90_chars"] == len("A useful chunk body " * 5)
    assert payload["metadata"]["missing_metadata_counts"]["doc_id"] == 1
    assert payload["metadata"]["required_metadata_fields"] == list(
        REQUIRED_EVALUATION_METADATA_FIELDS
    )
    assert "missing_required_metadata" in payload["warnings"]


def test_evaluate_documents_reports_section_metrics():
    docs = [
        Document(
            page_content="Overview",
            metadata=_metadata(
                section_id="sec_1",
                section_title="Overview",
                section_chunk_index=0,
            ),
        ),
        Document(
            page_content="Preamble body text.",
            metadata=_metadata(
                chunk_id="chunk_2",
                chunk_index=1,
                content_sha1="content_2",
                section_id="sec_0",
                section_title="Preamble",
                section_chunk_index=0,
            ),
        ),
    ]

    payload = evaluate_documents(docs)

    assert payload["metadata"]["section_metadata_coverage"] == 1.0
    assert payload["structure"]["chunks_with_section_id"] == 2
    assert payload["structure"]["unique_section_count"] == 2
    assert payload["structure"]["section_title_only_chunk_count"] == 1
    assert payload["structure"]["preamble_chunk_count"] == 1


def test_evaluate_documents_handles_empty_input():
    payload = evaluate_documents([])

    assert payload["summary"]["total_chunks"] == 0
    assert payload["summary"]["source_count"] == 0
    assert payload["summary"]["duplicate_ratio"] == 0.0
    assert payload["metadata"]["required_metadata_coverage"] == 1.0
    assert payload["per_source"] == []
