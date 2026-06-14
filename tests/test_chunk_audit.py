from __future__ import annotations

from langchain_core.documents import Document

from src.rag.audit import audit_chunks


def test_audit_chunks_returns_statistics_without_modifying_documents():
    docs = [
        Document(page_content="short", metadata={"subject": "s", "source_file": "a.txt", "doc_type": "notes"}),
        Document(page_content="Repeated chunk text " * 10, metadata={"subject": "s", "source_file": "a.txt", "doc_type": "notes"}),
        Document(page_content="Repeated chunk text " * 10, metadata={"subject": "s", "source_file": "b.txt", "doc_type": "notes"}),
        Document(page_content="", metadata={"source_file": "b.txt"}),
    ]
    before = [(doc.page_content, dict(doc.metadata)) for doc in docs]

    report = audit_chunks(docs, min_chars=20, max_chars=120)

    assert report.total_chunks == 4
    assert report.source_count == 2
    assert report.too_short_count == 2
    assert report.empty_chunk_count == 1
    assert report.duplicate_chunk_count == 1
    assert report.missing_metadata_counts["subject"] == 1
    assert "duplicate_chunks_detected" in report.warnings
    assert [(doc.page_content, doc.metadata) for doc in docs] == before


def test_audit_chunks_serializes_to_json_ready_dict():
    docs = [
        Document(page_content="A useful paragraph " * 20, metadata={"subject": "x", "source_file": "x.txt", "doc_type": "notes"})
    ]

    payload = audit_chunks(docs).to_dict()

    assert payload["total_chunks"] == 1
    assert isinstance(payload["per_source"], list)
    assert payload["per_source"][0]["source_file"] == "x.txt"
