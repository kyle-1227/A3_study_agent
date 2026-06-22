from __future__ import annotations

from pathlib import Path
import tempfile

from langchain_core.documents import Document
import pytest

from src.rag.audit import audit_chunks
import scripts.audit_chunks as audit_script


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def test_audit_chunks_returns_statistics_without_modifying_documents():
    docs = [
        Document(
            page_content="short",
            metadata={"subject": "s", "source_file": "a.txt", "doc_type": "notes"},
        ),
        Document(
            page_content="Repeated chunk text " * 10,
            metadata={"subject": "s", "source_file": "a.txt", "doc_type": "notes"},
        ),
        Document(
            page_content="Repeated chunk text " * 10,
            metadata={"subject": "s", "source_file": "b.txt", "doc_type": "notes"},
        ),
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


def test_audit_chunks_serializes_to_json_ready_dict_with_old_fields():
    docs = [
        Document(
            page_content="A useful paragraph " * 20,
            metadata={"subject": "x", "source_file": "x.txt", "doc_type": "notes"},
        )
    ]

    payload = audit_chunks(docs).to_dict()

    for key in [
        "total_chunks",
        "source_count",
        "min_chars",
        "max_chars",
        "avg_chars",
        "too_short_count",
        "too_long_count",
        "empty_chunk_count",
        "duplicate_chunk_count",
        "missing_metadata_counts",
        "per_source",
        "warnings",
    ]:
        assert key in payload
    assert payload["total_chunks"] == 1
    assert isinstance(payload["per_source"], list)
    assert payload["per_source"][0]["source_file"] == "x.txt"


def test_audit_chunks_adds_short_chunk_samples_with_truncated_preview():
    long_short_text = "x" * 220
    docs = [
        Document(
            page_content=long_short_text,
            metadata={"source_file": "short.txt", "chunk_index": 7},
        )
    ]

    payload = audit_chunks(docs, min_chars=300).to_dict()

    assert payload["short_chunk_samples"]
    sample = payload["short_chunk_samples"][0]
    assert sample["source_file"] == "short.txt"
    assert sample["chunk_index"] == 7
    assert sample["chunk_chars"] == 220
    assert len(sample["preview"]) == 160
    assert (
        payload["per_source"][0]["short_chunk_samples"][0]["preview"]
        == sample["preview"]
    )


def test_audit_chunks_detects_single_chunk_large_source():
    docs = [
        Document(
            page_content="A large source extracted into one chunk.",
            metadata={
                "source_file": "large.pdf",
                "source_relpath": "data/python/large.pdf",
                "source_file_size": 2_000_000,
            },
        )
    ]

    payload = audit_chunks(docs).to_dict()

    assert payload["suspicious_source_files"]
    suspicious = payload["suspicious_source_files"][0]
    assert suspicious["source_file"] == "large.pdf"
    assert suspicious["source_relpath"] == "data/python/large.pdf"
    assert suspicious["source_file_size"] == 2_000_000
    assert suspicious["chunk_count"] == 1
    assert "single_chunk_large_source" in suspicious["reasons"]
    assert "very_low_chunk_count_for_large_source" in suspicious["reasons"]


def test_audit_chunks_handles_missing_phase4a_metadata():
    docs = [Document(page_content="legacy document text", metadata={})]

    payload = audit_chunks(docs).to_dict()

    assert payload["source_count"] == 1
    assert payload["per_source"][0]["source_file"] == "unknown"
    assert payload["per_source"][0]["source_relpath"] == "unknown"
    assert payload["per_source"][0]["source_file_size"] == 0
    assert payload["short_chunk_samples"][0]["chunk_index"] == 0


def test_audit_chunks_script_skips_needs_ocr_directory(local_tmp_path, monkeypatch):
    data_dir = local_tmp_path / "data"
    formal_dir = data_dir / "python"
    quarantined_dir = data_dir / "_needs_ocr"
    formal_dir.mkdir(parents=True)
    quarantined_dir.mkdir(parents=True)
    (formal_dir / "formal.pdf").write_bytes(b"%PDF formal")
    (quarantined_dir / "quarantined.pdf").write_bytes(b"%PDF quarantined")
    calls: list[str] = []

    def fake_load_documents(directory, *, subject, doc_type):
        calls.append(subject)
        return [
            Document(
                page_content="formal text",
                metadata={"source_file": f"{subject}.pdf"},
            )
        ]

    monkeypatch.setattr(audit_script, "DATA_DIR", data_dir)
    monkeypatch.setattr(audit_script, "load_documents", fake_load_documents)

    docs, skipped = audit_script._load_all_documents()

    assert calls == ["python"]
    assert len(docs) == 1
    assert skipped == [
        {
            "subject": "_needs_ocr",
            "reason": "quarantined OCR-needed directory",
        }
    ]
