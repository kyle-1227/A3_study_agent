from __future__ import annotations

from langchain_core.documents import Document

from src.rag.chunking.validator import (
    ChunkQualityConfig,
    is_noise_chunk,
    validate_chunks,
)

SCALAR_TYPES = (str, int, float, bool, type(None))


def test_is_noise_chunk_detects_empty_and_page_markers():
    assert is_noise_chunk("")
    assert is_noise_chunk("   ")
    assert is_noise_chunk("12")
    assert is_noise_chunk("- 12 -")
    assert is_noise_chunk("Page 12")
    assert is_noise_chunk("第 12 页")


def test_is_noise_chunk_preserves_short_meaningful_text():
    assert not is_noise_chunk("RDD")
    assert not is_noise_chunk("变量")
    assert not is_noise_chunk("x = 1")
    assert not is_noise_chunk("a^2 + b^2 = c^2")
    assert not is_noise_chunk("def hello():")
    assert not is_noise_chunk("SQL SELECT")


def test_validate_chunks_diagnostic_mode_does_not_modify_documents():
    docs = [
        Document(
            page_content="Page 12", metadata={"source_file": "a.txt", "chunk_id": "c1"}
        ),
        Document(
            page_content="RDD", metadata={"source_file": "a.txt", "chunk_index": 1}
        ),
    ]
    before = [(doc.page_content, dict(doc.metadata)) for doc in docs]

    validated_docs, report = validate_chunks(docs, apply_changes=False)

    assert validated_docs == docs
    assert [(doc.page_content, doc.metadata) for doc in docs] == before
    assert report.input_count == 2
    assert report.output_count == 2
    assert report.dropped_count == 0
    assert report.noise_count == 1


def test_validate_chunks_apply_changes_drops_empty_and_noise_chunks():
    docs = [
        Document(page_content="", metadata={"source_file": "a.txt"}),
        Document(page_content="Page 12", metadata={"source_file": "a.txt"}),
        Document(
            page_content="Useful paragraph with enough signal.",
            metadata={"source_file": "a.txt"},
        ),
    ]

    validated_docs, report = validate_chunks(docs, apply_changes=True)

    assert [doc.page_content for doc in validated_docs] == [
        "Useful paragraph with enough signal."
    ]
    assert report.dropped_count == 2
    assert report.output_count == 1


def test_validate_chunks_apply_changes_merges_adjacent_short_chunks_same_source():
    docs = [
        Document(
            page_content="Short concept one",
            metadata={
                "source_file": "a.txt",
                "chunk_id": "old-1",
                "content_sha1": "old",
            },
        ),
        Document(
            page_content="Short concept two",
            metadata={
                "source_file": "a.txt",
                "chunk_id": "old-2",
                "content_sha1": "old",
            },
        ),
    ]

    validated_docs, report = validate_chunks(docs, apply_changes=True)

    assert len(validated_docs) == 1
    assert "Short concept one" in validated_docs[0].page_content
    assert "Short concept two" in validated_docs[0].page_content
    assert "chunk_id" not in validated_docs[0].metadata
    assert "content_sha1" not in validated_docs[0].metadata
    assert validated_docs[0].metadata["validation_requires_rehash"] is True
    assert validated_docs[0].metadata["validation_merged_chunk_count"] == 2
    assert report.merged_count == 1
    assert all(
        isinstance(value, SCALAR_TYPES) for value in validated_docs[0].metadata.values()
    )


def test_validate_chunks_does_not_merge_different_sources():
    docs = [
        Document(page_content="Short concept one", metadata={"source_file": "a.txt"}),
        Document(page_content="Short concept two", metadata={"source_file": "b.txt"}),
    ]

    validated_docs, report = validate_chunks(docs, apply_changes=True)

    assert len(validated_docs) == 2
    assert report.merged_count == 0


def test_validate_chunks_does_not_merge_over_max_chars():
    docs = [
        Document(page_content="alpha beta gamma", metadata={"source_file": "a.txt"}),
        Document(page_content="delta epsilon zeta", metadata={"source_file": "a.txt"}),
    ]
    config = ChunkQualityConfig(merge_below_chars=100, max_merged_chars=20)

    validated_docs, report = validate_chunks(docs, config=config, apply_changes=True)

    assert len(validated_docs) == 2
    assert report.merged_count == 0
