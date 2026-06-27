from __future__ import annotations

from pathlib import Path
import tempfile

from langchain_core.documents import Document
import pytest

from src.rag.chunking.structure_splitter import split_documents_by_structure
from src.rag.loader import load_documents

SCALAR_TYPES = (str, int, float, bool, type(None))
SAMPLE_TEXT = """Course introduction
This is preamble text.

# Chapter One
Alpha section body about data.
Alpha section details stay in chapter one.
Alpha section has enough explanatory content to avoid short-section merging.
Alpha section keeps its own metadata in chapter one.

## 1.1 Data Types
Beta section body about types.
Beta section details stay in data types.
Beta section has enough explanatory content to avoid short-section merging.
Beta section keeps its own metadata in data types.

# Chapter Two
Gamma section body about processing.
Gamma section details stay in chapter two.
Gamma section has enough explanatory content to avoid short-section merging.
Gamma section keeps its own metadata in chapter two.
"""


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def _source_doc(text: str = SAMPLE_TEXT) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": "doc_sample",
            "subject": "python",
            "source_file": "sample.txt",
        },
    )


def test_structure_splitter_adds_scalar_section_metadata_and_preserves_preamble():
    chunks = split_documents_by_structure(
        [_source_doc()], chunk_size=120, chunk_overlap=20
    )

    assert len(chunks) >= 4
    titles = [chunk.metadata["section_title"] for chunk in chunks]
    assert "Chapter One" in titles
    assert "1.1 Data Types" in titles
    assert "Chapter Two" in titles
    for chunk in chunks:
        metadata = chunk.metadata
        assert metadata["splitter_mode"] == "structure"
        assert metadata["section_id"].startswith("sec_")
        assert isinstance(metadata["section_path"], str)
        assert not isinstance(metadata["section_path"], (tuple, list, dict))
        assert all(isinstance(value, SCALAR_TYPES) for value in metadata.values())
    assert any("This is preamble text." in chunk.page_content for chunk in chunks)
    assert any(
        "Preamble" in chunk.metadata["merged_section_titles"] for chunk in chunks
    )


def test_structure_splitter_does_not_merge_across_sections():
    chunks = split_documents_by_structure(
        [_source_doc()], chunk_size=120, chunk_overlap=20
    )

    for chunk in chunks:
        title = chunk.metadata["section_title"]
        if title == "Chapter One":
            assert "Beta section body" not in chunk.page_content
            assert "Gamma section body" not in chunk.page_content
        if title == "1.1 Data Types":
            assert "Alpha section body" not in chunk.page_content
            assert "Gamma section body" not in chunk.page_content
        if title == "Chapter Two":
            assert "Alpha section body" not in chunk.page_content
            assert "Beta section body" not in chunk.page_content


def test_structure_splitter_merges_toc_entries_with_only_page_numbers_without_dropping():
    chunks = split_documents_by_structure(
        [
            _source_doc(
                "A. Table Entry\n"
                "12\n\n"
                "# Real Section\n"
                "Actual body text should be indexed.\n"
            )
        ],
        chunk_size=120,
        chunk_overlap=20,
    )

    titles = [chunk.metadata["section_title"] for chunk in chunks]
    assert "Table Entry" not in titles
    assert "Real Section" in titles
    assert any("A. Table Entry" in chunk.page_content for chunk in chunks)
    assert any(
        "Table Entry" in chunk.metadata["merged_section_titles"] for chunk in chunks
    )


def test_structure_splitter_section_id_is_stable():
    first = split_documents_by_structure(
        [_source_doc()], chunk_size=120, chunk_overlap=20
    )
    second = split_documents_by_structure(
        [_source_doc()], chunk_size=120, chunk_overlap=20
    )

    assert [chunk.metadata["section_id"] for chunk in first] == [
        chunk.metadata["section_id"] for chunk in second
    ]


def test_structure_splitter_fallback_document_metadata():
    chunks = split_documents_by_structure(
        [_source_doc("Useful body without headings. " * 20)],
        chunk_size=120,
        chunk_overlap=20,
    )

    assert chunks
    assert {chunk.metadata["section_title"] for chunk in chunks} == {"Document"}
    assert {chunk.metadata["section_level"] for chunk in chunks} == {0}
    assert {chunk.metadata["section_path"] for chunk in chunks} == {"Document"}


def test_load_documents_structure_mode_uses_structure_policy_and_global_chunk_index(
    local_tmp_path, monkeypatch
):
    monkeypatch.setenv("RAG_SPLITTER_MODE", "structure")
    source = local_tmp_path / "notes_2026.txt"
    source.write_text(
        "# Long Section\n"
        + ("Alpha body content. " * 120)
        + "\n# Short Section\nBeta body.",
        encoding="utf-8",
    )

    docs = load_documents(local_tmp_path, subject="general", doc_type="course_material")

    assert docs
    assert [doc.metadata["chunk_index"] for doc in docs] == list(range(len(docs)))
    assert {doc.metadata["chunk_policy_version"] for doc in docs} == {"structure_v1"}
    by_section: dict[str, list[int]] = {}
    for doc in docs:
        by_section.setdefault(doc.metadata["section_id"], []).append(
            doc.metadata["section_chunk_index"]
        )
    for section_indexes in by_section.values():
        assert section_indexes == list(range(len(section_indexes)))
