from __future__ import annotations

from langchain_core.documents import Document

from src.rag.chunking.splitter_factory import split_documents_by_mode
from src.rag.chunking.structure_splitter import split_documents_by_structure

SCALAR_TYPES = (str, int, float, bool, type(None))


def _doc(text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": "doc_structure",
            "source_relpath": "data/sample/source.txt",
            "source_file": "source.txt",
            "subject": "sample",
        },
    )


def _single_chunk(text: str) -> Document:
    chunks = split_documents_by_structure(
        [_doc(text)], chunk_size=1000, chunk_overlap=0
    )
    assert chunks
    return chunks[0]


def test_title_only_section_forward_merge_uses_target_title_and_keeps_text():
    chunk = _single_chunk(
        "# Chapter 1\n"
        "\n"
        "## 1.1 Title Only\n"
        "\n"
        "## 1.2 Real Section\n"
        "This is real body content with enough detail for the target section.\n"
        "It should own the final section title.\n"
    )

    assert chunk.metadata["section_title"] == "1.2 Real Section"
    assert "1.1 Title Only" in chunk.page_content
    assert "1.1 Title Only" in chunk.metadata["merged_section_titles"]
    assert chunk.metadata["merge_reason"] == "title_only_forward_merge"
    assert chunk.metadata["merged_section_count"] >= 2


def test_short_preamble_forward_merge_uses_first_body_title_and_keeps_content():
    chunk = _single_chunk(
        "Contents\n"
        "Copyright page\n"
        "\n"
        "# Real Section\n"
        "Useful body content starts here and should own metadata title.\n"
    )

    assert chunk.metadata["section_title"] == "Real Section"
    assert "Contents" in chunk.page_content
    assert "Copyright page" in chunk.page_content
    assert "Preamble" in chunk.metadata["merged_section_titles"]
    assert chunk.metadata["merge_reason"] == "short_preamble_forward_merge"


def test_short_section_merge_writes_scalar_bounded_metadata():
    chunks = split_documents_by_structure(
        [
            _doc(
                "## Brief Note\n"
                "Tiny body.\n"
                "\n"
                "## Full Topic\n" + ("Detailed explanation. " * 12)
            )
        ],
        chunk_size=1000,
        chunk_overlap=0,
    )

    merged = next(chunk for chunk in chunks if "Tiny body." in chunk.page_content)
    metadata = merged.metadata

    assert metadata["merged_section_count"] >= 2
    assert isinstance(metadata["merged_section_ids"], str)
    assert isinstance(metadata["merged_section_titles"], str)
    assert isinstance(metadata["merge_reason"], str)
    assert len(metadata["merged_section_ids"]) <= 500
    assert len(metadata["merged_section_titles"]) <= 300
    assert all(isinstance(value, SCALAR_TYPES) for value in metadata.values())
    assert not any(isinstance(value, (list, dict, set)) for value in metadata.values())


def test_short_section_does_not_cross_major_chapter_boundary():
    chunks = split_documents_by_structure(
        [
            _doc(
                "Chapter 1: Chapter One\n"
                "Small intro.\n\n"
                "Chapter 2: Chapter Two\n" + ("Large body. " * 30)
            )
        ],
        chunk_size=1000,
        chunk_overlap=0,
    )

    chapter_one = next(
        chunk for chunk in chunks if chunk.metadata["section_title"] == "Chapter One"
    )

    assert "Small intro." in chapter_one.page_content
    assert "Chapter Two" not in chapter_one.page_content
    assert chapter_one.metadata["merge_reason"] == "none"


def test_protected_short_content_is_preserved():
    text = (
        "## Code\n"
        "```python\n"
        "x = 1\n"
        "```\n"
        "\n"
        "## Formula\n"
        "E = mc^2\n"
        "\n"
        "## List\n"
        "- first\n"
        "- second\n"
        "\n"
        "## Table\n"
        "| a | b |\n"
        "| 1 | 2 |\n"
        "\n"
        "## Body\n" + ("Normal explanatory body. " * 10)
    )
    chunks = split_documents_by_structure(
        [_doc(text)], chunk_size=1000, chunk_overlap=0
    )
    joined = "\n".join(chunk.page_content for chunk in chunks)

    assert "```python" in joined
    assert "x = 1" in joined
    assert "E = mc^2" in joined
    assert "- first" in joined
    assert "| a | b |" in joined
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_merged_section_id_is_stable():
    text = "## Brief\nTiny body.\n\n## Real\n" + ("Stable body. " * 20)
    first = split_documents_by_structure([_doc(text)], chunk_size=1000, chunk_overlap=0)
    second = split_documents_by_structure(
        [_doc(text)], chunk_size=1000, chunk_overlap=0
    )

    assert [chunk.metadata["section_id"] for chunk in first] == [
        chunk.metadata["section_id"] for chunk in second
    ]


def test_recursive_mode_does_not_add_merge_metadata():
    chunks = split_documents_by_mode(
        [_doc("## Brief\nTiny body.\n\n## Real\n" + ("Body. " * 100))],
        mode="recursive",
        chunk_size=1000,
        chunk_overlap=0,
    )

    assert chunks
    assert all("merged_section_count" not in chunk.metadata for chunk in chunks)
