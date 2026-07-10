from __future__ import annotations

import tempfile
from pathlib import Path

from src.rag.cleaning import clean_document_text
from src.rag.loader import load_documents


def test_clean_document_text_removes_generic_noise_without_domain_rules():
    repeated_footer = "Generated PDF Footer"
    repeated_paragraph = (
        "First useful paragraph with enough detail to keep as source material "
        "while exceeding the duplicate paragraph safety threshold."
    )
    text = f"""
Table of Contents
1. Overview .......... 1
2. Practice .......... 2

{repeated_footer}
Page 1 of 3

{repeated_paragraph}

{repeated_footer}
- 2 -

{repeated_paragraph}

{repeated_footer}
第 3 页

Final useful paragraph.
"""

    cleaned, report = clean_document_text(
        text,
        source_file="generic.pdf",
        subject="subject_a",
        doc_type="course_material",
    )

    assert "Table of Contents" not in cleaned
    assert "Overview .........." not in cleaned
    assert repeated_footer not in cleaned
    assert "Page 1 of 3" not in cleaned
    assert "First useful paragraph" in cleaned
    assert report.removed_toc_lines >= 2
    assert report.removed_page_number_lines >= 3
    assert report.removed_repeated_lines == 3
    assert report.removed_duplicate_paragraphs == 1


def test_clean_document_text_preserves_indentation_and_body_numbers():
    text = """
Example

1
This standalone number is separated like a page marker.

Body list:
1
Do not remove this number when it is adjacent to content.

def sample():
    value = 1
    return value
"""

    cleaned, _ = clean_document_text(text)

    assert "Body list:\n1\nDo not remove" in cleaned
    assert "def sample():\n    value = 1\n    return value" in cleaned


def test_cleaning_metadata_is_scalar_only():
    _, report = clean_document_text("A\n\n\nB\n", source_file="x.txt")
    metadata = report.to_metadata()

    assert metadata
    assert all(key.startswith("cleaning_") for key in metadata)
    assert all(
        isinstance(value, (str, int, float, bool)) for value in metadata.values()
    )


def test_same_text_cleans_same_way_for_different_subjects():
    text = "Contents\nA .......... 1\n\nUseful body."

    cleaned_a, _ = clean_document_text(text, subject="alpha")
    cleaned_b, _ = clean_document_text(text, subject="beta")

    assert cleaned_a == cleaned_b


def test_loader_adds_only_cleaning_metadata_from_new_pipeline(monkeypatch):
    monkeypatch.delenv("RAG_SPLITTER_MODE", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "notes_2026.txt"
        source.write_text("Header\n\n\nUseful source content. " * 20, encoding="utf-8")

        docs = load_documents(tmpdir, subject="general", doc_type="course_material")

    assert docs
    metadata = docs[0].metadata
    assert metadata["subject"] == "general"
    assert metadata["source_file"] == "notes_2026.txt"
    assert metadata["year"] == "2026"
    assert "cleaning_chars_before" in metadata
    assert "cleaning_removed_blank_lines" in metadata
    assert not any(key.startswith("chunk_audit_") for key in metadata)
    assert "section_path" not in metadata
    assert "heading_count" not in metadata
