from __future__ import annotations

from pathlib import Path
import tempfile

from langchain_core.documents import Document
import pytest

from src.rag.chunking import splitter_factory
from src.rag.chunking.splitter_factory import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    get_splitter_mode,
    split_documents_by_mode,
)
from src.rag.loader import load_documents


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def test_get_splitter_mode_defaults_to_recursive_when_env_unset(monkeypatch):
    monkeypatch.delenv("RAG_SPLITTER_MODE", raising=False)

    assert get_splitter_mode() == "recursive"


@pytest.mark.parametrize("mode", ["recursive", "structure"])
def test_get_splitter_mode_accepts_valid_modes(monkeypatch, mode):
    monkeypatch.setenv("RAG_SPLITTER_MODE", mode)

    assert get_splitter_mode() == mode


@pytest.mark.parametrize("mode", ["", "   ", "abc"])
def test_get_splitter_mode_rejects_empty_blank_and_invalid(monkeypatch, mode):
    monkeypatch.setenv("RAG_SPLITTER_MODE", mode)

    with pytest.raises(ValueError, match="Invalid RAG_SPLITTER_MODE"):
        get_splitter_mode()


def test_recursive_mode_uses_default_chunk_size_and_overlap(monkeypatch):
    captured: dict[str, int] = {}

    class FakeRecursiveSplitter:
        def __init__(self, *, chunk_size, chunk_overlap, length_function):
            captured["chunk_size"] = chunk_size
            captured["chunk_overlap"] = chunk_overlap
            captured["length_function_result"] = length_function("abcd")

        def create_documents(self, *, texts, metadatas):
            return [
                Document(page_content=texts[0], metadata=dict(metadatas[0])),
            ]

    monkeypatch.setattr(
        splitter_factory,
        "RecursiveCharacterTextSplitter",
        FakeRecursiveSplitter,
    )

    docs = split_documents_by_mode(
        [Document(page_content="content", metadata={"source_file": "x.txt"})],
        mode="recursive",
    )

    assert docs[0].page_content == "content"
    assert captured == {
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "length_function_result": 4,
    }


def test_structure_mode_adds_section_metadata(monkeypatch):
    monkeypatch.setenv("RAG_SPLITTER_MODE", "structure")
    source = Document(
        page_content="# Heading\nUseful section body.",
        metadata={"doc_id": "doc_1", "source_file": "x.txt"},
    )

    docs = split_documents_by_mode([source])

    assert docs
    assert docs[0].metadata["splitter_mode"] == "structure"
    assert docs[0].metadata["section_title"] == "Heading"


def test_custom_splitter_load_documents_does_not_read_env_mode(
    local_tmp_path, monkeypatch
):
    monkeypatch.setenv("RAG_SPLITTER_MODE", "invalid")
    source = local_tmp_path / "notes_2026.txt"
    source.write_text("Useful source content. " * 80, encoding="utf-8")

    class CustomSplitter:
        def create_documents(self, *, texts, metadatas):
            return [
                Document(
                    page_content="custom chunk",
                    metadata={**metadatas[0], "custom_splitter": "yes"},
                )
            ]

    docs = load_documents(
        local_tmp_path,
        subject="general",
        doc_type="course_material",
        splitter=CustomSplitter(),
    )

    assert len(docs) == 1
    assert docs[0].page_content == "custom chunk"
    assert docs[0].metadata["custom_splitter"] == "yes"
    assert "section_title" not in docs[0].metadata
