"""Structure-aware splitter built on document section detection."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.chunking.models import DocumentSection
from src.rag.chunking.structure_detector import (
    detect_document_sections,
    get_section_text,
)
from src.rag.ids import sha1_text

DEFAULT_STRUCTURE_CHUNK_SIZE = 1000
DEFAULT_STRUCTURE_CHUNK_OVERLAP = 200
_SCALAR_TYPES = (str, int, float, bool, type(None))


def _assert_scalar_metadata(metadata: dict[str, Any]) -> None:
    invalid = {
        key: type(value).__name__
        for key, value in metadata.items()
        if not isinstance(value, _SCALAR_TYPES)
    }
    if invalid:
        details = ", ".join(
            f"{key}={type_name}" for key, type_name in sorted(invalid.items())
        )
        raise TypeError(f"Structure splitter metadata must be scalar-only: {details}")


def _section_title(section: DocumentSection) -> str:
    if section.heading_style == "fallback_full_document":
        return "Document"
    return section.title or "Document"


def _section_path(section: DocumentSection, title: str) -> str:
    if section.heading_style == "fallback_full_document":
        return "Document"
    if section.heading_style == "preamble":
        return "Preamble"
    if section.section_path:
        return " > ".join(str(item) for item in section.section_path)
    return title


def _section_has_body(section_text: str, section: DocumentSection) -> bool:
    if section.heading_style in {"fallback_full_document", "preamble"}:
        return bool(section_text.strip())
    lines = section_text.splitlines()
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return bool(body) and any(char.isalpha() for char in body)


def _section_id(
    *,
    doc_id: str,
    section_index: int,
    title: str,
    start_char: int,
    end_char: int,
) -> str:
    digest = sha1_text(f"{doc_id}|{section_index}|{title}|{start_char}|{end_char}")
    return f"sec_{digest[:16]}"


class StructureAwareSplitter:
    """Split documents by detected sections, then recursively within each section."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_STRUCTURE_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_STRUCTURE_CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: Document) -> list[Document]:
        if not document.page_content.strip():
            return []

        sections = detect_document_sections(document.page_content)
        output: list[Document] = []
        for section_index, section in enumerate(sections):
            section_text = get_section_text(document.page_content, section)
            if not _section_has_body(section_text, section):
                continue

            title = _section_title(section)
            section_metadata = {
                **document.metadata,
                "splitter_mode": "structure",
                "section_id": _section_id(
                    doc_id=str(document.metadata.get("doc_id") or ""),
                    section_index=section_index,
                    title=title,
                    start_char=section.start_char,
                    end_char=section.end_char,
                ),
                "section_title": title,
                "section_level": section.level,
                "section_path": _section_path(section, title),
                "section_index": section_index,
                "section_start_char": section.start_char,
                "section_end_char": section.end_char,
            }
            _assert_scalar_metadata(section_metadata)

            section_chunks = self._splitter.create_documents(
                texts=[section_text],
                metadatas=[section_metadata],
            )
            for section_chunk_index, chunk in enumerate(section_chunks):
                metadata = {
                    **chunk.metadata,
                    "section_chunk_index": section_chunk_index,
                }
                _assert_scalar_metadata(metadata)
                output.append(
                    Document(page_content=chunk.page_content, metadata=metadata)
                )
        return output


def split_documents_by_structure(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_STRUCTURE_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_STRUCTURE_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents with structure-aware section metadata."""

    return StructureAwareSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).split_documents(documents)
