"""Document loader: PDF / Markdown / TXT → chunked LangChain Documents."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.rag.cleaning import clean_document_text

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def _guess_year(filename: str) -> Optional[str]:
    """Try to extract a 4-digit year from the filename."""
    m = re.search(r"(20\d{2})", filename)
    return m.group(1) if m else None


def _read_pdf(path: Path) -> str:
    import fitz  # PyMuPDF

    text_parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_READERS = {
    ".pdf": _read_pdf,
    ".md": _read_text,
    ".txt": _read_text,
}


def load_documents(
    data_dir: str | Path,
    subject: str,
    doc_type: str = "exam",
    splitter=None,
) -> list[Document]:
    """Load all supported files under *data_dir* and split into chunks.

    Each chunk carries metadata ``{subject, source_file, year, doc_type}``
    plus scalar ``cleaning_*`` fields from the source cleaning report.

    Parameters
    ----------
    splitter : optional
        A text splitter with a ``create_documents(texts, metadatas)`` method.
        When *None* (default), the built-in ``RecursiveCharacterTextSplitter``
        is used.  Pass a ``SectionAwareSplitter`` for exam papers.
    """
    active_splitter = splitter if splitter is not None else _splitter

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: list[Document] = []
    for filepath in sorted(data_dir.iterdir()):
        ext = filepath.suffix.lower()
        reader = _READERS.get(ext)
        if reader is None:
            continue

        raw_text = reader(filepath)
        if not raw_text.strip():
            continue

        cleaned_text, cleaning_report = clean_document_text(
            raw_text,
            source_file=filepath.name,
            doc_type=doc_type,
            subject=subject,
        )
        if not cleaned_text.strip():
            continue

        metadata = {
            "subject": subject,
            "source_file": filepath.name,
            "year": _guess_year(filepath.name) or "unknown",
            "doc_type": doc_type,
            **cleaning_report.to_metadata(),
        }

        chunks = active_splitter.create_documents(
            texts=[cleaned_text],
            metadatas=[metadata],
        )
        documents.extend(chunks)

    return documents
