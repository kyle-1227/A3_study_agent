"""Document loader: PDF / Markdown / TXT → chunked LangChain Documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from src.rag.chunking.splitter_factory import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_policy_version_for_mode,
    get_splitter_mode,
    split_documents_by_mode,
)
from src.rag.cleaning import clean_document_text
from src.rag.ids import enrich_chunk_metadata, enrich_source_metadata

CHUNK_SIZE = DEFAULT_CHUNK_SIZE
CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP


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
        When *None* (default), documents are split through the configured
        splitter factory. The default mode is recursive; setting
        ``RAG_SPLITTER_MODE=structure`` enables structure-aware section
        splitting.
    """
    splitter_mode = get_splitter_mode() if splitter is None else None
    chunk_policy_version = (
        chunk_policy_version_for_mode(splitter_mode)
        if splitter_mode is not None
        else None
    )

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

        metadata = enrich_source_metadata(
            {
                "subject": subject,
                "source_file": filepath.name,
                "year": _guess_year(filepath.name) or "unknown",
                "doc_type": doc_type,
                **cleaning_report.to_metadata(),
            },
            source_path=filepath,
            subject=subject,
        )

        if splitter is not None:
            chunks = splitter.create_documents(
                texts=[cleaned_text],
                metadatas=[metadata],
            )
        else:
            chunks = split_documents_by_mode(
                [Document(page_content=cleaned_text, metadata=metadata)],
                mode=splitter_mode,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        documents.extend(
            enrich_chunk_metadata(
                chunk,
                doc_id=str(metadata["doc_id"]),
                chunk_index=chunk_index,
                **(
                    {"chunk_policy_version": chunk_policy_version}
                    if chunk_policy_version is not None
                    else {}
                ),
            )
            for chunk_index, chunk in enumerate(chunks)
        )

    return documents
