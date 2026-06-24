"""Feature-flagged splitter selection for RAG documents."""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.chunking.structure_splitter import split_documents_by_structure
from src.rag.ids import CHUNK_POLICY_VERSION, STRUCTURE_CHUNK_POLICY_VERSION

RECURSIVE_SPLITTER_MODE = "recursive"
STRUCTURE_SPLITTER_MODE = "structure"
VALID_SPLITTER_MODES = (RECURSIVE_SPLITTER_MODE, STRUCTURE_SPLITTER_MODE)
RAG_SPLITTER_MODE_ENV = "RAG_SPLITTER_MODE"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def get_splitter_mode() -> str:
    """Return the configured splitter mode, defaulting only when env is unset."""

    if RAG_SPLITTER_MODE_ENV not in os.environ:
        return RECURSIVE_SPLITTER_MODE

    raw_mode = os.environ[RAG_SPLITTER_MODE_ENV]
    mode = raw_mode.strip()
    if not mode or mode not in VALID_SPLITTER_MODES:
        expected = ", ".join(VALID_SPLITTER_MODES)
        raise ValueError(
            f"Invalid RAG_SPLITTER_MODE={raw_mode!r}. Expected one of: {expected}."
        )
    return mode


def chunk_policy_version_for_mode(mode: str) -> str:
    """Return the chunk policy version for a validated splitter mode."""

    if mode == RECURSIVE_SPLITTER_MODE:
        return CHUNK_POLICY_VERSION
    if mode == STRUCTURE_SPLITTER_MODE:
        return STRUCTURE_CHUNK_POLICY_VERSION
    expected = ", ".join(VALID_SPLITTER_MODES)
    raise ValueError(
        f"Invalid RAG_SPLITTER_MODE={mode!r}. Expected one of: {expected}."
    )


def split_documents_by_mode(
    documents: list[Document],
    *,
    mode: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents using the configured mode without indexing side effects."""

    active_mode = get_splitter_mode() if mode is None else mode
    chunk_size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
    chunk_overlap = DEFAULT_CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    if active_mode == RECURSIVE_SPLITTER_MODE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return splitter.create_documents(
            texts=[doc.page_content for doc in documents],
            metadatas=[dict(doc.metadata) for doc in documents],
        )

    if active_mode == STRUCTURE_SPLITTER_MODE:
        return split_documents_by_structure(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    expected = ", ".join(VALID_SPLITTER_MODES)
    raise ValueError(
        f"Invalid RAG_SPLITTER_MODE={active_mode!r}. Expected one of: {expected}."
    )
