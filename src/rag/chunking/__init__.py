"""Standalone RAG chunking helpers."""

from src.rag.chunking.models import DocumentSection
from src.rag.chunking.splitter_factory import (
    chunk_policy_version_for_mode,
    get_splitter_mode,
    split_documents_by_mode,
)
from src.rag.chunking.structure_detector import (
    detect_document_sections,
    get_section_text,
)

__all__ = [
    "DocumentSection",
    "chunk_policy_version_for_mode",
    "detect_document_sections",
    "get_section_text",
    "get_splitter_mode",
    "split_documents_by_mode",
]
