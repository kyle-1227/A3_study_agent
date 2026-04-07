"""Section-aware document splitter for Chinese exam papers (REQ-05).

Splits exam papers by section headers (e.g. "一、现代文阅读", "四、写作")
before applying character-level sub-chunking within each section.
Each resulting chunk carries a ``section_title`` metadata field.

Design: ADR-005 — independent module, drop-in compatible with
``RecursiveCharacterTextSplitter.create_documents()`` interface.
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Matches top-level Chinese section headers:
#   一、现代文阅读  /  二．填空题  /  三.选择题
# Does NOT match sub-section markers like （一）or (1).
SECTION_PATTERN = re.compile(
    r"^([一二三四五六七八九十]+[、.．]\s*.+)",
    re.MULTILINE,
)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200


class SectionAwareSplitter:
    """Split text by section headers, then sub-chunk within each section."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    # ------------------------------------------------------------------
    # Public API (compatible with RecursiveCharacterTextSplitter)
    # ------------------------------------------------------------------

    def create_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> list[Document]:
        """Split *texts* into section-aware chunks.

        Parameters match ``RecursiveCharacterTextSplitter.create_documents``
        so this class can be used as a drop-in replacement in ``load_documents``.
        """
        all_chunks: list[Document] = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            base_meta = metadatas[i] if metadatas else {}
            sections = self._split_into_sections(text)
            for title, body in sections:
                if not body.strip():
                    continue
                chunk_meta = {**base_meta, "section_title": title}
                chunks = self._splitter.create_documents(
                    texts=[body],
                    metadatas=[chunk_meta],
                )
                all_chunks.extend(chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Split *text* into ``(title, body)`` pairs by section headers.

        Returns a single ``("", full_text)`` entry when no headers are found.
        Any preamble text before the first header is prepended to the first
        section's body.
        """
        matches = list(SECTION_PATTERN.finditer(text))

        if not matches:
            return [("", text)]

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()

        for idx, match in enumerate(matches):
            title = match.group(1).strip()
            body_start = match.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()

            # Prepend preamble to first section
            if idx == 0 and preamble:
                body = preamble + "\n" + body

            sections.append((title, body))

        return sections
