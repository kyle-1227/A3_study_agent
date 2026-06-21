"""Standalone chunk quality diagnostics and conservative validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_core.documents import Document


@dataclass(frozen=True)
class ChunkQualityConfig:
    drop_below_chars: int = 60
    merge_below_chars: int = 180
    max_merged_chars: int = 1200
    preview_chars: int = 160


@dataclass(frozen=True)
class ChunkAction:
    action: str
    reason: str
    source_file: str
    chunk_index: int | None
    chunk_chars: int
    preview: str


@dataclass(frozen=True)
class ChunkQualityReport:
    input_count: int
    output_count: int
    dropped_count: int
    merged_count: int
    short_count: int
    empty_count: int
    noise_count: int
    actions: tuple[ChunkAction, ...] = ()


_PAGE_NUMBER = re.compile(r"^\d{1,4}$")
_DASHED_PAGE_NUMBER = re.compile(r"^[-—–]\s*\d{1,4}\s*[-—–]$")
_EN_PAGE_NUMBER = re.compile(r"^page\s+\d{1,4}$", re.IGNORECASE)
_CJK_PAGE_NUMBER = re.compile(r"^第\s*\d{1,4}\s*页$")
_MEANINGFUL_TEXT = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


def _preview(text: str, *, max_chars: int) -> str:
    return " ".join(text.split())[:max_chars]


def _source_file(doc: Document) -> str:
    value = doc.metadata.get("source_file")
    return value if isinstance(value, str) else ""


def _chunk_index(doc: Document) -> int | None:
    value = doc.metadata.get("chunk_index")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def is_noise_chunk(text: str) -> bool:
    """Return True for empty chunks and obvious page-marker noise."""

    stripped = text.strip()
    if not stripped:
        return True
    if _PAGE_NUMBER.fullmatch(stripped):
        return True
    if _DASHED_PAGE_NUMBER.fullmatch(stripped):
        return True
    if _EN_PAGE_NUMBER.fullmatch(stripped):
        return True
    if _CJK_PAGE_NUMBER.fullmatch(stripped):
        return True
    return len(stripped) <= 20 and _MEANINGFUL_TEXT.search(stripped) is None


def _diagnose_actions(
    documents: list[Document],
    *,
    config: ChunkQualityConfig,
) -> tuple[list[ChunkAction], int, int, int]:
    actions: list[ChunkAction] = []
    empty_count = 0
    noise_count = 0
    short_count = 0
    for position, doc in enumerate(documents):
        text = doc.page_content
        empty = not text.strip()
        noise = is_noise_chunk(text)
        short = len(text) < config.merge_below_chars
        chunk_index = _chunk_index(doc)
        if chunk_index is None:
            chunk_index = position

        if empty:
            empty_count += 1
            actions.append(
                ChunkAction(
                    action="flag",
                    reason="empty_chunk",
                    source_file=_source_file(doc),
                    chunk_index=chunk_index,
                    chunk_chars=len(text),
                    preview=_preview(text, max_chars=config.preview_chars),
                )
            )
        elif noise:
            noise_count += 1
            actions.append(
                ChunkAction(
                    action="flag",
                    reason="noise_chunk",
                    source_file=_source_file(doc),
                    chunk_index=chunk_index,
                    chunk_chars=len(text),
                    preview=_preview(text, max_chars=config.preview_chars),
                )
            )
        if short:
            short_count += 1
            actions.append(
                ChunkAction(
                    action="flag",
                    reason="short_chunk",
                    source_file=_source_file(doc),
                    chunk_index=chunk_index,
                    chunk_chars=len(text),
                    preview=_preview(text, max_chars=config.preview_chars),
                )
            )
    return actions, short_count, empty_count, noise_count


def _merged_document(buffer: list[Document]) -> Document:
    merged_text = "\n\n".join(doc.page_content for doc in buffer)
    metadata = dict(buffer[0].metadata)
    metadata.pop("chunk_id", None)
    metadata.pop("content_sha1", None)
    metadata["validation_requires_rehash"] = True
    metadata["validation_merged_chunk_count"] = len(buffer)
    return Document(page_content=merged_text, metadata=metadata)


def _flush_buffer(
    buffer: list[Document],
    output: list[Document],
    actions: list[ChunkAction],
    *,
    config: ChunkQualityConfig,
) -> int:
    if not buffer:
        return 0
    if len(buffer) == 1:
        output.append(buffer[0])
        buffer.clear()
        return 0

    merged = _merged_document(buffer)
    output.append(merged)
    actions.append(
        ChunkAction(
            action="merge",
            reason="short_adjacent_same_source",
            source_file=_source_file(buffer[0]),
            chunk_index=_chunk_index(buffer[0]),
            chunk_chars=len(merged.page_content),
            preview=_preview(merged.page_content, max_chars=config.preview_chars),
        )
    )
    buffer.clear()
    return 1


def _apply_changes(
    documents: list[Document],
    *,
    config: ChunkQualityConfig,
    actions: list[ChunkAction],
) -> tuple[list[Document], int, int]:
    output: list[Document] = []
    buffer: list[Document] = []
    dropped_count = 0
    merged_count = 0

    for doc in documents:
        if is_noise_chunk(doc.page_content):
            merged_count += _flush_buffer(buffer, output, actions, config=config)
            dropped_count += 1
            actions.append(
                ChunkAction(
                    action="drop",
                    reason="noise_or_empty_chunk",
                    source_file=_source_file(doc),
                    chunk_index=_chunk_index(doc),
                    chunk_chars=len(doc.page_content),
                    preview=_preview(doc.page_content, max_chars=config.preview_chars),
                )
            )
            continue

        is_short = len(doc.page_content) < config.merge_below_chars
        if not is_short:
            merged_count += _flush_buffer(buffer, output, actions, config=config)
            output.append(doc)
            continue

        same_source = not buffer or _source_file(buffer[-1]) == _source_file(doc)
        candidate_text = "\n\n".join(
            [*(item.page_content for item in buffer), doc.page_content]
        )
        can_merge = same_source and len(candidate_text) <= config.max_merged_chars
        if not can_merge:
            merged_count += _flush_buffer(buffer, output, actions, config=config)
        buffer.append(doc)

    merged_count += _flush_buffer(buffer, output, actions, config=config)
    return output, dropped_count, merged_count


def validate_chunks(
    documents: list[Document],
    *,
    config: ChunkQualityConfig | None = None,
    apply_changes: bool = False,
) -> tuple[list[Document], ChunkQualityReport]:
    """Diagnose chunk quality, optionally applying conservative changes."""

    active_config = config or ChunkQualityConfig()
    diagnostic_actions, short_count, empty_count, noise_count = _diagnose_actions(
        documents,
        config=active_config,
    )
    if not apply_changes:
        return list(documents), ChunkQualityReport(
            input_count=len(documents),
            output_count=len(documents),
            dropped_count=0,
            merged_count=0,
            short_count=short_count,
            empty_count=empty_count,
            noise_count=noise_count,
            actions=tuple(diagnostic_actions),
        )

    actions = list(diagnostic_actions)
    output, dropped_count, merged_count = _apply_changes(
        documents,
        config=active_config,
        actions=actions,
    )
    return output, ChunkQualityReport(
        input_count=len(documents),
        output_count=len(output),
        dropped_count=dropped_count,
        merged_count=merged_count,
        short_count=short_count,
        empty_count=empty_count,
        noise_count=noise_count,
        actions=tuple(actions),
    )
