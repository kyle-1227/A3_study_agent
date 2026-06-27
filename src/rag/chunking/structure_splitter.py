"""Structure-aware splitter built on document section detection."""

from __future__ import annotations

from dataclasses import dataclass
import re
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
_SECTION_KIND_NORMAL = "normal"
_SECTION_KIND_TITLE_ONLY = "title_only"
_SECTION_KIND_SHORT = "short_section"
_SECTION_KIND_SHORT_PREAMBLE = "short_preamble"
_SECTION_KIND_EMPTY = "empty"
_MERGE_REASON_NONE = "none"
_MAJOR_BOUNDARY_STYLES = {
    "chapter",
    "cjk_chapter",
}
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_MATH_MARKER = re.compile(r"(?:=|\+|\*|/|\^|∫|∑|√|≤|≥|≠|≈|\bO\()")


@dataclass(frozen=True)
class StructureMergeConfig:
    """Centralized section merge thresholds for the structure splitter."""

    min_section_chars: int = 120
    min_chunk_chars: int = 80
    max_merged_section_chars: int = 1600
    max_merged_section_ids_chars: int = 500
    max_merged_section_titles_chars: int = 300
    merge_title_only_sections: bool = True
    merge_short_preamble: bool = True
    preserve_code_blocks: bool = True
    preserve_table_like_blocks: bool = True
    preserve_list_like_blocks: bool = True
    preserve_math_like_blocks: bool = True


@dataclass(frozen=True)
class _SectionUnit:
    index: int
    section: DocumentSection
    text: str
    title: str
    path: str
    original_section_id: str
    body_text: str
    kind: str


@dataclass
class _MergedSection:
    parts: list[_SectionUnit]
    primary: _SectionUnit
    merge_reason: str = _MERGE_REASON_NONE


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


def _section_body_text(section_text: str, section: DocumentSection) -> str:
    if section.heading_style in {"fallback_full_document", "preamble"}:
        return section_text.strip()
    lines = section_text.splitlines()
    return "\n".join(lines[1:]).strip() if len(lines) > 1 else ""


def _line_count_matching(text: str, marker: str) -> int:
    return sum(1 for line in text.splitlines() if marker in line)


def _is_protected_content(text: str, config: StructureMergeConfig) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if config.preserve_code_blocks and (
        "```" in text or any(line.startswith(("    ", "\t")) for line in lines)
    ):
        return True
    if config.preserve_table_like_blocks and (
        _line_count_matching(text, "|") >= 2 or _line_count_matching(text, "\t") >= 2
    ):
        return True
    if config.preserve_list_like_blocks and len(_LIST_MARKER.findall(text)) >= 2:
        return True
    return bool(config.preserve_math_like_blocks and _MATH_MARKER.search(text))


def _classify_section(
    *,
    section: DocumentSection,
    section_text: str,
    body_text: str,
    config: StructureMergeConfig,
) -> str:
    if not section_text.strip():
        return _SECTION_KIND_EMPTY

    protected = _is_protected_content(section_text, config)
    body_chars = len(body_text.strip())
    if section.heading_style == "preamble":
        if body_chars < config.min_section_chars:
            return _SECTION_KIND_SHORT_PREAMBLE
        return _SECTION_KIND_NORMAL
    if section.heading_style == "fallback_full_document":
        return _SECTION_KIND_NORMAL
    if body_chars == 0:
        return _SECTION_KIND_SHORT if protected else _SECTION_KIND_TITLE_ONLY
    if body_chars < config.min_section_chars:
        return _SECTION_KIND_SHORT
    return _SECTION_KIND_NORMAL


def _section_id(
    *,
    doc_id: str,
    source_relpath: str,
    section_index: int,
    title: str,
    start_char: int,
    end_char: int,
) -> str:
    digest = sha1_text(
        f"{doc_id}|{source_relpath}|{section_index}|{title}|{start_char}|{end_char}"
    )
    return f"sec_{digest[:16]}"


def _merged_section_id(
    *,
    doc_id: str,
    source_relpath: str,
    title: str,
    start_char: int,
    end_char: int,
    child_section_ids: str,
) -> str:
    digest = sha1_text(
        f"{doc_id}|{source_relpath}|{start_char}|{end_char}|{child_section_ids}|{title}"
    )
    return f"sec_{digest[:16]}"


def _has_body(unit: _SectionUnit) -> bool:
    return bool(unit.body_text.strip())


def _text_size(units: list[_SectionUnit]) -> int:
    return sum(len(unit.text) for unit in units) + max(0, len(units) - 1) * 2


def _within_merge_size(units: list[_SectionUnit], config: StructureMergeConfig) -> bool:
    return _text_size(units) <= config.max_merged_section_chars


def _major_label(unit: _SectionUnit) -> str:
    if unit.section.heading_style in {"fallback_full_document", "preamble"}:
        return ""
    if unit.section.section_path:
        return str(unit.section.section_path[0])
    return unit.title if unit.section.level <= 1 else ""


def _crosses_major_boundary(left: _SectionUnit, right: _SectionUnit) -> bool:
    if left.section.heading_style == "preamble":
        return False
    left_major = _major_label(left)
    right_major = _major_label(right)
    both_major_styles = (
        left.section.heading_style in _MAJOR_BOUNDARY_STYLES
        and right.section.heading_style in _MAJOR_BOUNDARY_STYLES
    )
    if not both_major_styles:
        return False
    if left.section.level <= 1 and right.section.level <= 1 and both_major_styles:
        return bool(left.title and right.title and left.title != right.title)
    return bool(
        left_major
        and right_major
        and left_major != right_major
        and (left.section.level <= 1 or right.section.level <= 1)
    )


def _can_merge_units(
    units: list[_SectionUnit],
    *,
    config: StructureMergeConfig,
    enforce_boundary: bool,
) -> bool:
    if not units or not _within_merge_size(units, config):
        return False
    if not enforce_boundary:
        return True
    return not any(
        _crosses_major_boundary(left, right)
        for left, right in zip(units, units[1:], strict=False)
    )


def _merge_reason(existing: str, new_reason: str) -> str:
    if existing == _MERGE_REASON_NONE:
        return new_reason
    reasons = existing.split("|")
    if new_reason in reasons:
        return existing
    return f"{existing}|{new_reason}"


def _make_single(unit: _SectionUnit) -> _MergedSection:
    return _MergedSection(parts=[unit], primary=unit)


def _make_merged(
    parts: list[_SectionUnit], *, primary: _SectionUnit, reason: str
) -> _MergedSection:
    return _MergedSection(parts=parts, primary=primary, merge_reason=reason)


def _first_body_unit(units: list[_SectionUnit]) -> _SectionUnit | None:
    return next((unit for unit in units if _has_body(unit)), None)


def _bounded_join(values: list[str], *, separator: str, max_chars: int) -> str:
    text = separator.join(values)
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3]}..."


def _merged_text(merged: _MergedSection) -> str:
    return "\n\n".join(part.text.strip("\n") for part in merged.parts if part.text)


def _append_to_previous(
    merged: _MergedSection, unit: _SectionUnit, *, reason: str
) -> None:
    merged.parts.append(unit)
    merged.merge_reason = _merge_reason(merged.merge_reason, reason)


def _forward_prefix_merge(
    units: list[_SectionUnit],
    start_index: int,
    *,
    config: StructureMergeConfig,
) -> tuple[_MergedSection, int] | None:
    first = units[start_index]
    if first.kind == _SECTION_KIND_SHORT_PREAMBLE:
        reason = "short_preamble_forward_merge"
        allow_prefix_title_only = True
    elif first.kind == _SECTION_KIND_TITLE_ONLY:
        reason = "title_only_forward_merge"
        allow_prefix_title_only = True
    else:
        return None

    prefix = [first]
    target_index = start_index + 1
    while (
        allow_prefix_title_only
        and target_index < len(units)
        and units[target_index].kind == _SECTION_KIND_TITLE_ONLY
    ):
        prefix.append(units[target_index])
        target_index += 1

    if target_index >= len(units):
        return None
    target = units[target_index]
    if not _has_body(target):
        return None
    candidate_parts = [*prefix, target]
    if not _can_merge_units(candidate_parts, config=config, enforce_boundary=True):
        return None
    return (
        _make_merged(candidate_parts, primary=target, reason=reason),
        target_index + 1,
    )


def _forward_short_section_merge(
    units: list[_SectionUnit],
    start_index: int,
    *,
    config: StructureMergeConfig,
) -> tuple[_MergedSection, int] | None:
    unit = units[start_index]
    candidate_parts = [unit]
    next_index = start_index + 1
    while next_index < len(units):
        expanded_parts = [*candidate_parts, units[next_index]]
        if not _can_merge_units(expanded_parts, config=config, enforce_boundary=True):
            break
        candidate_parts = expanded_parts
        next_index += 1

    if len(candidate_parts) <= 1:
        return None
    first_body_target = _first_body_unit(candidate_parts[1:])
    primary = unit
    if len(unit.body_text.strip()) < config.min_chunk_chars and first_body_target:
        primary = first_body_target
    return (
        _make_merged(
            candidate_parts,
            primary=primary,
            reason="short_section_forward_merge",
        ),
        next_index,
    )


def _try_backward_merge(
    output: list[_MergedSection],
    unit: _SectionUnit,
    *,
    config: StructureMergeConfig,
    reason: str,
) -> bool:
    if not output:
        return False
    previous = output[-1]
    if previous.parts[-1].index != unit.index - 1:
        return False
    candidate_parts = [*previous.parts, unit]
    if not _can_merge_units(candidate_parts, config=config, enforce_boundary=True):
        return False
    _append_to_previous(previous, unit, reason=reason)
    return True


def _merge_sections(
    units: list[_SectionUnit], config: StructureMergeConfig
) -> list[_MergedSection]:
    output: list[_MergedSection] = []
    index = 0
    while index < len(units):
        unit = units[index]

        if (
            unit.kind == _SECTION_KIND_SHORT_PREAMBLE and config.merge_short_preamble
        ) or (
            unit.kind == _SECTION_KIND_TITLE_ONLY and config.merge_title_only_sections
        ):
            forward = _forward_prefix_merge(units, index, config=config)
            if forward is not None:
                merged, next_index = forward
                output.append(merged)
                index = next_index
                continue

            if unit.kind == _SECTION_KIND_TITLE_ONLY and _try_backward_merge(
                output,
                unit,
                config=config,
                reason="title_only_backward_merge",
            ):
                index += 1
                continue

            output.append(_make_single(unit))
            index += 1
            continue

        if unit.kind == _SECTION_KIND_SHORT:
            forward = _forward_short_section_merge(units, index, config=config)
            if forward is not None:
                merged, next_index = forward
                output.append(merged)
                index = next_index
                continue

            if _try_backward_merge(
                output,
                unit,
                config=config,
                reason="short_section_backward_merge",
            ):
                index += 1
                continue

        output.append(_make_single(unit))
        index += 1
    return output


def _build_units(
    *,
    text: str,
    sections: list[DocumentSection],
    document_metadata: dict[str, Any],
    config: StructureMergeConfig,
) -> list[_SectionUnit]:
    doc_id = str(document_metadata.get("doc_id") or "")
    source_relpath = str(
        document_metadata.get("source_relpath")
        or document_metadata.get("source_file")
        or ""
    )
    units: list[_SectionUnit] = []
    for section_index, section in enumerate(sections):
        section_text = get_section_text(text, section)
        title = _section_title(section)
        body_text = _section_body_text(section_text, section)
        kind = _classify_section(
            section=section,
            section_text=section_text,
            body_text=body_text,
            config=config,
        )
        units.append(
            _SectionUnit(
                index=section_index,
                section=section,
                text=section_text,
                title=title,
                path=_section_path(section, title),
                original_section_id=_section_id(
                    doc_id=doc_id,
                    source_relpath=source_relpath,
                    section_index=section_index,
                    title=title,
                    start_char=section.start_char,
                    end_char=section.end_char,
                ),
                body_text=body_text,
                kind=kind,
            )
        )
    return units


def _merged_metadata(
    *,
    document_metadata: dict[str, Any],
    merged: _MergedSection,
    merged_index: int,
    config: StructureMergeConfig,
) -> dict[str, Any]:
    primary = merged.primary
    title = _section_title(primary.section)
    child_ids = _bounded_join(
        [part.original_section_id for part in merged.parts],
        separator="|",
        max_chars=config.max_merged_section_ids_chars,
    )
    child_titles = _bounded_join(
        [part.title for part in merged.parts],
        separator="|",
        max_chars=config.max_merged_section_titles_chars,
    )
    start_char = min(part.section.start_char for part in merged.parts)
    end_char = max(part.section.end_char for part in merged.parts)
    doc_id = str(document_metadata.get("doc_id") or "")
    source_relpath = str(
        document_metadata.get("source_relpath")
        or document_metadata.get("source_file")
        or ""
    )
    metadata = {
        **document_metadata,
        "splitter_mode": "structure",
        "section_id": _merged_section_id(
            doc_id=doc_id,
            source_relpath=source_relpath,
            title=title,
            start_char=start_char,
            end_char=end_char,
            child_section_ids=child_ids,
        ),
        "section_title": title,
        "section_level": primary.section.level,
        "section_path": _section_path(primary.section, title),
        "section_index": merged_index,
        "section_start_char": start_char,
        "section_end_char": end_char,
        "merged_section_count": len(merged.parts),
        "merged_section_ids": child_ids,
        "merged_section_titles": child_titles,
        "merge_reason": merged.merge_reason,
    }
    _assert_scalar_metadata(metadata)
    return metadata


class StructureAwareSplitter:
    """Split documents by detected sections, then recursively within each section."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_STRUCTURE_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_STRUCTURE_CHUNK_OVERLAP,
        merge_config: StructureMergeConfig | None = None,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self._merge_config = merge_config or StructureMergeConfig()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: Document) -> list[Document]:
        if not document.page_content.strip():
            return []

        sections = detect_document_sections(document.page_content)
        units = _build_units(
            text=document.page_content,
            sections=sections,
            document_metadata=document.metadata,
            config=self._merge_config,
        )
        merged_sections = _merge_sections(units, self._merge_config)

        output: list[Document] = []
        for merged_index, merged in enumerate(merged_sections):
            section_text = _merged_text(merged)
            if not section_text.strip():
                continue

            section_metadata = _merged_metadata(
                document_metadata=document.metadata,
                merged=merged,
                merged_index=merged_index,
                config=self._merge_config,
            )
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
    merge_config: StructureMergeConfig | None = None,
) -> list[Document]:
    """Split documents with structure-aware section metadata."""

    return StructureAwareSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        merge_config=merge_config,
    ).split_documents(documents)
