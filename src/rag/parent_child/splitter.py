"""Deterministic, offset-preserving parent and child span construction."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from src.rag.chunking.structure_detector import detect_document_sections
from src.rag.parent_child.exceptions import (
    AtomicSpanTooLargeError,
    ParentChildInvariantError,
)
from src.rag.parent_child.ids import (
    make_child_id,
    make_parent_id,
    make_policy_fingerprint,
    make_section_id,
    sha1_content,
)
from src.rag.parent_child.loader import page_range_for_span
from src.rag.parent_child.models import (
    ChildDocument,
    ChildMetadata,
    CleanedSourceDocument,
    ParentChildBundle,
    ParentChildPolicy,
    ParentRecord,
)


_FENCE_OPEN_RE = re.compile(r"^ {0,3}([`~]{3,})")
_TABLE_DELIMITER_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+\S")
_DISPLAY_MATH_PATTERNS = (
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int


@dataclass(frozen=True)
class _AtomicSpan:
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class _StructureUnit:
    start: int
    end: int
    title: str
    level: int
    section_path: tuple[str, ...]


@dataclass(frozen=True)
class _ParentSpan:
    start: int
    end: int
    section_start: int
    section_title: str
    section_path: tuple[str, ...]


def _line_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    result: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        result.append((cursor, end, line))
        cursor = end
    if cursor < len(text):
        result.append((cursor, len(text), text[cursor:]))
    return tuple(result)


def _fenced_code_spans(text: str) -> list[_AtomicSpan]:
    lines = _line_spans(text)
    spans: list[_AtomicSpan] = []
    index = 0
    while index < len(lines):
        start, _, line = lines[index]
        match = _FENCE_OPEN_RE.match(line.rstrip("\r\n"))
        if match is None:
            index += 1
            continue
        fence = match.group(1)
        fence_char = fence[0]
        fence_length = len(fence)
        end = len(text)
        closing_index = len(lines) - 1
        for candidate_index in range(index + 1, len(lines)):
            _, candidate_end, candidate_line = lines[candidate_index]
            stripped = candidate_line.lstrip(" ")
            closing = re.match(r"^([`~]+)\s*(?:\r?\n)?$", stripped)
            if (
                closing is not None
                and closing.group(1)[0] == fence_char
                and len(closing.group(1)) >= fence_length
            ):
                end = candidate_end
                closing_index = candidate_index
                break
        spans.append(_AtomicSpan(start=start, end=end, kind="fenced_code"))
        index = closing_index + 1
    return spans


def _display_math_spans(text: str) -> list[_AtomicSpan]:
    return [
        _AtomicSpan(match.start(), match.end(), "display_math")
        for pattern in _DISPLAY_MATH_PATTERNS
        for match in pattern.finditer(text)
    ]


def _markdown_table_spans(text: str) -> list[_AtomicSpan]:
    lines = _line_spans(text)
    spans: list[_AtomicSpan] = []
    index = 0
    while index < len(lines):
        if "|" not in lines[index][2]:
            index += 1
            continue
        group_start = index
        while index < len(lines) and "|" in lines[index][2] and lines[index][2].strip():
            index += 1
        group = lines[group_start:index]
        if len(group) >= 2 and any(
            _TABLE_DELIMITER_RE.match(line.rstrip("\r\n")) is not None
            for _, _, line in group
        ):
            spans.append(
                _AtomicSpan(
                    start=group[0][0],
                    end=group[-1][1],
                    kind="markdown_table",
                )
            )
    return spans


def _list_block_spans(text: str) -> list[_AtomicSpan]:
    lines = _line_spans(text)
    spans: list[_AtomicSpan] = []
    index = 0
    while index < len(lines):
        if _LIST_LINE_RE.match(lines[index][2]) is None:
            index += 1
            continue
        group_start = index
        while index < len(lines) and _LIST_LINE_RE.match(lines[index][2]) is not None:
            index += 1
        group = lines[group_start:index]
        spans.append(
            _AtomicSpan(
                start=group[0][0],
                end=group[-1][1],
                kind="list_block",
            )
        )
    return spans


def _merge_atomic_spans(spans: list[_AtomicSpan]) -> tuple[_AtomicSpan, ...]:
    if not spans:
        return ()
    ordered = sorted(spans, key=lambda item: (item.start, item.end, item.kind))
    merged: list[_AtomicSpan] = [ordered[0]]
    for span in ordered[1:]:
        previous = merged[-1]
        if span.start < previous.end:
            merged[-1] = _AtomicSpan(
                start=previous.start,
                end=max(previous.end, span.end),
                kind="+".join(sorted(set(previous.kind.split("+")) | {span.kind})),
            )
        else:
            merged.append(span)
    return tuple(merged)


def _detect_atomic_spans(
    text: str, policy: ParentChildPolicy
) -> tuple[_AtomicSpan, ...]:
    spans: list[_AtomicSpan] = []
    if policy.atomic_fenced_code_blocks:
        spans.extend(_fenced_code_spans(text))
    if policy.atomic_markdown_tables:
        spans.extend(_markdown_table_spans(text))
    if policy.atomic_list_blocks:
        spans.extend(_list_block_spans(text))
    if policy.atomic_display_math:
        spans.extend(_display_math_spans(text))
    return _merge_atomic_spans(spans)


def _assert_atomic_limit(
    spans: tuple[_AtomicSpan, ...], *, hard_max: int, absolute_offset: int
) -> None:
    for span in spans:
        block_chars = span.end - span.start
        if block_chars > hard_max:
            raise AtomicSpanTooLargeError(
                block_kind=span.kind,
                block_chars=block_chars,
                hard_max_chars=hard_max,
                start_char=absolute_offset + span.start,
                end_char=absolute_offset + span.end,
            )


def _atomic_containing(
    spans: tuple[_AtomicSpan, ...], point: int
) -> _AtomicSpan | None:
    for span in spans:
        if span.start < point < span.end:
            return span
    return None


def _preferred_cut(
    text: str,
    *,
    start: int,
    end: int,
    target_size: int,
    hard_max: int,
    separators: tuple[str, ...],
    atomic_spans: tuple[_AtomicSpan, ...],
) -> int:
    target = min(start + target_size, end)
    if target == end:
        return end

    cut = target
    for separator in separators:
        position = text.rfind(separator, start + 1, target + 1)
        if position >= 0:
            candidate = position + len(separator)
            if start < candidate <= target:
                cut = candidate
                break

    atomic = _atomic_containing(atomic_spans, cut)
    if atomic is not None:
        if atomic.start > start:
            cut = atomic.start
        else:
            cut = atomic.end
    if cut <= start:
        raise ParentChildInvariantError("Recursive splitter failed to make progress")
    if cut - start > hard_max:
        raise ParentChildInvariantError(
            "Atomic-preserving cut exceeds the configured hard maximum"
        )
    return cut


def _next_start(
    *,
    current_start: int,
    cut: int,
    overlap: int,
    atomic_spans: tuple[_AtomicSpan, ...],
) -> int:
    if overlap == 0:
        return cut
    candidate = max(current_start, cut - overlap)
    atomic = _atomic_containing(atomic_spans, candidate)
    if atomic is not None:
        candidate = atomic.start if atomic.start > current_start else atomic.end
    if candidate <= current_start or candidate >= cut:
        return cut
    return candidate


def _recursive_spans(
    text: str,
    *,
    start: int,
    end: int,
    target_size: int,
    overlap: int,
    hard_max: int,
    separators: tuple[str, ...],
    atomic_spans: tuple[_AtomicSpan, ...],
) -> tuple[_Span, ...]:
    if end <= start:
        raise ParentChildInvariantError(
            "Recursive splitter input span must be non-empty"
        )
    result: list[_Span] = []
    cursor = start
    while cursor < end:
        cut = _preferred_cut(
            text,
            start=cursor,
            end=end,
            target_size=target_size,
            hard_max=hard_max,
            separators=separators,
            atomic_spans=atomic_spans,
        )
        if result and cut <= result[-1].end:
            cursor = result[-1].end
            continue
        result.append(_Span(start=cursor, end=cut))
        if cut == end:
            break
        cursor = _next_start(
            current_start=cursor,
            cut=cut,
            overlap=overlap,
            atomic_spans=atomic_spans,
        )
    return tuple(result)


def _is_inside_atomic(point: int, spans: tuple[_AtomicSpan, ...]) -> bool:
    return _atomic_containing(spans, point) is not None


def _structure_units(
    text: str, atomic_spans: tuple[_AtomicSpan, ...]
) -> tuple[_StructureUnit, ...]:
    detected = detect_document_sections(text)
    kept = [
        section
        for section in detected
        if not _is_inside_atomic(section.start_char, atomic_spans)
    ]
    if not kept:
        return (_StructureUnit(0, len(text), "", 0, ()),)

    ordered = sorted(kept, key=lambda section: section.start_char)
    starts: list[int] = []
    descriptors: list[tuple[str, int]] = []
    for section in ordered:
        start = max(0, min(section.start_char, len(text)))
        if starts and start <= starts[-1]:
            if start == starts[-1]:
                continue
            raise ParentChildInvariantError("Detected section starts are not ordered")
        starts.append(start)
        descriptors.append((section.title, section.level))
    starts[0] = 0

    stack: list[tuple[int, str]] = []
    units: list[_StructureUnit] = []
    for index, ((title, level), start) in enumerate(
        zip(descriptors, starts, strict=True)
    ):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        if end <= start:
            continue
        if level <= 0 or not title:
            section_path: tuple[str, ...] = ()
        else:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            section_path = tuple(item[1] for item in stack)
        units.append(
            _StructureUnit(
                start=start,
                end=end,
                title=title,
                level=level,
                section_path=section_path,
            )
        )
    if not units:
        raise ParentChildInvariantError("Structure detector produced no usable span")
    return tuple(units)


def _parent_spans(
    text: str,
    *,
    units: tuple[_StructureUnit, ...],
    atomic_spans: tuple[_AtomicSpan, ...],
    policy: ParentChildPolicy,
) -> tuple[_ParentSpan, ...]:
    result: list[_ParentSpan] = []
    pending: _ParentSpan | None = None

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            result.append(pending)
            pending = None

    for unit in units:
        starts_major_section = unit.level <= policy.major_section_max_level
        if starts_major_section:
            flush_pending()

        unit_chars = unit.end - unit.start
        if unit_chars > policy.parent_size:
            flush_pending()
            unit_atomic = tuple(
                span
                for span in atomic_spans
                if unit.start <= span.start and span.end <= unit.end
            )
            for span in _recursive_spans(
                text,
                start=unit.start,
                end=unit.end,
                target_size=policy.parent_size,
                overlap=policy.parent_overlap,
                hard_max=policy.parent_hard_max,
                separators=policy.parent_separators,
                atomic_spans=unit_atomic,
            ):
                result.append(
                    _ParentSpan(
                        start=span.start,
                        end=span.end,
                        section_start=unit.start,
                        section_title=unit.title,
                        section_path=unit.section_path,
                    )
                )
            continue

        if pending is None:
            pending = _ParentSpan(
                start=unit.start,
                end=unit.end,
                section_start=unit.start,
                section_title=unit.title,
                section_path=unit.section_path,
            )
            continue
        combined_chars = unit.end - pending.start
        pending_chars = pending.end - pending.start
        should_merge_short_unit = (
            pending_chars < policy.short_unit_chars
            or unit_chars < policy.short_unit_chars
        )
        if should_merge_short_unit and combined_chars <= policy.parent_size:
            pending = replace(pending, end=unit.end)
        else:
            flush_pending()
            pending = _ParentSpan(
                start=unit.start,
                end=unit.end,
                section_start=unit.start,
                section_title=unit.title,
                section_path=unit.section_path,
            )
    flush_pending()
    return tuple(result)


def _build_parent_records(
    source: CleanedSourceDocument,
    *,
    spans: tuple[_ParentSpan, ...],
    generation_id: str,
    policy_id: str,
) -> tuple[ParentRecord, ...]:
    parents: list[ParentRecord] = []
    for parent_index, span in enumerate(spans):
        content = source.content[span.start : span.end]
        content_sha1 = sha1_content(content)
        page_start, page_end = page_range_for_span(
            source, start_char=span.start, end_char=span.end
        )
        parents.append(
            ParentRecord(
                schema_version="parent_record_v1",
                parent_id=make_parent_id(
                    doc_id=source.doc_id,
                    policy_id=policy_id,
                    parent_index=parent_index,
                    exact_parent_content_sha1=content_sha1,
                ),
                doc_id=source.doc_id,
                subject=source.subject,
                generation_id=generation_id,
                policy_id=policy_id,
                parent_index=parent_index,
                source_file=source.source_file,
                source_relpath=source.source_relpath,
                source_file_sha1=source.source_file_sha1,
                doc_type=source.doc_type,
                extraction_method=source.extraction_method,
                cleaning_policy_id=source.cleaning_policy_id,
                section_id=make_section_id(
                    doc_id=source.doc_id,
                    section_start_char=span.section_start,
                    section_title=span.section_title,
                    section_path=span.section_path,
                ),
                section_title=span.section_title,
                section_path=span.section_path,
                pagination_kind=source.pagination_kind,
                page_start=page_start,
                page_end=page_end,
                start_char=span.start,
                end_char=span.end,
                parent_chars=len(content),
                content_sha1=content_sha1,
                content=content,
            )
        )
    return tuple(parents)


def _build_child_documents(
    source: CleanedSourceDocument,
    *,
    parents: tuple[ParentRecord, ...],
    policy: ParentChildPolicy,
) -> tuple[ChildDocument, ...]:
    children: list[ChildDocument] = []
    for parent in parents:
        atomic_spans = _detect_atomic_spans(parent.content, policy)
        _assert_atomic_limit(
            atomic_spans,
            hard_max=policy.child_hard_max,
            absolute_offset=parent.start_char,
        )
        child_spans = _recursive_spans(
            parent.content,
            start=0,
            end=parent.parent_chars,
            target_size=policy.child_size,
            overlap=policy.child_overlap,
            hard_max=policy.child_hard_max,
            separators=policy.child_separators,
            atomic_spans=atomic_spans,
        )
        for child_index, span in enumerate(child_spans):
            content = parent.content[span.start : span.end]
            content_sha1 = sha1_content(content)
            start_char = parent.start_char + span.start
            end_char = parent.start_char + span.end
            page_start, page_end = page_range_for_span(
                source, start_char=start_char, end_char=end_char
            )
            metadata = ChildMetadata(
                schema_version="child_metadata_v1",
                child_id=make_child_id(
                    parent_id=parent.parent_id,
                    child_index=child_index,
                    exact_child_content_sha1=content_sha1,
                ),
                parent_id=parent.parent_id,
                doc_id=parent.doc_id,
                subject=parent.subject,
                generation_id=parent.generation_id,
                policy_id=parent.policy_id,
                child_index=child_index,
                child_start_in_parent=span.start,
                child_end_in_parent=span.end,
                start_char=start_char,
                end_char=end_char,
                child_chars=len(content),
                content_sha1=content_sha1,
                source_file=parent.source_file,
                source_relpath=parent.source_relpath,
                source_file_sha1=parent.source_file_sha1,
                doc_type=parent.doc_type,
                section_id=parent.section_id,
                section_title=parent.section_title,
                section_path=parent.section_path,
                pagination_kind=parent.pagination_kind,
                page_start=page_start,
                page_end=page_end,
            )
            children.append(
                ChildDocument(
                    schema_version="child_document_v1",
                    content=content,
                    metadata=metadata,
                )
            )
    return tuple(children)


def build_parent_child_bundle(
    source: CleanedSourceDocument,
    policy: ParentChildPolicy,
    generation_id: str,
) -> ParentChildBundle:
    """Build and business-validate a deterministic parent-child bundle."""

    if source.loader_policy_id != policy.loader_policy_id:
        raise ParentChildInvariantError(
            "Cleaned source loader_policy_id does not match chunk policy"
        )
    if source.cleaning_policy_id != policy.cleaning_policy_id:
        raise ParentChildInvariantError(
            "Cleaned source cleaning_policy_id does not match chunk policy"
        )

    atomic_spans = _detect_atomic_spans(source.content, policy)
    _assert_atomic_limit(
        atomic_spans,
        hard_max=policy.parent_hard_max,
        absolute_offset=0,
    )
    units = _structure_units(source.content, atomic_spans)
    spans = _parent_spans(
        source.content,
        units=units,
        atomic_spans=atomic_spans,
        policy=policy,
    )
    policy_id = make_policy_fingerprint(policy)
    parents = _build_parent_records(
        source,
        spans=spans,
        generation_id=generation_id,
        policy_id=policy_id,
    )
    children = _build_child_documents(source, parents=parents, policy=policy)
    bundle = ParentChildBundle(
        schema_version="parent_child_bundle_v1",
        generation_id=generation_id,
        policy_id=policy_id,
        policy=policy,
        source=source,
        parents=parents,
        children=children,
    )

    from src.rag.parent_child.validation import validate_parent_child_bundle

    validate_parent_child_bundle(bundle)
    return bundle
