"""Standalone document section detection.

This module detects structural headings only. It is not integrated into
``load_documents`` and does not write Chroma metadata in this phase.
"""

from __future__ import annotations

import re

from src.rag.chunking.models import DocumentSection


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.{1,160})\s*$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,5})[.)、]?\s+(.{2,160})\s*$")
_LETTER_HEADING = re.compile(r"^([A-Z])[.)]\s+(.{2,160})\s*$")
_CHINESE_NUMBER_HEADING = re.compile(
    r"^([一二三四五六七八九十百千]+)[、.．]\s*(.{2,160})\s*$"
)
_CJK_PAREN_HEADING = re.compile(
    r"^[（(]([一二三四五六七八九十百千]+)[）)]\s*(.{2,160})\s*$"
)
_CJK_CHAPTER_HEADING = re.compile(
    r"^第\s*([一二三四五六七八九十百千\d]+)\s*[章节篇]\s*(.{0,160})\s*$"
)
_EN_CHAPTER_HEADING = re.compile(
    r"^(chapter|section)\s+([0-9A-Za-z.]+)\s*[:：.-]?\s*(.{0,160})\s*$", re.IGNORECASE
)
_CJK_COURSE_HEADING = re.compile(
    r"^(实验|任务|模块|项目)\s*([一二三四五六七八九十百千\d]+)\s*[:：、.\-]?\s*(.{2,160})\s*$"
)
_EN_COURSE_HEADING = re.compile(
    r"^(module|lab|task|project)\s+([0-9A-Za-z.]+)\s*[:：.-]\s*(.{2,160})\s*$",
    re.IGNORECASE,
)

_CJK_COURSE_STYLES = {
    "实验": "cjk_lab",
    "任务": "cjk_task",
    "模块": "cjk_module",
    "项目": "cjk_project",
}
_EN_COURSE_LEVELS = {
    "module": 1,
    "lab": 2,
    "task": 2,
    "project": 2,
}


def _line_offsets(text: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        offsets.append(cursor)
        cursor += len(line)
    return offsets


def _looks_like_sentence(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) > 160:
        return True
    return stripped.endswith(("。", ".", "!", "?", "！", "？", "；", ";"))


def _detect_heading(line: str) -> tuple[str, int, str] | None:
    stripped = line.strip()
    if not stripped or _looks_like_sentence(stripped):
        return None

    match = _MARKDOWN_HEADING.match(stripped)
    if match:
        return match.group(2).strip(), len(match.group(1)), "markdown"

    match = _NUMBERED_HEADING.match(stripped)
    if match:
        number = match.group(1)
        title = match.group(2).strip()
        return title, number.count(".") + 1, "numbered"

    match = _LETTER_HEADING.match(stripped)
    if match:
        return match.group(2).strip(), 1, "lettered"

    match = _CHINESE_NUMBER_HEADING.match(stripped)
    if match:
        return match.group(2).strip(), 1, "cjk_numbered"

    match = _CJK_PAREN_HEADING.match(stripped)
    if match:
        return match.group(2).strip(), 2, "cjk_parenthesized"

    match = _CJK_CHAPTER_HEADING.match(stripped)
    if match:
        suffix = match.group(2).strip()
        title = suffix or stripped
        return title, 1, "cjk_chapter"

    match = _EN_CHAPTER_HEADING.match(stripped)
    if match:
        suffix = match.group(3).strip()
        title = suffix or stripped
        return title, 1, "chapter"

    match = _CJK_COURSE_HEADING.match(stripped)
    if match:
        label = match.group(1)
        return match.group(3).strip(), 2, _CJK_COURSE_STYLES[label]

    match = _EN_COURSE_HEADING.match(stripped)
    if match:
        label = match.group(1).casefold()
        return match.group(3).strip(), _EN_COURSE_LEVELS[label], label

    return None


def _build_section_path(
    stack: list[tuple[int, str]], level: int, title: str
) -> tuple[str, ...]:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))
    return tuple(item[1] for item in stack)


def get_section_text(text: str, section: DocumentSection) -> str:
    """Return a safely bounded text span for a detected section."""

    start_char = max(0, min(section.start_char, len(text)))
    end_char = max(0, min(section.end_char, len(text)))
    if end_char <= start_char:
        return ""
    return text[start_char:end_char]


def detect_document_sections(text: str) -> list[DocumentSection]:
    """Detect section headings and return their text spans."""

    lines = text.splitlines()
    if text == "":
        return []

    offsets = _line_offsets(text)
    headings: list[dict[str, object]] = []
    stack: list[tuple[int, str]] = []

    for index, line in enumerate(lines):
        detected = _detect_heading(line)
        if not detected:
            continue
        title, level, style = detected
        path = _build_section_path(stack, level, title)
        headings.append(
            {
                "title": title,
                "level": level,
                "start_line": index,
                "start_char": offsets[index] if index < len(offsets) else 0,
                "heading_style": style,
                "section_path": path,
            }
        )

    if not headings:
        return [
            DocumentSection(
                title="",
                level=0,
                start_line=0,
                end_line=max(0, len(lines) - 1),
                start_char=0,
                end_char=len(text),
                heading_style="fallback_full_document",
                section_path=(),
            )
        ]

    sections: list[DocumentSection] = []
    first_heading = headings[0]
    first_heading_start = int(first_heading["start_char"])
    if text[:first_heading_start].strip():
        sections.append(
            DocumentSection(
                title="Preamble",
                level=0,
                start_line=0,
                end_line=int(first_heading["start_line"]) - 1,
                start_char=0,
                end_char=first_heading_start,
                heading_style="preamble",
                section_path=(),
            )
        )

    for index, heading in enumerate(headings):
        next_heading = headings[index + 1] if index + 1 < len(headings) else None
        end_line = (
            int(next_heading["start_line"]) - 1 if next_heading else len(lines) - 1
        )
        end_char = int(next_heading["start_char"]) if next_heading else len(text)
        start_char = max(0, min(int(heading["start_char"]), len(text)))
        bounded_end_char = max(0, min(end_char, len(text)))
        if start_char < bounded_end_char and end_line >= int(heading["start_line"]):
            sections.append(
                DocumentSection(
                    title=str(heading["title"]),
                    level=int(heading["level"]),
                    start_line=int(heading["start_line"]),
                    end_line=end_line,
                    start_char=start_char,
                    end_char=bounded_end_char,
                    heading_style=str(heading["heading_style"]),
                    section_path=tuple(heading["section_path"]),
                )
            )

    return sections
