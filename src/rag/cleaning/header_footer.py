"""Generic header, footer, and page-number cleanup helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re


_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*[-–—]\s*\d{1,4}\s*[-–—]\s*$"),
    re.compile(r"^\s*Page\s+\d{1,4}(?:\s+of\s+\d{1,4})?\s*$", re.IGNORECASE),
    re.compile(r"^\s*第\s*\d{1,4}\s*页\s*$"),
    re.compile(r"^\s*第\s*\d{1,4}\s*页\s*/\s*共\s*\d{1,4}\s*页\s*$"),
    re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$"),
]
_PLAIN_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")


@dataclass(frozen=True)
class HeaderFooterStats:
    """Counts from page-number and repeated-line cleanup."""

    removed_page_number_lines: int = 0
    removed_repeated_lines: int = 0
    suspected_header_footer_lines: tuple[str, ...] = ()


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


def is_page_number_line(line: str, *, previous_line: str | None = None, next_line: str | None = None) -> bool:
    """Return True for high-confidence page-number-only lines."""

    if any(pattern.match(line) for pattern in _PAGE_NUMBER_PATTERNS):
        return True

    if not _PLAIN_NUMBER_PATTERN.match(line):
        return False

    # A bare number is too ambiguous in body text. Treat it as a page number
    # only at a clear boundary or when separated from content by blank lines.
    return _is_blank(previous_line) or _is_blank(next_line)


def remove_page_number_lines(lines: list[str]) -> tuple[list[str], int]:
    """Remove high-confidence page-number-only lines from a line list."""

    output: list[str] = []
    removed = 0
    for index, line in enumerate(lines):
        previous_line = lines[index - 1] if index > 0 else None
        next_line = lines[index + 1] if index + 1 < len(lines) else None
        if is_page_number_line(line, previous_line=previous_line, next_line=next_line):
            removed += 1
            continue
        output.append(line)
    return output, removed


def _normalized_repeated_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def remove_repeated_header_footer_lines(
    lines: list[str],
    *,
    min_repetitions: int = 3,
    min_chars: int = 3,
    max_chars: int = 120,
) -> tuple[list[str], int, tuple[str, ...]]:
    """Remove short repeated lines that look like recurring headers/footers."""

    normalized_lines = [_normalized_repeated_line(line) for line in lines]
    counts = Counter(
        line
        for line in normalized_lines
        if min_chars <= len(line) <= max_chars
    )
    repeated = {
        line
        for line, count in counts.items()
        if count >= min_repetitions
    }
    if not repeated:
        return lines, 0, ()

    output: list[str] = []
    removed = 0
    for original, normalized in zip(lines, normalized_lines):
        if normalized in repeated:
            removed += 1
            continue
        output.append(original)

    preview = tuple(sorted(repeated)[:5])
    return output, removed, preview


def clean_header_footer_lines(lines: list[str]) -> tuple[list[str], HeaderFooterStats]:
    """Apply conservative page-number and repeated-line cleanup."""

    without_pages, page_count = remove_page_number_lines(lines)
    without_repeated, repeated_count, suspected = remove_repeated_header_footer_lines(without_pages)
    return without_repeated, HeaderFooterStats(
        removed_page_number_lines=page_count,
        removed_repeated_lines=repeated_count,
        suspected_header_footer_lines=suspected,
    )
