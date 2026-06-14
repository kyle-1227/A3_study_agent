"""Table-of-contents cleanup helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re


_TOC_HEADING_PATTERN = re.compile(r"^\s*(目录|contents|table\s+of\s+contents)\s*$", re.IGNORECASE)
_LEADER_TOC_LINE_PATTERN = re.compile(r"^\s*.{2,120}?[\s.·・…]{2,}\d{1,4}\s*$")
_TOC_NUMBERED_LINE_PATTERN = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[A-Za-z][.)])\s+.{2,120}\s+\d{1,4}\s*$")


@dataclass(frozen=True)
class TocStats:
    """Counts from table-of-contents cleanup."""

    removed_toc_lines: int = 0


def _looks_like_toc_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_LEADER_TOC_LINE_PATTERN.match(stripped) or _TOC_NUMBERED_LINE_PATTERN.match(stripped))


def remove_toc_lines(lines: list[str], *, max_block_lines: int = 120) -> tuple[list[str], TocStats]:
    """Remove generic table-of-contents blocks and leader-dot TOC rows."""

    output: list[str] = []
    removed = 0
    in_toc_block = False
    block_line_count = 0
    non_toc_run = 0

    for line in lines:
        if _TOC_HEADING_PATTERN.match(line):
            in_toc_block = True
            block_line_count = 0
            non_toc_run = 0
            removed += 1
            continue

        is_toc_line = _looks_like_toc_line(line)
        if is_toc_line:
            removed += 1
            if in_toc_block:
                block_line_count += 1
            continue

        if in_toc_block:
            block_line_count += 1
            if not line.strip() and block_line_count <= max_block_lines:
                removed += 1
                continue
            non_toc_run += 1
            in_toc_block = False

        output.append(line)

    return output, TocStats(removed_toc_lines=removed)
