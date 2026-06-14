"""Whitespace normalization helpers for RAG source documents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WhitespaceStats:
    """Counts produced while normalizing whitespace."""

    removed_blank_lines: int = 0


def normalize_whitespace(text: str, *, max_consecutive_blank_lines: int = 1) -> tuple[str, WhitespaceStats]:
    """Normalize line endings and excessive blank lines without changing indentation.

    The function intentionally avoids collapsing spaces within non-empty lines so
    code blocks, tables, formulas, and aligned examples keep their shape.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]

    output: list[str] = []
    blank_run = 0
    removed_blank_lines = 0
    for line in lines:
        if line.strip():
            blank_run = 0
            output.append(line)
            continue

        blank_run += 1
        if blank_run <= max_consecutive_blank_lines:
            output.append("")
        else:
            removed_blank_lines += 1

    while output and not output[0].strip():
        output.pop(0)
        removed_blank_lines += 1
    while output and not output[-1].strip():
        output.pop()
        removed_blank_lines += 1

    return "\n".join(output), WhitespaceStats(removed_blank_lines=removed_blank_lines)
