"""Duplicate paragraph cleanup helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DuplicateStats:
    """Counts from duplicate paragraph cleanup."""

    removed_duplicate_paragraphs: int = 0


def _split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    return re.split(r"\n\s*\n", text)


def _paragraph_key(paragraph: str) -> str:
    return re.sub(r"\s+", " ", paragraph.strip()).casefold()


def remove_duplicate_paragraphs(text: str, *, min_chars: int = 80) -> tuple[str, DuplicateStats]:
    """Remove exact duplicate long paragraphs after whitespace normalization."""

    paragraphs = _split_paragraphs(text)
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0

    for paragraph in paragraphs:
        key = _paragraph_key(paragraph)
        if len(key) >= min_chars and key in seen:
            removed += 1
            continue
        if key:
            seen.add(key)
        kept.append(paragraph.strip("\n"))

    return "\n\n".join(part for part in kept if part.strip()), DuplicateStats(
        removed_duplicate_paragraphs=removed
    )
