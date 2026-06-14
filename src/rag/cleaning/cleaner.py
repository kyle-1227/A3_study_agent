"""Document cleaning pipeline for RAG ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.rag.cleaning.duplicates import remove_duplicate_paragraphs
from src.rag.cleaning.header_footer import clean_header_footer_lines
from src.rag.cleaning.toc import remove_toc_lines
from src.rag.cleaning.whitespace import normalize_whitespace


@dataclass(frozen=True)
class CleaningReport:
    """Summary of conservative source-text cleanup."""

    source_file: str = ""
    doc_type: str = ""
    subject: str = ""
    chars_before: int = 0
    chars_after: int = 0
    lines_before: int = 0
    lines_after: int = 0
    removed_blank_lines: int = 0
    removed_repeated_lines: int = 0
    removed_page_number_lines: int = 0
    removed_toc_lines: int = 0
    removed_duplicate_paragraphs: int = 0
    suspected_header_footer_lines: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_metadata(self) -> dict[str, str | int | float | bool]:
        """Return Chroma-safe scalar metadata with a cleaning_ prefix."""

        return {
            "cleaning_chars_before": self.chars_before,
            "cleaning_chars_after": self.chars_after,
            "cleaning_lines_before": self.lines_before,
            "cleaning_lines_after": self.lines_after,
            "cleaning_removed_blank_lines": self.removed_blank_lines,
            "cleaning_removed_repeated_lines": self.removed_repeated_lines,
            "cleaning_removed_page_number_lines": self.removed_page_number_lines,
            "cleaning_removed_toc_lines": self.removed_toc_lines,
            "cleaning_removed_duplicate_paragraphs": self.removed_duplicate_paragraphs,
            "cleaning_suspected_header_footer_count": len(self.suspected_header_footer_lines),
            "cleaning_suspected_header_footer_preview": " | ".join(self.suspected_header_footer_lines[:5]),
            "cleaning_warning_count": len(self.warnings),
            "cleaning_warnings": " | ".join(self.warnings[:5]),
        }

    def to_trace_payload(self) -> dict[str, Any]:
        """Return a compact report payload for logs or audit scripts."""

        return {
            "source_file": self.source_file,
            "doc_type": self.doc_type,
            "subject": self.subject,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "lines_before": self.lines_before,
            "lines_after": self.lines_after,
            "removed_blank_lines": self.removed_blank_lines,
            "removed_repeated_lines": self.removed_repeated_lines,
            "removed_page_number_lines": self.removed_page_number_lines,
            "removed_toc_lines": self.removed_toc_lines,
            "removed_duplicate_paragraphs": self.removed_duplicate_paragraphs,
            "suspected_header_footer_count": len(self.suspected_header_footer_lines),
            "warnings": list(self.warnings),
        }


def _line_count(text: str) -> int:
    return len(text.splitlines()) if text else 0


def _build_warnings(*, chars_before: int, chars_after: int, lines_before: int, lines_after: int) -> tuple[str, ...]:
    warnings: list[str] = []
    if chars_before and chars_after / chars_before < 0.5:
        warnings.append("large_character_reduction")
    if lines_before and lines_after / lines_before < 0.5:
        warnings.append("large_line_reduction")
    if chars_before and not chars_after:
        warnings.append("cleaned_text_empty")
    return tuple(warnings)


def clean_document_text(
    text: str,
    *,
    source_file: str = "",
    doc_type: str = "",
    subject: str = "",
    config: dict[str, Any] | None = None,
) -> tuple[str, CleaningReport]:
    """Clean source text with conservative, non-domain-specific rules."""

    _ = config or {}
    chars_before = len(text)
    lines_before = _line_count(text)

    text, whitespace_stats = normalize_whitespace(text)
    lines = text.split("\n") if text else []
    lines, header_footer_stats = clean_header_footer_lines(lines)
    lines, toc_stats = remove_toc_lines(lines)
    text = "\n".join(lines)
    text, duplicate_stats = remove_duplicate_paragraphs(text)
    text, final_whitespace_stats = normalize_whitespace(text)

    chars_after = len(text)
    lines_after = _line_count(text)
    warnings = _build_warnings(
        chars_before=chars_before,
        chars_after=chars_after,
        lines_before=lines_before,
        lines_after=lines_after,
    )

    report = CleaningReport(
        source_file=source_file,
        doc_type=doc_type,
        subject=subject,
        chars_before=chars_before,
        chars_after=chars_after,
        lines_before=lines_before,
        lines_after=lines_after,
        removed_blank_lines=whitespace_stats.removed_blank_lines + final_whitespace_stats.removed_blank_lines,
        removed_repeated_lines=header_footer_stats.removed_repeated_lines,
        removed_page_number_lines=header_footer_stats.removed_page_number_lines,
        removed_toc_lines=toc_stats.removed_toc_lines,
        removed_duplicate_paragraphs=duplicate_stats.removed_duplicate_paragraphs,
        suspected_header_footer_lines=header_footer_stats.suspected_header_footer_lines,
        warnings=warnings,
    )
    return text, report
