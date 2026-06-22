"""Data models for OCR-to-Markdown conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_PREVIEW_CHARS = 200


def bounded_preview(text: str, *, max_chars: int = REPORT_PREVIEW_CHARS) -> str:
    """Return a whitespace-normalized preview for reports."""

    return " ".join(text.split())[:max_chars]


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float | None = None
    bbox: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"text": self.text}
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        return payload


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    image_path: Path
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class OCRPageResult:
    page_number: int
    text: str
    image_path: str
    line_count: int
    confidence: float | None = None
    warnings: tuple[str, ...] = ()

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_markdown(self) -> str:
        body = self.text.strip()
        return f"## Page {self.page_number}\n\n{body}\n"

    def to_report_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "char_count": self.char_count,
            "line_count": self.line_count,
            "warnings": list(self.warnings),
            "preview": bounded_preview(self.text),
        }
        if self.confidence is not None:
            payload["confidence"] = round(self.confidence, 4)
        return payload


@dataclass(frozen=True)
class OCRDocumentResult:
    subject: str
    source_file: str
    source_relpath: str
    markdown_path: str
    report_path: str
    page_count: int
    processed_page_count: int
    pages: tuple[OCRPageResult, ...]
    warnings: tuple[str, ...] = ()

    def to_report_dict(self) -> dict[str, Any]:
        confidence_values = [
            page.confidence for page in self.pages if page.confidence is not None
        ]
        avg_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        )
        payload: dict[str, Any] = {
            "subject": self.subject,
            "source_file": self.source_file,
            "source_relpath": self.source_relpath,
            "markdown_path": self.markdown_path,
            "report_path": self.report_path,
            "page_count": self.page_count,
            "processed_page_count": self.processed_page_count,
            "total_extracted_chars": sum(page.char_count for page in self.pages),
            "warnings": list(self.warnings),
            "pages": [page.to_report_dict() for page in self.pages],
        }
        if avg_confidence is not None:
            payload["avg_confidence"] = round(avg_confidence, 4)
        return payload


@dataclass(frozen=True)
class BatchOCRReport:
    processed: tuple[dict[str, Any], ...] = ()
    skipped_existing: tuple[dict[str, Any], ...] = ()
    failed: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": list(self.processed),
            "skipped_existing": list(self.skipped_existing),
            "failed": list(self.failed),
        }
