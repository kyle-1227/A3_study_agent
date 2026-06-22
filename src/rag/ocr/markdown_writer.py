"""Markdown writer for OCR output."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.rag.ocr.models import OCRPageResult


def build_ocr_markdown(
    *,
    title: str,
    subject: str,
    source_file: str,
    source_relpath: str,
    ocr_engine: str,
    ocr_lang: str,
    ocr_dpi: int,
    pages: Sequence[OCRPageResult],
) -> str:
    """Build the full Markdown document. Full OCR text belongs here only."""

    lines = [
        f"<!-- ocr_subject: {subject} -->",
        f"<!-- ocr_source_file: {source_file} -->",
        f"<!-- ocr_source_relpath: {source_relpath} -->",
        f"<!-- ocr_engine: {ocr_engine} -->",
        f"<!-- ocr_lang: {ocr_lang} -->",
        f"<!-- ocr_dpi: {ocr_dpi} -->",
        f"<!-- ocr_page_count: {len(pages)} -->",
        "",
        f"# {title}",
        "",
    ]
    for page in pages:
        lines.append(page.to_markdown())
    return "\n".join(lines).rstrip() + "\n"


def write_ocr_markdown(
    output_path: str | Path,
    *,
    title: str,
    subject: str,
    source_file: str,
    source_relpath: str,
    ocr_engine: str,
    ocr_lang: str,
    ocr_dpi: int,
    pages: Sequence[OCRPageResult],
) -> Path:
    """Write OCR Markdown and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_ocr_markdown(
        title=title,
        subject=subject,
        source_file=source_file,
        source_relpath=source_relpath,
        ocr_engine=ocr_engine,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
        pages=pages,
    )
    path.write_text(markdown, encoding="utf-8")
    return path
