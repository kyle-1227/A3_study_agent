"""PDF page rendering for OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.rag.ocr.models import RenderedPage


def _selected_page_numbers(
    *,
    page_count: int,
    pages: Sequence[int] | None,
    start_page: int | None,
    end_page: int | None,
) -> list[int]:
    if pages is not None:
        selected = list(dict.fromkeys(pages))
    else:
        start = start_page if start_page is not None else 1
        end = end_page if end_page is not None else page_count
        selected = list(range(start, end + 1))

    if not selected:
        raise ValueError("No PDF pages selected for OCR.")
    invalid = [page for page in selected if page < 1 or page > page_count]
    if invalid:
        raise ValueError(
            f"PDF page selection out of range: {invalid}; page_count={page_count}"
        )
    return selected


def render_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 200,
    pages: Sequence[int] | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
) -> tuple[RenderedPage, ...]:
    """Render selected 1-based PDF pages to PNG files."""

    if dpi <= 0:
        raise ValueError("dpi must be positive.")

    import fitz  # PyMuPDF

    pdf = Path(pdf_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[RenderedPage] = []
    with fitz.open(pdf) as document:
        selected = _selected_page_numbers(
            page_count=document.page_count,
            pages=pages,
            start_page=start_page,
            end_page=end_page,
        )
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        for page_number in selected:
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = target_dir / f"page_{page_number:04d}.png"
            pixmap.save(image_path)
            rendered.append(
                RenderedPage(
                    page_number=page_number,
                    image_path=image_path,
                    width=pixmap.width,
                    height=pixmap.height,
                )
            )
    return tuple(rendered)
