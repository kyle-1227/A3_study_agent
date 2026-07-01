"""OCR-to-Markdown pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.rag.ids import make_source_relpath
from src.rag.ocr.engines import OCREngine, create_ocr_engine
from src.rag.ocr.markdown_writer import write_ocr_markdown
from src.rag.ocr.models import BatchOCRReport, OCRDocumentResult, OCRLine, OCRPageResult
from src.rag.ocr.pdf_render import render_pdf_pages


def _short_error(exc: Exception) -> str:
    message = str(exc).splitlines()[0] if str(exc).splitlines() else str(exc)
    return f"{type(exc).__name__}: {message[:200]}"


def _confidence(lines: Sequence[OCRLine]) -> float | None:
    values = [line.confidence for line in lines if line.confidence is not None]
    return sum(values) / len(values) if values else None


def default_markdown_path(
    pdf_path: str | Path,
    *,
    subject: str,
    project_root: str | Path | None = None,
) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / "data_ocr" / subject / f"{Path(pdf_path).stem}.md"


def default_report_path(
    pdf_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / "reports" / "ocr" / f"{Path(pdf_path).stem}_ocr_report.json"


def default_temp_dir(
    pdf_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / "reports" / "ocr" / "tmp" / Path(pdf_path).stem


def _write_report(result: OCRDocumentResult, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.to_report_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cleanup_images(image_paths: Sequence[Path]) -> None:
    for image_path in image_paths:
        if image_path.exists():
            image_path.unlink()


def ocr_pdf_to_markdown(
    pdf_path: str | Path,
    *,
    subject: str,
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    temp_dir: str | Path | None = None,
    engine: OCREngine | None = None,
    engine_name: str = "paddleocr",
    lang: str = "ch",
    dpi: int = 200,
    pages: Sequence[int] | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    keep_images: bool = False,
) -> OCRDocumentResult:
    """Render a PDF, OCR selected pages, write Markdown and a bounded report."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    source = Path(pdf_path)
    markdown = (
        Path(output_path)
        if output_path is not None
        else default_markdown_path(
            source,
            subject=subject,
            project_root=root,
        )
    )
    report = (
        Path(report_path)
        if report_path is not None
        else default_report_path(
            source,
            project_root=root,
        )
    )
    image_dir = (
        Path(temp_dir)
        if temp_dir is not None
        else default_temp_dir(
            source,
            project_root=root,
        )
    )
    active_engine = (
        engine
        if engine is not None
        else create_ocr_engine(engine=engine_name, lang=lang)
    )

    rendered_pages = render_pdf_pages(
        source,
        image_dir,
        dpi=dpi,
        pages=pages,
        start_page=start_page,
        end_page=end_page,
    )
    page_results: list[OCRPageResult] = []
    for rendered_page in rendered_pages:
        warnings: list[str] = []
        lines: Sequence[OCRLine] = ()
        try:
            lines = active_engine.recognize(rendered_page.image_path)
        except Exception as exc:
            warnings.append(f"ocr_failed: {_short_error(exc)}")
        text = "\n".join(line.text for line in lines).strip()
        page_results.append(
            OCRPageResult(
                page_number=rendered_page.page_number,
                text=text,
                image_path=str(rendered_page.image_path),
                line_count=len(lines),
                confidence=_confidence(lines),
                warnings=tuple(warnings),
            )
        )

    source_relpath = make_source_relpath(source, project_root=root)
    write_ocr_markdown(
        markdown,
        title=source.stem,
        subject=subject,
        source_file=source.name,
        source_relpath=source_relpath,
        ocr_engine=engine_name,
        ocr_lang=lang,
        ocr_dpi=dpi,
        pages=page_results,
    )
    result = OCRDocumentResult(
        subject=subject,
        source_file=source.name,
        source_relpath=source_relpath,
        markdown_path=str(markdown),
        report_path=str(report),
        page_count=len(rendered_pages),
        processed_page_count=len(page_results),
        pages=tuple(page_results),
        warnings=tuple(warning for page in page_results for warning in page.warnings),
    )
    _write_report(result, report)
    if not keep_images:
        _cleanup_images([page.image_path for page in rendered_pages])
    return result


def _pdf_paths(input_path: Path) -> tuple[Path, ...]:
    if input_path.is_file():
        return (input_path,)
    if not input_path.is_dir():
        raise FileNotFoundError(f"OCR input path not found: {input_path}")
    return tuple(sorted(path for path in input_path.rglob("*.pdf") if path.is_file()))


def _relative_output_path(
    pdf_path: Path,
    *,
    input_root: Path,
    output_dir: Path,
) -> Path:
    if input_root.is_dir():
        relative_parent = pdf_path.parent.relative_to(input_root)
        return output_dir / relative_parent / f"{pdf_path.stem}.md"
    return output_dir / f"{pdf_path.stem}.md"


def ocr_pdf_batch(
    input_path: str | Path,
    *,
    subject: str,
    project_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    temp_root: str | Path | None = None,
    engine: OCREngine | None = None,
    engine_name: str = "paddleocr",
    lang: str = "ch",
    dpi: int = 200,
    pages: Sequence[int] | None = None,
    keep_images: bool = False,
    overwrite: bool = False,
) -> BatchOCRReport:
    """Run OCR for one PDF or every PDF under a directory."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    source_input = Path(input_path)
    pdfs = _pdf_paths(source_input)
    output_base = (
        Path(output_dir) if output_dir is not None else root / "data_ocr" / subject
    )
    report_base = (
        Path(report_dir) if report_dir is not None else root / "reports" / "ocr"
    )
    temp_base = (
        Path(temp_root) if temp_root is not None else root / "reports" / "ocr" / "tmp"
    )
    active_engine = (
        engine
        if engine is not None
        else create_ocr_engine(
            engine=engine_name,
            lang=lang,
        )
    )

    processed: list[dict] = []
    skipped_existing: list[dict] = []
    failed: list[dict] = []
    for pdf in pdfs:
        markdown = _relative_output_path(
            pdf,
            input_root=source_input,
            output_dir=output_base,
        )
        report = report_base / f"{pdf.stem}_ocr_report.json"
        if not overwrite and markdown.exists() and report.exists():
            skipped_existing.append(
                {
                    "source_file": pdf.name,
                    "output_path": str(markdown),
                    "report_path": str(report),
                    "reason": "output_and_report_exist",
                }
            )
            continue
        if not overwrite and (markdown.exists() or report.exists()):
            failed.append(
                {
                    "source_file": pdf.name,
                    "output_path": str(markdown),
                    "report_path": str(report),
                    "error": "partial_existing_output",
                }
            )
            continue
        try:
            if active_engine is None:
                active_engine = create_ocr_engine(engine=engine_name, lang=lang)
            result = ocr_pdf_to_markdown(
                pdf,
                subject=subject,
                project_root=root,
                output_path=markdown,
                report_path=report,
                temp_dir=temp_base / pdf.stem,
                engine=active_engine,
                dpi=dpi,
                pages=pages,
                keep_images=keep_images,
            )
            processed.append(
                {
                    "source_file": result.source_file,
                    "source_relpath": result.source_relpath,
                    "output_path": result.markdown_path,
                    "report_path": result.report_path,
                    "processed_page_count": result.processed_page_count,
                    "warnings": list(result.warnings),
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "source_file": pdf.name,
                    "source_path": str(pdf),
                    "error": _short_error(exc),
                }
            )

    if not keep_images and temp_base.exists():
        for child in temp_base.iterdir():
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()
    return BatchOCRReport(
        processed=tuple(processed),
        skipped_existing=tuple(skipped_existing),
        failed=tuple(failed),
    )


def write_batch_report(report: BatchOCRReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
