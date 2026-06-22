"""PDF text extraction diagnostics for RAG source files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.rag.ids import make_source_relpath

PREVIEW_CHARS = 200
MAX_FIRST_NON_EMPTY_PAGES = 5
LOW_TEXT_FILE_SIZE = 1_000_000


@dataclass(frozen=True)
class PdfPagePreview:
    page_index: int
    chars: int
    preview: str


@dataclass(frozen=True)
class PdfTextInspection:
    subject: str
    source_file: str
    source_relpath: str
    file_size: int
    page_count: int
    total_extracted_chars: int
    empty_page_count: int
    min_page_chars: int
    max_page_chars: int
    avg_page_chars: float
    first_non_empty_pages: tuple[PdfPagePreview, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["avg_page_chars"] = round(self.avg_page_chars, 2)
        payload["first_non_empty_pages"] = [
            asdict(item) for item in self.first_non_empty_pages
        ]
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True)
class PdfInspectionSkipped:
    source_file: str
    source_relpath: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PdfInspectionReport:
    data_dir: str
    subject: str | None
    pdf_count: int
    suspicious_count: int
    inspections: tuple[PdfTextInspection, ...] = ()
    skipped: tuple[PdfInspectionSkipped, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "subject": self.subject,
            "pdf_count": self.pdf_count,
            "suspicious_count": self.suspicious_count,
            "inspections": [item.to_dict() for item in self.inspections],
            "skipped": [item.to_dict() for item in self.skipped],
        }


def _preview(text: str, *, max_chars: int = PREVIEW_CHARS) -> str:
    normalized = " ".join(text.split())
    return normalized[:max_chars]


def _warnings(
    *,
    file_size: int,
    page_count: int,
    total_extracted_chars: int,
    empty_page_count: int,
    avg_page_chars: float,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if page_count > 0 and total_extracted_chars == 0:
        warnings.append("no_text_extracted")
    if file_size >= LOW_TEXT_FILE_SIZE and total_extracted_chars < 2000:
        warnings.append("very_low_extracted_text")
    if page_count >= 10 and empty_page_count / page_count >= 0.8:
        warnings.append("many_empty_pages")
    if page_count >= 10 and avg_page_chars < 100:
        warnings.append("low_text_density")
    return tuple(warnings)


def inspect_pdf_page_texts(
    page_texts: list[str],
    *,
    subject: str,
    source_file: str,
    source_relpath: str,
    file_size: int,
) -> PdfTextInspection:
    """Inspect extracted page texts without storing full text."""

    lengths = [len(text) for text in page_texts]
    page_count = len(page_texts)
    total_extracted_chars = sum(lengths)
    empty_page_count = sum(1 for text in page_texts if not text.strip())
    avg_page_chars = mean(lengths) if lengths else 0.0
    previews: list[PdfPagePreview] = []
    for page_index, text in enumerate(page_texts):
        if not text.strip():
            continue
        previews.append(
            PdfPagePreview(
                page_index=page_index,
                chars=len(text),
                preview=_preview(text),
            )
        )
        if len(previews) >= MAX_FIRST_NON_EMPTY_PAGES:
            break

    return PdfTextInspection(
        subject=subject,
        source_file=source_file,
        source_relpath=source_relpath,
        file_size=file_size,
        page_count=page_count,
        total_extracted_chars=total_extracted_chars,
        empty_page_count=empty_page_count,
        min_page_chars=min(lengths) if lengths else 0,
        max_page_chars=max(lengths) if lengths else 0,
        avg_page_chars=avg_page_chars,
        first_non_empty_pages=tuple(previews),
        warnings=_warnings(
            file_size=file_size,
            page_count=page_count,
            total_extracted_chars=total_extracted_chars,
            empty_page_count=empty_page_count,
            avg_page_chars=avg_page_chars,
        ),
    )


def inspect_pdf_file(
    path: str | Path,
    *,
    subject: str,
    project_root: str | Path | None = None,
) -> PdfTextInspection:
    """Inspect one PDF file using PyMuPDF text extraction."""

    import fitz  # PyMuPDF

    pdf_path = Path(path)
    page_texts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_texts.append(page.get_text())

    return inspect_pdf_page_texts(
        page_texts,
        subject=subject,
        source_file=pdf_path.name,
        source_relpath=make_source_relpath(pdf_path, project_root=project_root),
        file_size=pdf_path.stat().st_size,
    )


def _short_error(exc: Exception) -> str:
    first_line = str(exc).splitlines()[0] if str(exc).splitlines() else str(exc)
    return f"{type(exc).__name__}: {first_line[:200]}"


def _subject_for_pdf(path: Path, *, data_dir: Path, subject: str | None) -> str:
    if subject:
        return subject
    try:
        relative = path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        return path.parent.name
    return relative.parts[0] if len(relative.parts) > 1 else ""


def _pdf_paths(
    data_dir: Path,
    *,
    subject: str | None,
    exclude_needs_ocr: bool = False,
) -> list[Path]:
    root = data_dir / subject if subject else data_dir
    if not root.is_dir():
        return []
    paths = []
    for path in root.rglob("*.pdf"):
        if not path.is_file():
            continue
        if exclude_needs_ocr:
            try:
                relative = path.resolve().relative_to(data_dir.resolve())
            except ValueError:
                relative = path
            if "_needs_ocr" in relative.parts:
                continue
        paths.append(path)
    return sorted(paths)


def inspect_pdf_tree(
    data_dir: str | Path,
    *,
    subject: str | None = None,
    project_root: str | Path | None = None,
    exclude_needs_ocr: bool = False,
) -> PdfInspectionReport:
    """Inspect all PDFs under a data directory, skipping unreadable files."""

    root = Path(data_dir)
    pdf_paths = _pdf_paths(
        root,
        subject=subject,
        exclude_needs_ocr=exclude_needs_ocr,
    )
    inspections: list[PdfTextInspection] = []
    skipped: list[PdfInspectionSkipped] = []
    for pdf_path in pdf_paths:
        source_relpath = make_source_relpath(pdf_path, project_root=project_root)
        try:
            inspections.append(
                inspect_pdf_file(
                    pdf_path,
                    subject=_subject_for_pdf(pdf_path, data_dir=root, subject=subject),
                    project_root=project_root,
                )
            )
        except Exception as exc:
            skipped.append(
                PdfInspectionSkipped(
                    source_file=pdf_path.name,
                    source_relpath=source_relpath,
                    error=_short_error(exc),
                )
            )

    return PdfInspectionReport(
        data_dir=str(root),
        subject=subject,
        pdf_count=len(pdf_paths),
        suspicious_count=sum(1 for item in inspections if item.warnings),
        inspections=tuple(inspections),
        skipped=tuple(skipped),
    )
