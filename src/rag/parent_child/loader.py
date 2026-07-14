"""Page-aware source extraction and offset-preserving cleaning assembly."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.rag.ids import make_doc_id, sha1_file
from src.rag.parent_child.exceptions import (
    EmptySourceError,
    ParentChildInvariantError,
    SourceExtractionError,
    SourcePathError,
    UnsupportedSourceTypeError,
)
from src.rag.parent_child.ids import (
    make_loader_policy_fingerprint,
    sha1_content,
)
from src.rag.parent_child.models import (
    CleanedSourceDocument,
    PageAwareLoaderConfig,
    PageSpan,
    SourceEntry,
    SourcePage,
)
from src.rag.parent_child.tesseract_ocr import extract_pdf_pages_with_tesseract


def _resolve_source(
    entry: SourceEntry, config: PageAwareLoaderConfig
) -> tuple[Path, str]:
    try:
        data_root = entry.data_root.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SourcePathError("Configured data_root does not exist") from exc
    if not data_root.is_dir():
        raise SourcePathError("Configured data_root must be a directory")

    try:
        source_path = entry.source_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SourcePathError("Configured source_path does not exist") from exc
    if not source_path.is_file():
        raise SourcePathError("Configured source_path must be a regular file")
    try:
        source_relpath = source_path.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise SourcePathError(
            "Resolved source_path must remain inside the configured data_root"
        ) from exc

    extension = source_path.suffix.lower()
    if extension not in config.supported_extensions:
        raise UnsupportedSourceTypeError(
            f"Source extension is not explicitly supported: {extension or '<none>'}"
        )
    return source_path, source_relpath


def _extract_pdf_pages(path: Path) -> tuple[str, ...]:
    import fitz

    try:
        with fitz.open(path) as document:
            return tuple(page.get_text("text") for page in document)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SourceExtractionError(
            "PyMuPDF could not extract the configured PDF"
        ) from exc


def _extract_text_page(path: Path) -> tuple[str, ...]:
    try:
        return (path.read_bytes().decode("utf-8", errors="strict"),)
    except (OSError, UnicodeDecodeError) as exc:
        raise SourceExtractionError(
            "Text source must be readable strict UTF-8"
        ) from exc


def _prepare_page_lines(
    raw_text: str, config: PageAwareLoaderConfig
) -> tuple[str, ...]:
    text = raw_text
    if "\x00" in text:
        if config.nul_character_policy == "reject":
            raise SourceExtractionError(
                "Extracted source page contains a forbidden NUL character"
            )
        if config.nul_character_policy != "replace_with_space_v1":
            raise AssertionError("PageAwareLoaderConfig must validate NUL policy")
        # This is an explicit, length-preserving cleaning transform. It prevents
        # Chroma FTS5 trigram corruption without shifting cleaned offsets.
        text = text.replace("\x00", " ")
    if config.normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    if config.strip_trailing_whitespace:
        lines = [line.rstrip(" \t\f\v") for line in lines]
    if config.strip_outer_blank_lines:
        first = 0
        last = len(lines)
        while first < last and not lines[first].strip():
            first += 1
        while last > first and not lines[last - 1].strip():
            last -= 1
        lines = lines[first:last]
    return tuple(lines)


def _page_position_noise(
    prepared_pages: tuple[tuple[str, ...], ...],
    *,
    line_limit: int,
    from_end: bool,
    minimum_pages: int,
    minimum_ratio: float,
) -> frozenset[str]:
    """Find exact nonblank lines repeated at the same page edge."""

    counts: Counter[str] = Counter()
    for lines in prepared_pages:
        edge = lines[-line_limit:] if from_end else lines[:line_limit]
        counts.update({line for line in edge if line.strip()})
    page_count = len(prepared_pages)
    return frozenset(
        line
        for line, count in counts.items()
        if count >= minimum_pages and count / page_count >= minimum_ratio
    )


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = blank
    return collapsed


def _deduplicate_paragraphs(lines: list[str]) -> list[str]:
    """Remove exact repeated paragraphs within one page, retaining first order."""

    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
            continue
        if current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)

    seen: set[str] = set()
    kept: list[str] = []
    for paragraph in paragraphs:
        exact = "\n".join(paragraph)
        if exact in seen:
            continue
        seen.add(exact)
        if kept:
            kept.append("")
        kept.extend(paragraph)
    return kept


def _clean_prepared_page(
    lines: tuple[str, ...],
    *,
    header_noise: frozenset[str],
    footer_noise: frozenset[str],
    config: PageAwareLoaderConfig,
) -> str:
    page_lines = list(lines)
    last_header_index = min(config.header_top_lines, len(page_lines))
    first_footer_index = max(0, len(page_lines) - config.footer_bottom_lines)
    page_lines = [
        line
        for index, line in enumerate(page_lines)
        if not (
            (index < last_header_index and line in header_noise)
            or (index >= first_footer_index and line in footer_noise)
        )
    ]
    if config.collapse_blank_lines:
        page_lines = _collapse_blank_lines(page_lines)
    if config.paragraph_deduplication:
        page_lines = _deduplicate_paragraphs(page_lines)
    if config.strip_outer_blank_lines:
        while page_lines and not page_lines[0].strip():
            page_lines.pop(0)
        while page_lines and not page_lines[-1].strip():
            page_lines.pop()
    return "\n".join(page_lines)


def _make_pages(
    raw_pages: tuple[str, ...],
    *,
    extraction_method: str,
    config: PageAwareLoaderConfig,
) -> tuple[SourcePage, ...]:
    prepared_pages = tuple(
        _prepare_page_lines(raw_text, config) for raw_text in raw_pages
    )
    header_noise = _page_position_noise(
        prepared_pages,
        line_limit=config.header_top_lines,
        from_end=False,
        minimum_pages=config.repeated_line_min_pages,
        minimum_ratio=config.repeated_line_min_ratio,
    )
    footer_noise = _page_position_noise(
        prepared_pages,
        line_limit=config.footer_bottom_lines,
        from_end=True,
        minimum_pages=config.repeated_line_min_pages,
        minimum_ratio=config.repeated_line_min_ratio,
    )
    pages: list[SourcePage] = []
    for page_number, (raw_text, prepared_lines) in enumerate(
        zip(raw_pages, prepared_pages, strict=True), start=1
    ):
        cleaned_text = _clean_prepared_page(
            prepared_lines,
            header_noise=header_noise,
            footer_noise=footer_noise,
            config=config,
        )
        pages.append(
            SourcePage(
                schema_version="source_page_v1",
                page_number=page_number,
                extraction_method=extraction_method,
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                raw_chars=len(raw_text),
                cleaned_chars=len(cleaned_text),
                raw_content_sha1=sha1_content(raw_text),
                cleaned_content_sha1=sha1_content(cleaned_text),
            )
        )
    return tuple(pages)


def _assemble_pages(
    pages: tuple[SourcePage, ...], page_separator: str
) -> tuple[str, tuple[PageSpan, ...]]:
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    last_index = len(pages) - 1
    for index, page in enumerate(pages):
        start_char = cursor
        parts.append(page.cleaned_text)
        cursor += page.cleaned_chars
        content_end_char = cursor
        if index < last_index:
            parts.append(page_separator)
            cursor += len(page_separator)
        spans.append(
            PageSpan(
                schema_version="page_span_v1",
                page_number=page.page_number,
                start_char=start_char,
                content_end_char=content_end_char,
                end_char=cursor,
            )
        )
    return "".join(parts), tuple(spans)


def load_cleaned_source(
    entry: SourceEntry, loader_config: PageAwareLoaderConfig
) -> CleanedSourceDocument:
    """Extract and clean one source without losing page ordinals or offsets."""

    source_path, source_relpath = _resolve_source(entry, loader_config)
    extension = source_path.suffix.lower()
    if extension == ".pdf":
        pagination_kind = "physical"
        pdf_ocr = loader_config.pdf_ocr
        if pdf_ocr is not None and source_relpath in pdf_ocr.source_relpaths:
            extraction_method = pdf_ocr.extraction_method
            raw_pages = extract_pdf_pages_with_tesseract(source_path, pdf_ocr)
        else:
            extraction_method = loader_config.pdf_extraction_method
            raw_pages = _extract_pdf_pages(source_path)
    elif extension in {".md", ".txt"}:
        extraction_method = loader_config.text_extraction_method
        pagination_kind = "logical"
        raw_pages = _extract_text_page(source_path)
    else:
        raise UnsupportedSourceTypeError(
            f"No page extractor is implemented for configured extension: {extension}"
        )

    if not raw_pages:
        raise EmptySourceError("Source contains no physical or logical pages")
    pages = _make_pages(
        raw_pages,
        extraction_method=extraction_method,
        config=loader_config,
    )
    content, page_spans = _assemble_pages(pages, loader_config.page_separator)
    if not content.strip():
        raise EmptySourceError("Page-aware cleaning produced no indexable content")

    source_file_sha1 = sha1_file(source_path)
    document = CleanedSourceDocument(
        schema_version="cleaned_source_document_v1",
        doc_id=make_doc_id(
            subject=entry.subject,
            source_relpath=source_relpath,
            file_sha1=source_file_sha1,
        ),
        subject=entry.subject,
        source_file=source_path.name,
        source_relpath=source_relpath,
        source_file_sha1=source_file_sha1,
        doc_type=entry.doc_type,
        extraction_method=extraction_method,
        cleaning_policy_id=loader_config.cleaning_policy_id,
        loader_policy_id=make_loader_policy_fingerprint(loader_config),
        pagination_kind=pagination_kind,
        page_separator=loader_config.page_separator,
        source_pages=pages,
        page_spans=page_spans,
        content=content,
        content_sha1=sha1_content(content),
    )
    return document


def page_range_for_span(
    source: CleanedSourceDocument, *, start_char: int, end_char: int
) -> tuple[int, int]:
    """Map a non-empty cleaned-document span to inclusive page ordinals."""

    if start_char < 0 or end_char <= start_char or end_char > len(source.content):
        raise ParentChildInvariantError(
            "Page lookup span must be non-empty and inside cleaned source content"
        )
    pages = tuple(
        span.page_number
        for span in source.page_spans
        if span.start_char < end_char and start_char < span.end_char
    )
    if not pages:
        raise ParentChildInvariantError(
            "Cleaned-document span does not overlap an owned page segment"
        )
    return pages[0], pages[-1]
