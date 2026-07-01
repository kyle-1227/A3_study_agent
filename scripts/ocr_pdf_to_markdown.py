"""Convert OCR-needed PDFs to Markdown outside the default RAG build path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.ocr.pipeline import (  # noqa: E402
    default_markdown_path,
    default_report_path,
    ocr_pdf_batch,
    ocr_pdf_to_markdown,
    write_batch_report,
)


def _parse_sample_pages(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    pages: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid page range: {token}")
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(token))
    if not pages or any(page < 1 for page in pages):
        raise argparse.ArgumentTypeError(
            "Sample pages must be positive 1-based page numbers."
        )
    return tuple(dict.fromkeys(pages))


def _is_inside(path: Path, parent: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_parent = parent.resolve(strict=False)
    return resolved_path == resolved_parent or resolved_parent in resolved_path.parents


def _warn_if_formal_data_output(path: Path) -> None:
    formal_dir = PROJECT_ROOT / "data" / "python"
    if _is_inside(path, formal_dir):
        print(
            "WARNING: output is inside data/python; this will enter the formal RAG "
            "data directory."
        )


def _single_status_payload(
    *,
    source: Path,
    output_path: Path,
    report_path: Path,
    reason: str,
) -> dict[str, str]:
    return {
        "source_file": source.name,
        "output_path": str(output_path),
        "report_path": str(report_path),
        "reason": reason,
    }


def _run_single(args: argparse.Namespace, pages: tuple[int, ...] | None) -> int:
    source = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else default_markdown_path(
            source, subject=args.subject, project_root=PROJECT_ROOT
        )
    )
    report_path = (
        Path(args.report)
        if args.report
        else default_report_path(source, project_root=PROJECT_ROOT)
    )
    _warn_if_formal_data_output(output_path)

    if not args.overwrite and output_path.exists() and report_path.exists():
        payload = _single_status_payload(
            source=source,
            output_path=output_path,
            report_path=report_path,
            reason="output_and_report_exist",
        )
        print("=== OCR single ===")
        print("Processed        : 0")
        print("Skipped existing : 1")
        print("Failed           : 0")
        print(f"Skipped          : {payload}")
        return 0

    if not args.overwrite and (output_path.exists() or report_path.exists()):
        payload = _single_status_payload(
            source=source,
            output_path=output_path,
            report_path=report_path,
            reason="partial_existing_output",
        )
        print("=== OCR single ===")
        print("Processed        : 0")
        print("Skipped existing : 0")
        print("Failed           : 1")
        print(f"Failure          : {payload}")
        return 1

    result = ocr_pdf_to_markdown(
        source,
        subject=args.subject,
        project_root=PROJECT_ROOT,
        output_path=output_path,
        report_path=report_path,
        engine_name=args.engine,
        lang=args.lang,
        dpi=args.dpi,
        pages=pages,
        keep_images=args.keep_images,
    )
    print("=== OCR single ===")
    print("Processed        : 1")
    print("Skipped existing : 0")
    print("Failed           : 0")
    print(f"Markdown         : {result.markdown_path}")
    print(f"Report           : {result.report_path}")
    print(f"Pages            : {result.processed_page_count}")
    return 0


def _run_batch(args: argparse.Namespace, pages: tuple[int, ...] | None) -> int:
    source = Path(args.input)
    if args.output:
        raise ValueError("--output is only valid for a single PDF input.")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data_ocr" / args.subject
    )
    report_dir = (
        Path(args.report_dir) if args.report_dir else PROJECT_ROOT / "reports" / "ocr"
    )
    batch_report = (
        Path(args.batch_report)
        if args.batch_report
        else report_dir / "batch_ocr_report.json"
    )
    _warn_if_formal_data_output(output_dir)

    report = ocr_pdf_batch(
        source,
        subject=args.subject,
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        report_dir=report_dir,
        temp_root=PROJECT_ROOT / "reports" / "ocr" / "tmp",
        engine_name=args.engine,
        lang=args.lang,
        dpi=args.dpi,
        pages=pages,
        keep_images=args.keep_images,
        overwrite=args.overwrite,
    )
    write_batch_report(report, batch_report)
    payload = report.to_dict()
    print("=== OCR batch ===")
    print(f"Processed        : {len(payload['processed'])}")
    print(f"Skipped existing : {len(payload['skipped_existing'])}")
    print(f"Failed           : {len(payload['failed'])}")
    print(f"Batch report     : {batch_report}")
    return 1 if payload["failed"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert OCR-needed PDFs to Markdown.")
    parser.add_argument("input", help="PDF file or directory containing PDFs.")
    parser.add_argument("--subject", required=True, help="Subject for OCR output.")
    parser.add_argument("--engine", default="paddleocr", help="OCR engine name.")
    parser.add_argument("--lang", default="ch", help="OCR engine language.")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI.")
    parser.add_argument(
        "--sample-pages",
        help="1-based pages to OCR, for example 1,2,5-7.",
    )
    parser.add_argument("--output", help="Markdown output path for one PDF.")
    parser.add_argument("--report", help="JSON report path for one PDF.")
    parser.add_argument("--output-dir", help="Markdown output directory for batch.")
    parser.add_argument("--report-dir", help="JSON report directory for batch.")
    parser.add_argument("--batch-report", help="Batch JSON report path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep rendered page images under reports/ocr/tmp.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pages = _parse_sample_pages(args.sample_pages)
    source = Path(args.input)
    if source.is_file():
        if args.output_dir or args.report_dir or args.batch_report:
            raise ValueError(
                "--output-dir, --report-dir and --batch-report require a directory input."
            )
        return _run_single(args, pages)
    return _run_batch(args, pages)


if __name__ == "__main__":
    raise SystemExit(main())
