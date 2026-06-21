"""Inspect PDF text extraction quality for data/ course materials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pdf_inspection import inspect_pdf_tree  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
REPORT_PATH = PROJECT_ROOT / "reports" / "pdf_text_inspection_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PDF text extraction quality.")
    parser.add_argument("--subject", help="Only inspect one subject directory.")
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR), help="Data directory to scan."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report = inspect_pdf_tree(data_dir, subject=args.subject, project_root=PROJECT_ROOT)
    payload = {
        "report_path": str(REPORT_PATH),
        **report.to_dict(),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== pdf text inspection ===")
    print(f"Data dir      : {data_dir}")
    print(f"PDF files     : {report.pdf_count}")
    print(f"Suspicious    : {report.suspicious_count}")
    print(f"Report saved  : {REPORT_PATH}")


if __name__ == "__main__":
    main()
