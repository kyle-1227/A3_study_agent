"""Generate a standalone chunk audit report for data/ course materials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.rag.audit import audit_chunks  # noqa: E402
from src.rag.loader import load_documents  # noqa: E402


DATA_DIR = project_root / "data"
REPORT_PATH = project_root / "reports" / "chunk_audit_report.json"
COURSE_DOC_TYPE = "course_material"


def _load_all_documents() -> tuple[list, list[dict[str, str]]]:
    all_docs = []
    skipped: list[dict[str, str]] = []

    if not DATA_DIR.is_dir():
        return [], [{"subject": "", "reason": f"data directory not found: {DATA_DIR}"}]

    for directory in sorted(path for path in DATA_DIR.iterdir() if path.is_dir()):
        subject = directory.name
        if subject == "_needs_ocr":
            skipped.append(
                {
                    "subject": subject,
                    "reason": "quarantined OCR-needed directory",
                }
            )
            continue
        if not any(directory.iterdir()):
            skipped.append({"subject": subject, "reason": "empty directory"})
            continue
        try:
            docs = load_documents(directory, subject=subject, doc_type=COURSE_DOC_TYPE)
        except Exception as exc:
            skipped.append(
                {"subject": subject, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        all_docs.extend(docs)

    return all_docs, skipped


def main() -> None:
    docs, skipped = _load_all_documents()
    report = audit_chunks(docs)
    payload = {
        "data_dir": str(DATA_DIR),
        "report_path": str(REPORT_PATH),
        "skipped": skipped,
        "audit": report.to_dict(),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== chunk audit ===")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Chunks       : {report.total_chunks}")
    print(f"Sources      : {report.source_count}")
    print(f"Warnings     : {', '.join(report.warnings) if report.warnings else 'none'}")
    print(f"Suspicious sources: {len(report.suspicious_source_files)}")
    print(f"Report saved : {REPORT_PATH}")


if __name__ == "__main__":
    main()
