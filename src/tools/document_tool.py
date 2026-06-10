"""Markdown document artifact helpers."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

DEFAULT_REVIEW_DOC_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "review_docs"


def get_review_doc_artifact_dir() -> Path:
    """Return the directory used for generated Markdown review documents."""
    root = Path(os.getenv("REVIEW_DOC_ARTIFACT_DIR", str(DEFAULT_REVIEW_DOC_ARTIFACT_DIR)))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _safe_filename_stem(value: str, default: str = "review-doc") -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", value.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip(".-")[:80]
    return cleaned or default


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    cells = _parse_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _parse_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _clean_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def _add_markdown_table(document, rows: list[list[str]]) -> None:
    rows = [row for row in rows if row]
    if not rows:
        return
    cols = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=cols)
    table.style = "Table Grid"
    for row_idx, row in enumerate(rows):
        for col_idx in range(cols):
            value = row[col_idx] if col_idx < len(row) else ""
            table.cell(row_idx, col_idx).text = _clean_inline_markdown(value)


def _write_docx_artifact(markdown_text: str, title: str, file_path: Path) -> None:
    from docx import Document

    document = Document()
    lines = markdown_text.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if _is_table_row(stripped):
            table_rows: list[list[str]] = []
            while index < len(lines) and _is_table_row(lines[index]):
                if not _is_table_separator(lines[index]):
                    table_rows.append(_parse_table_row(lines[index]))
                index += 1
            _add_markdown_table(document, table_rows)
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            document.add_heading(_clean_inline_markdown(heading.group(2)), level=level)
            index += 1
            continue

        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        if bullet:
            document.add_paragraph(_clean_inline_markdown(bullet.group(1)), style="List Bullet")
            index += 1
            continue

        document.add_paragraph(_clean_inline_markdown(stripped))
        index += 1

    if not lines:
        document.add_heading(title, level=1)
    document.save(file_path)


def create_markdown_artifact(markdown_text: str, title: str) -> dict:
    """Save Markdown text as .md and .docx artifacts and return public metadata."""
    artifact_id = uuid.uuid4().hex
    filename_stem = _safe_filename_stem(title)
    filename = f"{filename_stem}.md"
    docx_filename = f"{filename_stem}.docx"
    artifact_dir = get_review_doc_artifact_dir() / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    file_path = artifact_dir / filename
    docx_path = artifact_dir / docx_filename
    file_path.write_text(markdown_text.rstrip() + "\n", encoding="utf-8")
    _write_docx_artifact(markdown_text, title, docx_path)

    return {
        "artifact_id": artifact_id,
        "filename": filename,
        "docx_filename": docx_filename,
        "markdown_url": f"/artifacts/review-docs/{artifact_id}/{filename}",
        "docx_url": f"/artifacts/review-docs/{artifact_id}/{docx_filename}",
        "title": title,
    }
