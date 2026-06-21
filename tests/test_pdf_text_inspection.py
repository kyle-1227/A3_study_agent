from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from src.rag import pdf_inspection
from src.rag.pdf_inspection import inspect_pdf_page_texts, inspect_pdf_tree


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def test_inspect_pdf_page_texts_returns_basic_statistics():
    inspection = inspect_pdf_page_texts(
        ["First page text", "", "Third page text"],
        subject="python",
        source_file="sample.pdf",
        source_relpath="data/python/sample.pdf",
        file_size=1000,
    )

    assert inspection.page_count == 3
    assert inspection.total_extracted_chars == len("First page text") + len(
        "Third page text"
    )
    assert inspection.empty_page_count == 1
    assert inspection.min_page_chars == 0
    assert inspection.max_page_chars == len("Third page text")
    assert inspection.first_non_empty_pages[0].page_index == 0
    assert inspection.warnings == ()


def test_inspect_pdf_page_texts_warns_when_no_text_extracted():
    inspection = inspect_pdf_page_texts(
        ["", ""],
        subject="python",
        source_file="empty.pdf",
        source_relpath="data/python/empty.pdf",
        file_size=1000,
    )

    assert "no_text_extracted" in inspection.warnings


def test_inspect_pdf_page_texts_warns_for_many_empty_pages_and_low_density():
    inspection = inspect_pdf_page_texts(
        ["tiny"] + [""] * 9,
        subject="python",
        source_file="sparse.pdf",
        source_relpath="data/python/sparse.pdf",
        file_size=1_500_000,
    )

    assert "very_low_extracted_text" in inspection.warnings
    assert "many_empty_pages" in inspection.warnings
    assert "low_text_density" in inspection.warnings


def test_inspect_pdf_page_texts_truncates_preview_and_omits_full_text():
    long_text = "A" * 260 + "FULL_TEXT_SHOULD_NOT_APPEAR"
    inspection = inspect_pdf_page_texts(
        [long_text],
        subject="python",
        source_file="long.pdf",
        source_relpath="data/python/long.pdf",
        file_size=1000,
    )

    payload = json.dumps(inspection.to_dict(), ensure_ascii=False)

    assert len(inspection.first_non_empty_pages[0].preview) == 200
    assert "FULL_TEXT_SHOULD_NOT_APPEAR" not in payload


def test_inspect_pdf_tree_skips_bad_pdf_without_traceback(local_tmp_path, monkeypatch):
    data_dir = local_tmp_path / "data"
    subject_dir = data_dir / "python"
    subject_dir.mkdir(parents=True)
    good_pdf = subject_dir / "good.pdf"
    bad_pdf = subject_dir / "bad.pdf"
    good_pdf.write_bytes(b"%PDF good")
    bad_pdf.write_bytes(b"%PDF bad")

    def fake_inspect_pdf_file(path, *, subject, project_root):
        if Path(path).name == "bad.pdf":
            raise RuntimeError("broken pdf\ntraceback line that should not be stored")
        return inspect_pdf_page_texts(
            ["useful text"],
            subject=subject,
            source_file=Path(path).name,
            source_relpath=f"data/python/{Path(path).name}",
            file_size=100,
        )

    monkeypatch.setattr(pdf_inspection, "inspect_pdf_file", fake_inspect_pdf_file)

    report = inspect_pdf_tree(data_dir, project_root=local_tmp_path)
    payload = report.to_dict()

    assert report.pdf_count == 2
    assert len(report.inspections) == 1
    assert len(report.skipped) == 1
    assert payload["skipped"][0]["source_file"] == "bad.pdf"
    assert payload["skipped"][0]["source_relpath"].endswith("data/python/bad.pdf")
    assert "RuntimeError: broken pdf" == payload["skipped"][0]["error"]
    assert "traceback line" not in json.dumps(payload, ensure_ascii=False)
