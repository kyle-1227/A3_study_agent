from __future__ import annotations

import json

from src.rag.ocr.markdown_writer import build_ocr_markdown
from src.rag.ocr.models import OCRDocumentResult, OCRPageResult


def test_ocr_markdown_contains_metadata_and_page_text():
    pages = (
        OCRPageResult(
            page_number=1,
            text="FULL OCR TEXT PAGE ONE",
            image_path="tmp/page_0001.png",
            line_count=1,
        ),
        OCRPageResult(
            page_number=2,
            text="FULL OCR TEXT PAGE TWO",
            image_path="tmp/page_0002.png",
            line_count=1,
        ),
    )

    markdown = build_ocr_markdown(
        title="sample",
        subject="python",
        source_file="sample.pdf",
        source_relpath="data/_needs_ocr/python/sample.pdf",
        ocr_engine="paddleocr",
        ocr_lang="ch",
        ocr_dpi=200,
        pages=pages,
    )

    assert "<!-- ocr_subject: python -->" in markdown
    assert "<!-- ocr_source_file: sample.pdf -->" in markdown
    assert "<!-- ocr_source_relpath: data/_needs_ocr/python/sample.pdf -->" in markdown
    assert "<!-- ocr_engine: paddleocr -->" in markdown
    assert "<!-- ocr_lang: ch -->" in markdown
    assert "<!-- ocr_dpi: 200 -->" in markdown
    assert "## Page 1" in markdown
    assert "## Page 2" in markdown
    assert "FULL OCR TEXT PAGE ONE" in markdown
    assert "FULL OCR TEXT PAGE TWO" in markdown


def test_ocr_report_uses_bounded_preview_without_full_text():
    secret_tail = "FULL_TEXT_SHOULD_ONLY_BE_IN_MARKDOWN"
    long_text = "A" * 260 + secret_tail
    page = OCRPageResult(
        page_number=1,
        text=long_text,
        image_path="tmp/page_0001.png",
        line_count=1,
        confidence=0.91,
    )
    result = OCRDocumentResult(
        subject="python",
        source_file="sample.pdf",
        source_relpath="data/_needs_ocr/python/sample.pdf",
        markdown_path="data_ocr/python/sample.md",
        report_path="reports/ocr/sample_ocr_report.json",
        page_count=1,
        processed_page_count=1,
        pages=(page,),
    )

    payload = result.to_report_dict()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["pages"][0]["preview"] == "A" * 200
    assert secret_tail not in serialized
    assert "text" not in payload["pages"][0]
    assert payload["avg_confidence"] == 0.91
