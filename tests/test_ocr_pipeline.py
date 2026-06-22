from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest

from src.rag.ocr.engines import StaticOCREngine
from src.rag.ocr.models import OCRLine, RenderedPage
from src.rag.ocr import pipeline
from src.rag.ocr.pipeline import ocr_pdf_batch, ocr_pdf_to_markdown


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def _fake_render(*_args, **kwargs):
    image_dir = Path(_args[1])
    image_dir.mkdir(parents=True, exist_ok=True)
    pages = kwargs.get("pages") or (1, 2)
    rendered = []
    for page_number in pages:
        image_path = image_dir / f"page_{page_number:04d}.png"
        image_path.write_bytes(b"png")
        rendered.append(
            RenderedPage(
                page_number=page_number,
                image_path=image_path,
                width=10,
                height=20,
            )
        )
    return tuple(rendered)


def test_ocr_pdf_to_markdown_writes_default_outputs_and_page_warnings(
    local_tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "render_pdf_pages", _fake_render)
    pdf_path = local_tmp_path / "data" / "_needs_ocr" / "python" / "sample.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF fake")
    engine = StaticOCREngine(
        pages=(
            (OCRLine(text="FULL OCR PAGE ONE", confidence=0.8),),
            RuntimeError("page boom\ntraceback should not appear"),
        )
    )

    result = ocr_pdf_to_markdown(
        pdf_path,
        subject="python",
        project_root=local_tmp_path,
        engine=engine,
    )

    markdown_path = local_tmp_path / "data_ocr" / "python" / "sample.md"
    report_path = local_tmp_path / "reports" / "ocr" / "sample_ocr_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert Path(result.markdown_path) == markdown_path
    assert Path(result.report_path) == report_path
    assert "FULL OCR PAGE ONE" in markdown_path.read_text(encoding="utf-8")
    assert report["source_relpath"] == "data/_needs_ocr/python/sample.pdf"
    assert report["processed_page_count"] == 2
    assert report["pages"][0]["confidence"] == 0.8
    assert report["pages"][1]["warnings"][0].startswith("ocr_failed:")
    assert "traceback should not appear" not in json.dumps(report, ensure_ascii=False)
    assert not (
        local_tmp_path / "reports" / "ocr" / "tmp" / "sample" / "page_0001.png"
    ).exists()


def test_ocr_pdf_to_markdown_creates_default_engine_without_real_paddleocr(
    local_tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "render_pdf_pages", _fake_render)
    calls: list[dict[str, str]] = []

    def fake_create_ocr_engine(*, engine, lang):
        calls.append({"engine": engine, "lang": lang})
        return StaticOCREngine(pages=((OCRLine(text="fake default OCR text"),),))

    monkeypatch.setattr(pipeline, "create_ocr_engine", fake_create_ocr_engine)
    pdf_path = local_tmp_path / "data" / "_needs_ocr" / "python" / "default.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF fake")

    result = ocr_pdf_to_markdown(
        pdf_path,
        subject="python",
        project_root=local_tmp_path,
        engine=None,
        engine_name="paddleocr",
        lang="ch",
        pages=(1,),
    )

    markdown = Path(result.markdown_path).read_text(encoding="utf-8")

    assert calls == [{"engine": "paddleocr", "lang": "ch"}]
    assert "fake default OCR text" in markdown
    assert "<!-- ocr_engine: paddleocr -->" in markdown
    assert "<!-- ocr_lang: ch -->" in markdown


def test_ocr_pdf_batch_records_processed_skipped_and_failed(
    local_tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "render_pdf_pages", _fake_render)
    input_dir = local_tmp_path / "pdfs"
    input_dir.mkdir()
    existing_pdf = input_dir / "existing.pdf"
    partial_pdf = input_dir / "partial.pdf"
    fresh_pdf = input_dir / "fresh.pdf"
    for pdf in (existing_pdf, partial_pdf, fresh_pdf):
        pdf.write_bytes(b"%PDF fake")

    output_dir = local_tmp_path / "out"
    report_dir = local_tmp_path / "reports"
    output_dir.mkdir()
    report_dir.mkdir()
    (output_dir / "existing.md").write_text("done", encoding="utf-8")
    (report_dir / "existing_ocr_report.json").write_text("{}", encoding="utf-8")
    (output_dir / "partial.md").write_text("half done", encoding="utf-8")

    report = ocr_pdf_batch(
        input_dir,
        subject="python",
        project_root=local_tmp_path,
        output_dir=output_dir,
        report_dir=report_dir,
        engine=StaticOCREngine(pages=((OCRLine(text="fresh text"),),)),
    ).to_dict()

    assert [item["source_file"] for item in report["skipped_existing"]] == [
        "existing.pdf"
    ]
    assert [item["source_file"] for item in report["failed"]] == ["partial.pdf"]
    assert [item["source_file"] for item in report["processed"]] == ["fresh.pdf"]
    assert report["failed"][0]["error"] == "partial_existing_output"


def test_ocr_pdf_batch_overwrite_processes_existing_outputs(
    local_tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "render_pdf_pages", _fake_render)
    input_dir = local_tmp_path / "pdfs"
    input_dir.mkdir()
    pdf_path = input_dir / "existing.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    output_dir = local_tmp_path / "out"
    report_dir = local_tmp_path / "reports"
    output_dir.mkdir()
    report_dir.mkdir()
    (output_dir / "existing.md").write_text("old", encoding="utf-8")
    (report_dir / "existing_ocr_report.json").write_text("{}", encoding="utf-8")

    report = ocr_pdf_batch(
        input_dir,
        subject="python",
        project_root=local_tmp_path,
        output_dir=output_dir,
        report_dir=report_dir,
        engine=StaticOCREngine(pages=((OCRLine(text="new text"),),)),
        overwrite=True,
        pages=(1,),
    ).to_dict()

    assert len(report["processed"]) == 1
    assert not report["skipped_existing"]
    assert "new text" in (output_dir / "existing.md").read_text(encoding="utf-8")
