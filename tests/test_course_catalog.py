"""Tests for dynamic course subject discovery."""

from __future__ import annotations

from src.rag.course_catalog import get_available_subjects_from_data, normalize_subject


def test_normalize_subject():
    assert normalize_subject(" Machine-Learning ") == "machine_learning"
    assert normalize_subject("Computer Science!") == "computer_science"
    assert normalize_subject("法学-导论") == "法学_导论"


def test_get_available_subjects_from_data(tmp_path):
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "intro.md").write_text("Python", encoding="utf-8")
    (tmp_path / "Machine Learning").mkdir()
    (tmp_path / "Machine Learning" / "ml.md").write_text("ML", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.md").write_text("x", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_text("x", encoding="utf-8")
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "smoke.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "_needs_ocr").mkdir()
    (tmp_path / "_needs_ocr" / "scan.pdf").write_bytes(b"scan")
    (tmp_path / "unclassified").mkdir()
    (tmp_path / "unclassified" / "unknown.pdf").write_bytes(b"unknown")
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.txt").write_text("not a dir", encoding="utf-8")

    assert get_available_subjects_from_data(tmp_path) == ["machine_learning", "python"]


def test_get_available_subjects_missing_dir(tmp_path):
    assert get_available_subjects_from_data(tmp_path / "missing") == []
