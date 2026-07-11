from __future__ import annotations

from pathlib import Path

import pytest

from src.config.rag_index_config import CatalogConfig
from src.rag.subject_catalog import (
    SubjectCatalog,
    SubjectNormalizationCollisionError,
    SubjectPolicyMapError,
)


def _config(data_root: Path) -> CatalogConfig:
    return CatalogConfig(
        data_root=data_root,
        supported_extensions=(".pdf", ".md", ".txt"),
        excluded_exact_names=("ignored",),
        excluded_prefixes=("tmp_",),
        exclude_hidden=True,
        exclude_cache_directories=True,
        cache_directory_names=("__pycache__", ".cache"),
        exclude_unclassified=True,
        unclassified_directory_name="unclassified",
        exclude_needs_ocr=True,
        needs_ocr_directory_name="_needs_ocr",
        normalization_version="subject_id_v1",
        symlink_policy="reject",
    )


def test_catalog_discovers_supported_sources_and_excludes_quarantine(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    (data / "math").mkdir(parents=True)
    (data / "math" / "notes.txt").write_text("limits", encoding="utf-8")
    (data / "math" / ".gitkeep").write_text("", encoding="utf-8")
    (data / "_needs_ocr" / "math").mkdir(parents=True)
    (data / "_needs_ocr" / "math" / "scan.pdf").write_bytes(b"not-inspected")
    (data / "__pycache__").mkdir()

    snapshot = SubjectCatalog(
        config=_config(data),
        subject_policy_map={"math": "a" * 64},
    ).discover()

    assert snapshot.subject_ids() == ("math",)
    assert tuple(item.source_relpath for item in snapshot.source_entries()) == (
        "math/notes.txt",
    )


def test_catalog_rejects_normalization_collision(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for name in ("Machine Learning", "machine-learning"):
        directory = data / name
        directory.mkdir(parents=True)
        (directory / "notes.txt").write_text("content", encoding="utf-8")

    with pytest.raises(SubjectNormalizationCollisionError):
        SubjectCatalog(
            config=_config(data),
            subject_policy_map={"machine_learning": "a" * 64},
        ).discover()


def test_catalog_requires_exact_subject_policy_map(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "math").mkdir(parents=True)
    (data / "math" / "notes.txt").write_text("limits", encoding="utf-8")

    with pytest.raises(SubjectPolicyMapError, match="missing policy"):
        SubjectCatalog(config=_config(data), subject_policy_map={}).discover()
