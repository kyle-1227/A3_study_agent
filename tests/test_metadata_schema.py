from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
from langchain_core.documents import Document

from scripts import reset_index
from src.rag.indexer import _content_id
from src.rag.loader import load_documents

SCALAR_TYPES = (str, int, float, bool, type(None))
REQUIRED_STABLE_FIELDS = {
    "doc_id",
    "chunk_id",
    "source_relpath",
    "source_file_sha1",
    "source_file_size",
    "chunk_index",
    "chunk_policy_version",
    "index_version",
    "content_sha1",
    "chunk_chars",
}


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def test_load_documents_outputs_scalar_metadata_with_stable_ids(local_tmp_path):
    source = local_tmp_path / "notes_2026.txt"
    source.write_text("Header\n\nUseful source content. " * 80, encoding="utf-8")

    docs = load_documents(local_tmp_path, subject="general", doc_type="course_material")

    assert docs
    for doc in docs:
        assert REQUIRED_STABLE_FIELDS.issubset(doc.metadata)
        assert all(isinstance(value, SCALAR_TYPES) for value in doc.metadata.values())
        assert doc.metadata["subject"] == "general"
        assert doc.metadata["source_file"] == "notes_2026.txt"
        assert doc.metadata["year"] == "2026"
        assert doc.metadata["doc_type"] == "course_material"
        assert doc.metadata["source_relpath"] == f"{local_tmp_path.name}/notes_2026.txt"
        assert doc.metadata["chunk_chars"] == len(doc.page_content)
        assert "section_path" not in doc.metadata
        assert "parent_id" not in doc.metadata
        assert not any(key.startswith("section_") for key in doc.metadata)
        assert any(key.startswith("cleaning_") for key in doc.metadata)


def test_indexer_content_id_prefers_chunk_id():
    doc = Document(page_content="content", metadata={"chunk_id": " stable_chunk_id "})

    assert _content_id(doc) == "stable_chunk_id"


def test_indexer_content_id_keeps_legacy_md5_behavior_without_chunk_id():
    doc = Document(page_content="content", metadata={"source_file": "legacy.txt"})
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

    assert _content_id(doc) == f"legacy.txt_{digest}"


def test_reset_index_dry_run_subprocess_does_not_delete(local_tmp_path):
    chroma_dir = local_tmp_path / "configured_chroma"
    chroma_dir.mkdir()
    env = os.environ.copy()
    env["CHROMA_PERSIST_DIR"] = str(chroma_dir)

    result = subprocess.run(
        [sys.executable, "scripts/reset_index.py"],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "=== reset_index ===" in result.stdout
    assert "Refusing to delete without --yes." in result.stdout
    assert str(chroma_dir) in result.stdout
    assert chroma_dir.exists()


def test_reset_index_remove_targets_deletes_only_allowed_targets(local_tmp_path):
    project_root = local_tmp_path / "project"
    chroma_dir = project_root / "configured_chroma"
    reports_dir = project_root / "reports"
    data_dir = project_root / "data"
    chroma_dir.mkdir(parents=True)
    reports_dir.mkdir()
    data_dir.mkdir()
    build_manifest = reports_dir / "build_manifest.json"
    parent_chunks = reports_dir / "parent_chunks.jsonl"
    chunk_audit = reports_dir / "chunk_audit_report.json"
    build_manifest.write_text("{}", encoding="utf-8")
    parent_chunks.write_text("{}", encoding="utf-8")
    chunk_audit.write_text("{}", encoding="utf-8")

    targets = reset_index.reset_targets(
        project_root=project_root, persist_directory=chroma_dir
    )
    messages = reset_index.remove_targets(targets, project_root=project_root)

    assert any(message.startswith("Removed:") for message in messages)
    assert not chroma_dir.exists()
    assert not build_manifest.exists()
    assert not parent_chunks.exists()
    assert reports_dir.exists()
    assert chunk_audit.exists()
    assert data_dir.exists()


def test_reset_index_resolves_env_configured_persist_dir(local_tmp_path, monkeypatch):
    project_root = local_tmp_path / "project"
    configured = local_tmp_path / "configured_chroma"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(configured))

    assert (
        reset_index.resolve_chroma_persist_dir(project_root=project_root)
        == configured.resolve()
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        ".",
        "..",
        "data",
        ".env",
        "reports",
    ],
)
def test_reset_index_refuses_protected_project_paths(local_tmp_path, relative_path):
    project_root = local_tmp_path / "project"
    target = project_root / relative_path
    project_root.mkdir(parents=True, exist_ok=True)
    target.mkdir(
        parents=True, exist_ok=True
    ) if relative_path != ".env" else target.touch()

    with pytest.raises(ValueError):
        reset_index.validate_reset_targets([target], project_root=project_root)
