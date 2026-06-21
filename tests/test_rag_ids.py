from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
from langchain_core.documents import Document

from src.rag.ids import (
    CHUNK_POLICY_VERSION,
    INDEX_VERSION,
    enrich_chunk_metadata,
    enrich_source_metadata,
    make_chunk_id,
    make_doc_id,
    make_source_relpath,
    normalize_for_hash,
    sha1_file,
    sha1_text,
)

SCALAR_TYPES = (str, int, float, bool, type(None))


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def test_sha1_text_is_stable():
    assert sha1_text("same") == sha1_text("same")
    assert sha1_text("same") != sha1_text("different")


def test_normalize_for_hash_collapses_whitespace():
    assert normalize_for_hash("A   B\n\nC\tD") == "A B C D"


def test_make_source_relpath_uses_posix_path_inside_project(local_tmp_path):
    project_root = local_tmp_path / "project"
    source = project_root / "data" / "python" / "x.pdf"
    source.parent.mkdir(parents=True)
    source.write_text("content", encoding="utf-8")

    assert make_source_relpath(source, project_root=project_root) == "data/python/x.pdf"


def test_make_source_relpath_marks_external_files(local_tmp_path):
    project_root = local_tmp_path / "project"
    project_root.mkdir()
    source = local_tmp_path / "outside.txt"
    source.write_text("content", encoding="utf-8")

    assert (
        make_source_relpath(source, project_root=project_root) == "external/outside.txt"
    )


def test_make_doc_id_is_stable_and_includes_file_sha1():
    doc_id = make_doc_id(
        subject="python",
        source_relpath="data/python/a.txt",
        file_sha1="abc",
    )

    assert doc_id == make_doc_id(
        subject="python",
        source_relpath="data/python/a.txt",
        file_sha1="abc",
    )
    assert doc_id != make_doc_id(
        subject="python",
        source_relpath="data/python/b.txt",
        file_sha1="abc",
    )
    assert doc_id != make_doc_id(
        subject="python",
        source_relpath="data/python/a.txt",
        file_sha1="def",
    )


def test_make_chunk_id_is_stable_and_depends_on_chunk_index():
    chunk_id = make_chunk_id(
        doc_id="doc_1",
        chunk_policy_version=CHUNK_POLICY_VERSION,
        chunk_index=0,
        content_sha1="abc",
    )

    assert chunk_id == make_chunk_id(
        doc_id="doc_1",
        chunk_policy_version=CHUNK_POLICY_VERSION,
        chunk_index=0,
        content_sha1="abc",
    )
    assert chunk_id != make_chunk_id(
        doc_id="doc_1",
        chunk_policy_version=CHUNK_POLICY_VERSION,
        chunk_index=1,
        content_sha1="abc",
    )


def test_enrich_source_metadata_adds_scalar_stable_fields(local_tmp_path):
    project_root = local_tmp_path / "project"
    source = project_root / "data" / "python" / "notes.txt"
    source.parent.mkdir(parents=True)
    source.write_text("hello world", encoding="utf-8")
    metadata = {
        "subject": "python",
        "source_file": "notes.txt",
        "doc_type": "course_material",
    }

    enriched = enrich_source_metadata(
        metadata,
        source_path=source,
        subject="python",
        project_root=project_root,
    )

    assert enriched["doc_id"].startswith("doc_")
    assert enriched["source_relpath"] == "data/python/notes.txt"
    assert enriched["source_file_sha1"] == sha1_file(source)
    assert enriched["source_file_size"] == source.stat().st_size
    assert enriched["index_version"] == INDEX_VERSION
    assert all(isinstance(value, SCALAR_TYPES) for value in enriched.values())


def test_enrich_chunk_metadata_adds_scalar_stable_fields():
    doc = Document(page_content="hello   world\nagain", metadata={"doc_id": "doc_1"})

    enriched = enrich_chunk_metadata(doc, doc_id="doc_1", chunk_index=0)
    enriched_again = enrich_chunk_metadata(doc, doc_id="doc_1", chunk_index=0)

    assert enriched.metadata["chunk_id"] == enriched_again.metadata["chunk_id"]
    assert enriched.metadata["chunk_id"].startswith("chunk_")
    assert enriched.metadata["chunk_index"] == 0
    assert enriched.metadata["chunk_policy_version"] == CHUNK_POLICY_VERSION
    assert enriched.metadata["content_sha1"] == sha1_text("hello world again")
    assert enriched.metadata["chunk_chars"] == len(doc.page_content)
    assert enriched.metadata["index_version"] == INDEX_VERSION
    assert all(isinstance(value, SCALAR_TYPES) for value in enriched.metadata.values())
