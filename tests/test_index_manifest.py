from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest
from langchain_core.documents import Document

from src.rag.index_manifest import (
    DEFAULT_MANIFEST_NOTES,
    build_manifest_from_documents,
    write_build_manifest,
)


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def _doc(
    content: str,
    *,
    doc_id: str,
    source_file: str,
    source_relpath: str,
    chunk_index: int,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "doc_id": doc_id,
            "subject": "python",
            "source_file": source_file,
            "source_relpath": source_relpath,
            "source_file_sha1": f"sha1-{doc_id}",
            "source_file_size": 123,
            "chunk_index": chunk_index,
        },
    )


def test_build_manifest_from_documents_groups_sources():
    docs = [
        _doc(
            "chunk text 1",
            doc_id="doc_a",
            source_file="a.txt",
            source_relpath="data/python/a.txt",
            chunk_index=0,
        ),
        _doc(
            "chunk text 2",
            doc_id="doc_a",
            source_file="a.txt",
            source_relpath="data/python/a.txt",
            chunk_index=1,
        ),
        _doc(
            "chunk text 3",
            doc_id="doc_b",
            source_file="b.txt",
            source_relpath="data/python/b.txt",
            chunk_index=0,
        ),
    ]

    manifest = build_manifest_from_documents(
        docs,
        collection_name="a3_study_docs",
        chroma_persist_dir="D:/project/chroma_store",
        embedding_model="embedding-model",
    )

    assert manifest.total_chunks == 3
    assert manifest.source_count == 2
    assert manifest.splitter_mode == "recursive"
    assert manifest.chunk_policy_version == "recursive_v1"
    assert manifest.notes == DEFAULT_MANIFEST_NOTES
    assert [source.chunk_count for source in manifest.sources] == [2, 1]
    assert [source.source_file for source in manifest.sources] == ["a.txt", "b.txt"]


def test_manifest_to_dict_is_json_serializable_and_omits_chunk_text():
    docs = [
        _doc(
            "chunk text with OPENROUTER_API_KEY=secret-value",
            doc_id="doc_a",
            source_file="a.txt",
            source_relpath="data/python/a.txt",
            chunk_index=0,
        )
    ]
    manifest = build_manifest_from_documents(
        docs,
        collection_name="a3_study_docs",
        chroma_persist_dir="D:/project/chroma_store",
        embedding_model="embedding-model",
    )

    payload = manifest.to_dict()
    dumped = json.dumps(payload, ensure_ascii=False)

    assert payload["total_chunks"] == 1
    assert "chunk text" not in dumped
    assert "secret-value" not in dumped
    assert "api_key" not in dumped.lower()


def test_write_build_manifest_writes_json(local_tmp_path):
    docs = [
        _doc(
            "content",
            doc_id="doc_a",
            source_file="a.txt",
            source_relpath="data/python/a.txt",
            chunk_index=0,
        )
    ]
    manifest = build_manifest_from_documents(
        docs,
        collection_name="a3_study_docs",
        chroma_persist_dir="D:/project/chroma_store",
        embedding_model="embedding-model",
    )
    output = local_tmp_path / "reports" / "build_manifest.json"

    write_build_manifest(manifest, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["collection_name"] == "a3_study_docs"
    assert payload["splitter_mode"] == "recursive"
    assert payload["notes"] == DEFAULT_MANIFEST_NOTES
    assert payload["sources"][0]["source_relpath"] == "data/python/a.txt"


def test_build_manifest_infers_structure_splitter_mode():
    doc = _doc(
        "content",
        doc_id="doc_a",
        source_file="a.txt",
        source_relpath="data/python/a.txt",
        chunk_index=0,
    )
    doc.metadata["splitter_mode"] = "structure"
    doc.metadata["chunk_policy_version"] = "structure_v1"

    manifest = build_manifest_from_documents(
        [doc],
        collection_name="a3_study_docs",
        chroma_persist_dir="D:/project/chroma_store",
        embedding_model="embedding-model",
    )

    assert manifest.splitter_mode == "structure"
    assert manifest.chunk_policy_version == "structure_v1"
