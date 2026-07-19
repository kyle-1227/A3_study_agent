from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings
import pytest

from src.rag.parent_child._storage_io import sha256_path
from src.rag.parent_child.chroma_runtime_snapshot import (
    CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
    ChromaRuntimeSnapshot,
    ChromaRuntimeSnapshotError,
)


def _build_chroma(path: Path) -> None:
    client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.create_collection(
        "snapshot_contract",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=["child-a"],
        documents=["alpha"],
        metadatas=[{"subject": "math"}],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    client.close()


def test_runtime_query_mutates_only_verified_chroma_copy(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    source = index_root / "generation-a" / "chroma_children"
    source.parent.mkdir(parents=True)
    _build_chroma(source)
    source_sha256 = sha256_path(source)

    with ChromaRuntimeSnapshot.create(
        index_root=index_root,
        source_directory=source,
        expected_source_sha256=source_sha256,
        owner_schema_version=CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
    ) as snapshot:
        assert snapshot.source_sha256 == source_sha256
        client = chromadb.PersistentClient(
            path=str(snapshot.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(
            "snapshot_contract",
            embedding_function=None,
        )
        result = collection.query(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=1,
        )
        assert result["ids"] == [["child-a"]]
        client.close()
        assert snapshot.snapshot_root.is_dir()

    assert sha256_path(source) == source_sha256
    runtime_root = index_root / ".runtime_chroma"
    assert runtime_root.is_dir()
    assert tuple(runtime_root.iterdir()) == ()


def test_runtime_snapshot_rejects_symlink_source(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    source = index_root / "generation-a" / "chroma_children"
    source.parent.mkdir(parents=True)
    _build_chroma(source)
    linked = index_root / "linked_chroma"
    try:
        linked.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {type(exc).__name__}")

    with pytest.raises(ChromaRuntimeSnapshotError, match="symlinks"):
        ChromaRuntimeSnapshot.create(
            index_root=index_root,
            source_directory=linked,
            expected_source_sha256=sha256_path(source),
            owner_schema_version=CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
        )


def test_runtime_snapshot_digest_mismatch_fails_before_creating_a_runtime_copy(
    tmp_path: Path,
) -> None:
    index_root = tmp_path / "indexes"
    source = index_root / "generation-a" / "chroma_children"
    source.parent.mkdir(parents=True)
    _build_chroma(source)

    with pytest.raises(ChromaRuntimeSnapshotError, match="differs"):
        ChromaRuntimeSnapshot.create(
            index_root=index_root,
            source_directory=source,
            expected_source_sha256="0" * 64,
            owner_schema_version=CHROMA_RUNTIME_OWNER_SCHEMA_VERSION,
        )
    assert not (index_root / ".runtime_chroma").exists()
