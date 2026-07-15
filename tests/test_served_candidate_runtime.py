from __future__ import annotations

from pathlib import Path

import pytest

from src.config.rag_index_config import load_rag_index_config
from src.graph.served_candidate import (
    ServedCandidateRuntimeError,
    _runtime_index_config,
)


ROOT = Path(__file__).resolve().parents[1]
INACTIVE_CONFIG = ROOT / "config" / "rag" / "index.production-candidate.inactive.yaml"


def test_runtime_index_config_revalidates_mounted_storage_root(tmp_path: Path) -> None:
    index_root = tmp_path / "parent_child"
    index_root.mkdir()
    source = load_rag_index_config(INACTIVE_CONFIG)

    runtime = _runtime_index_config(source=source, index_root=index_root)

    assert runtime.storage.index_root == index_root.resolve()
    assert (
        runtime.storage.registry_path
        == (index_root / "generation_registry.sqlite").resolve()
    )
    assert runtime.embedding == source.embedding
    assert runtime.reranker == source.reranker


def test_runtime_index_config_rejects_missing_mount(tmp_path: Path) -> None:
    source = load_rag_index_config(INACTIVE_CONFIG)

    with pytest.raises(ServedCandidateRuntimeError, match="existing"):
        _runtime_index_config(
            source=source,
            index_root=tmp_path / "missing",
        )
