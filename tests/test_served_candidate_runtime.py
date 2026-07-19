from pathlib import Path

import pytest

from src.config.rag_index_config import load_rag_index_config
from src.graph.served_candidate import (
    ServedPrimaryRuntimeError,
    _runtime_index_config,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "config" / "rag" / "index.production.yaml"


def test_runtime_index_config_revalidates_mounted_primary_root(tmp_path: Path) -> None:
    index_root = tmp_path / "parent_child"
    index_root.mkdir()
    source = load_rag_index_config(PRODUCTION_CONFIG)

    runtime = _runtime_index_config(source=source, index_root=index_root)

    assert runtime.storage.index_root == index_root.resolve()
    assert runtime.embedding == source.embedding
    assert runtime.reranker == source.reranker


def test_runtime_index_config_rejects_missing_mount(tmp_path: Path) -> None:
    source = load_rag_index_config(PRODUCTION_CONFIG)

    with pytest.raises(ServedPrimaryRuntimeError, match="existing"):
        _runtime_index_config(source=source, index_root=tmp_path / "missing")


def test_primary_serving_module_has_no_registry_or_sealed_runtime_dependency() -> None:
    source = (ROOT / "src" / "graph" / "served_candidate.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "GenerationRegistry",
        "validate_sealed_generation",
        "PARENT_CHILD_GENERATION_ID",
        "rollout_activation_enabled",
        "rollout_shadow_enabled",
    ):
        assert forbidden not in source
