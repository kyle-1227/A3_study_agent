from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.config.rag_index_config import load_rag_index_config
from src.graph.served_candidate import (
    ServedCandidateRuntimeError,
    _runtime_index_config,
    _validate_production_deployment,
)
from src.rag.parent_child.registry import DeploymentSnapshot


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "config" / "rag" / "index.production.yaml"


def test_runtime_index_config_revalidates_mounted_storage_root(tmp_path: Path) -> None:
    index_root = tmp_path / "parent_child"
    index_root.mkdir()
    source = load_rag_index_config(PRODUCTION_CONFIG)

    runtime = _runtime_index_config(source=source, index_root=index_root)

    assert runtime.storage.index_root == index_root.resolve()
    assert (
        runtime.storage.registry_path
        == (index_root / "generation_registry.sqlite").resolve()
    )
    assert runtime.embedding == source.embedding
    assert runtime.reranker == source.reranker


def test_runtime_index_config_rejects_missing_mount(tmp_path: Path) -> None:
    source = load_rag_index_config(PRODUCTION_CONFIG)

    with pytest.raises(ServedCandidateRuntimeError, match="existing"):
        _runtime_index_config(
            source=source,
            index_root=tmp_path / "missing",
        )


def _deployment(
    *,
    primary_generation_id: str | None = "generation-55",
    previous_generation_id: str | None = None,
    shadow_generation_id: str | None = None,
) -> DeploymentSnapshot:
    return DeploymentSnapshot(
        primary_generation_id=primary_generation_id,
        previous_generation_id=previous_generation_id,
        shadow_generation_id=shadow_generation_id,
        revision=1,
        updated_at_utc=datetime(2026, 7, 16, tzinfo=UTC),
    )


def test_production_deployment_accepts_exact_active_primary() -> None:
    _validate_production_deployment(
        generation_id="generation-55",
        deployment=_deployment(),
    )


@pytest.mark.parametrize(
    ("deployment", "expected_message"),
    [
        (_deployment(primary_generation_id=None), "active registry primary"),
        (_deployment(shadow_generation_id="shadow"), "empty shadow pointer"),
        (
            _deployment(previous_generation_id="generation-55"),
            "must be distinct",
        ),
    ],
)
def test_production_deployment_rejects_unsafe_pointer_state(
    deployment: DeploymentSnapshot,
    expected_message: str,
) -> None:
    with pytest.raises(
        ServedCandidateRuntimeError,
        match=expected_message,
    ):
        _validate_production_deployment(
            generation_id="generation-55",
            deployment=deployment,
        )
