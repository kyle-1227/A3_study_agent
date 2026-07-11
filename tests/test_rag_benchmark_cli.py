from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from scripts.run_parent_child_benchmark import (
    BenchmarkCliError,
    _parser,
    _select_requested_generation,
    _write_failure_artifact,
)
from src.rag.parent_child.registry import GenerationRegistry, GenerationRegistryRecord


def _record(generation_id: str) -> GenerationRegistryRecord:
    now = datetime.now(UTC)
    return GenerationRegistryRecord(
        generation_id=generation_id,
        state="READY",
        directory_relative_path=generation_id,
        manifest_sha256="a" * 64,
        created_at_utc=now,
        updated_at_utc=now,
        failure_code=None,
        failure_type=None,
    )


class _RegistryWithNoDeploymentLookup:
    def __init__(self, record: GenerationRegistryRecord) -> None:
        self._record = record
        self.requested_ids: list[str] = []

    def get_generation(self, generation_id: str) -> GenerationRegistryRecord:
        self.requested_ids.append(generation_id)
        return self._record

    def deployment(self) -> object:
        raise AssertionError("benchmark must not read an active deployment pointer")


def test_benchmark_selects_only_the_explicit_generation_id() -> None:
    requested_generation_id = "candidate-requested"
    registry = _RegistryWithNoDeploymentLookup(_record(requested_generation_id))

    selected = _select_requested_generation(
        registry=cast(GenerationRegistry, registry),
        candidate_generation_id=requested_generation_id,
    )

    assert selected.generation_id == requested_generation_id
    assert registry.requested_ids == [requested_generation_id]


def test_benchmark_rejects_a_registry_record_for_another_generation() -> None:
    registry = _RegistryWithNoDeploymentLookup(_record("candidate-other"))

    with pytest.raises(BenchmarkCliError, match="different generation"):
        _select_requested_generation(
            registry=cast(GenerationRegistry, registry),
            candidate_generation_id="candidate-requested",
        )


def test_benchmark_failure_marker_does_not_persist_exception_body(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    sentinel = "BENCHMARK_PROVIDER_BODY_SECRET"

    _write_failure_artifact(
        root=project_root,
        output_dir=project_root / "artifacts" / "benchmark",
        error=RuntimeError(sentinel),
    )

    failure = project_root / "artifacts" / "benchmark.failure.json"
    payload = failure.read_bytes()
    assert sentinel.encode("utf-8") not in payload
    assert b"RuntimeError" in payload


def test_benchmark_cli_requires_all_explicit_inputs() -> None:
    with pytest.raises(SystemExit) as missing:
        _parser().parse_args([])
    assert missing.value.code == 2
