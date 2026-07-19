from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.rag.parent_child.runtime_loader as runtime_loader

from src.rag.parent_child.primary import (
    PrimaryIndexMetadataV1,
    PrimaryIndexStateV1,
    PrimaryIndexValidationV1,
    PrimaryIndexWorkspace,
    load_primary_metadata,
    load_primary_state,
    primary_metadata_relative_path,
    primary_validation_relative_path,
)


_DIGEST = "a" * 64
_SUBJECTS = ("math",)
_POLICIES = {"math": "b" * 64}


def _metadata(revision: int) -> PrimaryIndexMetadataV1:
    return PrimaryIndexMetadataV1(
        schema_version="primary_index_metadata_v1",
        primary_revision=revision,
        artifact_identity=f"artifact-{revision}",
        built_at_utc=datetime(2026, 7, 19, tzinfo=UTC),
        collection_name="a3_parent_child_children",
        chroma_directory_relative_path="chroma_children",
        parent_store_relative_path="parents.sqlite",
        policy_manifest_relative_path="policy_manifest.json",
        subject_manifest_relative_path="subject_manifest.json",
        bm25_directory_relative_path="bm25",
        validation_relative_path="primary_validation.json",
        embedding_fingerprint=_DIGEST,
        embedding_dimension=3,
        distance_metric="cosine",
        bm25_tokenizer_fingerprint=_DIGEST,
        subject_policy_map=_POLICIES,
        available_subjects=_SUBJECTS,
        config_fingerprint=_DIGEST,
        validation_status="valid",
    )


def _validator(
    _staging: Path,
    metadata: PrimaryIndexMetadataV1,
) -> PrimaryIndexValidationV1:
    return PrimaryIndexValidationV1(
        schema_version="primary_index_validation_v1",
        primary_revision=metadata.primary_revision,
        artifact_identity=metadata.artifact_identity,
        validated_at_utc=datetime(2026, 7, 19, tzinfo=UTC),
        validation_status="valid",
        validated_subjects=metadata.available_subjects,
    )


def test_primary_publish_writes_only_validated_state_pointer(tmp_path: Path) -> None:
    workspace = PrimaryIndexWorkspace.create(index_root=tmp_path, build_id="build-1")
    result = workspace.publish(
        metadata=_metadata(1),
        validate_staging=_validator,
        now=datetime(2026, 7, 19, tzinfo=UTC),
    )

    state = load_primary_state(tmp_path)
    metadata = load_primary_metadata(tmp_path, state=state)
    assert result.primary_revision == 1
    assert state.primary_revision == metadata.primary_revision == 1
    assert state.metadata_relative_path == primary_metadata_relative_path(1)
    assert state.validation_relative_path == primary_validation_relative_path(1)
    assert not (tmp_path / "primary" / ".staging" / "build-1").exists()


def test_failed_staging_validation_never_replaces_current_primary(
    tmp_path: Path,
) -> None:
    first = PrimaryIndexWorkspace.create(index_root=tmp_path, build_id="build-1")
    first.publish(metadata=_metadata(1), validate_staging=_validator)
    before = load_primary_state(tmp_path)

    second = PrimaryIndexWorkspace.create(index_root=tmp_path, build_id="build-2")

    def fail(*_args: object) -> PrimaryIndexValidationV1:
        raise RuntimeError("incomplete staging")

    with pytest.raises(RuntimeError, match="incomplete staging"):
        second.publish(metadata=_metadata(2), validate_staging=fail)

    assert load_primary_state(tmp_path) == before
    assert second.staging_path.exists()
    assert not (tmp_path / "primary" / "revisions" / "r2").exists()


def test_state_rejects_control_paths_outside_its_revision() -> None:
    with pytest.raises(ValueError, match="control files"):
        PrimaryIndexStateV1(
            schema_version="primary_index_state_v1",
            primary_revision=1,
            active_directory_relative_path="primary/revisions/r1",
            metadata_relative_path="primary/revisions/r1/primary_metadata.json",
            validation_relative_path="primary/revisions/r2/primary_validation.json",
            updated_at_utc=datetime(2026, 7, 19, tzinfo=UTC),
            config_fingerprint=_DIGEST,
            validation_status="valid",
        )


def _raise(error: Exception):
    def raise_error(*_args: object, **_kwargs: object) -> object:
        raise error

    return raise_error


def test_runtime_loader_fails_closed_when_primary_state_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runtime_loader, "load_primary_state", _raise(FileNotFoundError("primary"))
    )

    with pytest.raises(runtime_loader.PrimaryRuntimeLoadError):
        runtime_loader._assert_primary_loaded(config=object(), index_root=tmp_path)


def test_runtime_loader_fails_closed_when_primary_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = object()
    monkeypatch.setattr(runtime_loader, "load_primary_state", lambda _root: state)
    monkeypatch.setattr(
        runtime_loader, "load_primary_metadata", _raise(FileNotFoundError("metadata"))
    )

    with pytest.raises(runtime_loader.PrimaryRuntimeLoadError):
        runtime_loader._assert_primary_loaded(config=object(), index_root=tmp_path)


def test_runtime_loader_validates_the_writable_snapshot_before_serving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = SimpleNamespace()
    metadata = SimpleNamespace(chroma_directory_relative_path="chroma_children")
    snapshot = SimpleNamespace(
        persist_directory=tmp_path / "runtime-chroma",
        closed=False,
    )

    def close_snapshot() -> None:
        snapshot.closed = True

    snapshot.close = close_snapshot
    config = SimpleNamespace(storage=SimpleNamespace(index_root=tmp_path))

    class ExpectedValidationStop(Exception):
        pass

    monkeypatch.setattr(
        runtime_loader,
        "_assert_primary_loaded",
        lambda **_kwargs: (state, metadata, tmp_path),
    )
    monkeypatch.setattr(
        runtime_loader,
        "ChromaRuntimeSnapshot",
        SimpleNamespace(create=lambda **_kwargs: snapshot),
    )

    def stop_after_snapshot(**kwargs: object) -> None:
        assert kwargs["chroma_snapshot"] is snapshot
        raise ExpectedValidationStop

    monkeypatch.setattr(
        runtime_loader,
        "validate_primary_revision",
        stop_after_snapshot,
    )

    with pytest.raises(ExpectedValidationStop):
        runtime_loader.load_primary_runtime(
            config=config,
            query_embedding_provider=object(),
            reranker=object(),
            bm25_tokenizer=lambda _text: (),
        )

    assert snapshot.closed is True
