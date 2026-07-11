"""Immutable staging, sealing, and verification for generation directories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.rag.parent_child._storage_io import (
    ArtifactPathError,
    model_json_bytes,
    resolve_under_root,
    sha256_bytes,
    sha256_file,
    sha256_path,
    validate_generation_id,
)
from src.rag.parent_child.manifests import (
    CompletenessReport,
    GenerationManifest,
    ManifestContractError,
    read_strict_model,
    write_strict_model,
)


_OWNER_FILE = ".a3_generation_owner.json"
_MANIFEST_FILE = "manifest.json"
_VALIDATION_FILE = "validation_report.json"


class GenerationWorkspaceError(RuntimeError):
    """Raised when a generation cannot be staged or safely sealed."""


class GenerationOwnershipMarker(BaseModel):
    """Marker required before generation directories may be moved or cleaned."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    owner: Literal["a3_parent_child_generation"]


class SealedGeneration(BaseModel):
    """Identity returned after a generation directory is atomically published."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    generation_id: str = Field(min_length=1)
    directory_relative_path: str = Field(min_length=1)
    manifest_sha256: str = Field(min_length=64, max_length=64)


def generation_staging_relative_path(generation_id: str) -> str:
    """Return the canonical staging path for a validated generation ID."""

    return f".staging/{validate_generation_id(generation_id)}"


def generation_final_relative_path(generation_id: str) -> str:
    """Return the canonical final path for a validated generation ID."""

    return validate_generation_id(generation_id)


def _artifact_size(path: Path) -> int:
    if path.is_symlink():
        raise ArtifactPathError("symlink artifacts are forbidden")
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise ArtifactPathError("artifact path does not exist")
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ArtifactPathError("symlink artifacts are forbidden")
        if child.is_file():
            total += child.stat().st_size
    return total


def _read_owner_marker(
    index_root: str | Path,
    generation_relative_path: str,
) -> GenerationOwnershipMarker:
    marker = read_strict_model(
        index_root,
        f"{generation_relative_path}/{_OWNER_FILE}",
        GenerationOwnershipMarker,
    )
    expected_generation_id = generation_relative_path.rsplit("/", maxsplit=1)[-1]
    if marker.generation_id != expected_generation_id:
        raise GenerationWorkspaceError("generation ownership marker mismatch")
    return marker


class GenerationWorkspace:
    """A new staging directory that can be sealed exactly once."""

    def __init__(
        self,
        *,
        index_root: Path,
        generation_id: str,
        marker_schema_version: str,
    ) -> None:
        self.index_root = index_root
        self.generation_id = generation_id
        self.marker_schema_version = marker_schema_version
        self.staging_relative_path = generation_staging_relative_path(generation_id)
        self.final_relative_path = generation_final_relative_path(generation_id)

    @classmethod
    def create(
        cls,
        index_root: str | Path,
        generation_id: str,
        *,
        marker_schema_version: str,
    ) -> GenerationWorkspace:
        """Create an owned empty staging directory without touching active artifacts."""

        if not marker_schema_version:
            raise ValueError("marker_schema_version is required")
        validated_id = validate_generation_id(generation_id)
        root_path = Path(index_root).resolve(strict=False)
        root_path.mkdir(parents=True, exist_ok=True)
        staging_relative_path = generation_staging_relative_path(validated_id)
        final_relative_path = generation_final_relative_path(validated_id)
        staging_path = resolve_under_root(
            root_path,
            staging_relative_path,
            must_exist=False,
        )
        final_path = resolve_under_root(
            root_path,
            final_relative_path,
            must_exist=False,
        )
        if staging_path.exists() or final_path.exists():
            raise FileExistsError(
                f"generation already exists in staging or final storage: {validated_id}"
            )
        staging_path.mkdir(parents=True, exist_ok=False)
        marker = GenerationOwnershipMarker(
            schema_version=marker_schema_version,
            generation_id=validated_id,
            owner="a3_parent_child_generation",
        )
        write_strict_model(
            root_path,
            f"{staging_relative_path}/{_OWNER_FILE}",
            marker,
            overwrite=False,
        )
        return cls(
            index_root=root_path,
            generation_id=validated_id,
            marker_schema_version=marker_schema_version,
        )

    @property
    def staging_path(self) -> Path:
        """Return the verified absolute staging path."""

        return resolve_under_root(
            self.index_root,
            self.staging_relative_path,
            must_exist=True,
        )

    def seal(
        self,
        manifest: GenerationManifest,
        report: CompletenessReport,
    ) -> SealedGeneration:
        """Verify every artifact and atomically move staging to its final directory."""

        marker = _read_owner_marker(self.index_root, self.staging_relative_path)
        if marker.schema_version != self.marker_schema_version:
            raise GenerationWorkspaceError("generation marker schema mismatch")
        if manifest.generation_id != self.generation_id:
            raise GenerationWorkspaceError("manifest generation mismatch")
        if report.generation_id != self.generation_id:
            raise GenerationWorkspaceError("validation report generation mismatch")
        if not report.validation_passed:
            raise GenerationWorkspaceError("failed validation report cannot be sealed")
        if manifest.validation_report_sha256 != sha256_bytes(model_json_bytes(report)):
            raise GenerationWorkspaceError("validation report digest mismatch")
        if manifest.counts != report.counts or manifest.integrity != report.integrity:
            raise GenerationWorkspaceError("manifest and validation counts differ")
        if manifest.parent_id_set_sha256 != report.parent_id_set_sha256:
            raise GenerationWorkspaceError("manifest parent ID digest mismatch")
        if manifest.child_id_set_sha256 != report.child_id_set_sha256:
            raise GenerationWorkspaceError("manifest child ID digest mismatch")

        policy_descriptors = []
        subject_descriptors = []
        for descriptor in manifest.artifacts:
            if descriptor.relative_path in {
                _MANIFEST_FILE,
                _VALIDATION_FILE,
                _OWNER_FILE,
            }:
                raise GenerationWorkspaceError(
                    "manifest cannot describe cyclic generation control files"
                )
            artifact_path = resolve_under_root(
                self.staging_path,
                descriptor.relative_path,
                must_exist=True,
            )
            if sha256_path(artifact_path) != descriptor.sha256:
                raise GenerationWorkspaceError(
                    f"artifact digest mismatch: {descriptor.relative_path}"
                )
            if _artifact_size(artifact_path) != descriptor.size_bytes:
                raise GenerationWorkspaceError(
                    f"artifact size mismatch: {descriptor.relative_path}"
                )
            if descriptor.artifact_type == "policy_manifest":
                policy_descriptors.append(descriptor)
            if descriptor.artifact_type == "subject_manifest":
                subject_descriptors.append(descriptor)
        if len(policy_descriptors) != 1 or (
            policy_descriptors[0].sha256 != manifest.policy_manifest_sha256
        ):
            raise GenerationWorkspaceError("policy manifest descriptor mismatch")
        if len(subject_descriptors) != 1 or (
            subject_descriptors[0].sha256 != manifest.subject_manifest_sha256
        ):
            raise GenerationWorkspaceError("subject manifest descriptor mismatch")

        write_strict_model(
            self.staging_path,
            _VALIDATION_FILE,
            report,
            overwrite=False,
        )
        write_strict_model(
            self.staging_path,
            _MANIFEST_FILE,
            manifest,
            overwrite=False,
        )
        manifest_sha256 = sha256_file(self.staging_path / _MANIFEST_FILE)
        final_path = resolve_under_root(
            self.index_root,
            self.final_relative_path,
            must_exist=False,
        )
        if final_path.exists():
            raise FileExistsError(final_path)
        staging_path = self.staging_path
        if not staging_path.is_relative_to(
            self.index_root
        ) or not final_path.is_relative_to(self.index_root):
            raise GenerationWorkspaceError("generation move escapes index root")
        os.replace(staging_path, final_path)
        return SealedGeneration(
            generation_id=self.generation_id,
            directory_relative_path=self.final_relative_path,
            manifest_sha256=manifest_sha256,
        )


def validate_sealed_generation(
    index_root: str | Path,
    generation_id: str,
    *,
    expected_manifest_sha256: str,
    expected_marker_schema_version: str,
) -> GenerationManifest:
    """Revalidate a final generation before activation or runtime loading."""

    relative_path = generation_final_relative_path(generation_id)
    marker = _read_owner_marker(index_root, relative_path)
    if marker.schema_version != expected_marker_schema_version:
        raise GenerationWorkspaceError("generation marker schema mismatch")
    generation_path = resolve_under_root(index_root, relative_path, must_exist=True)
    manifest_path = resolve_under_root(
        generation_path,
        _MANIFEST_FILE,
        must_exist=True,
    )
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise GenerationWorkspaceError("sealed generation manifest digest mismatch")
    manifest = read_strict_model(
        generation_path,
        _MANIFEST_FILE,
        GenerationManifest,
    )
    if manifest.generation_id != generation_id:
        raise GenerationWorkspaceError("sealed manifest generation mismatch")
    report = read_strict_model(
        generation_path,
        _VALIDATION_FILE,
        CompletenessReport,
    )
    if not report.validation_passed:
        raise GenerationWorkspaceError("sealed validation report is not passing")
    if sha256_bytes(model_json_bytes(report)) != manifest.validation_report_sha256:
        raise GenerationWorkspaceError("sealed validation report digest mismatch")
    if manifest.counts != report.counts or manifest.integrity != report.integrity:
        raise GenerationWorkspaceError("sealed validation counts mismatch")
    if manifest.parent_id_set_sha256 != report.parent_id_set_sha256:
        raise GenerationWorkspaceError("sealed parent ID digest mismatch")
    if manifest.child_id_set_sha256 != report.child_id_set_sha256:
        raise GenerationWorkspaceError("sealed child ID digest mismatch")
    for descriptor in manifest.artifacts:
        artifact_path = resolve_under_root(
            generation_path,
            descriptor.relative_path,
            must_exist=True,
        )
        if sha256_path(artifact_path) != descriptor.sha256:
            raise ManifestContractError(
                f"sealed artifact digest mismatch: {descriptor.relative_path}"
            )
        if _artifact_size(artifact_path) != descriptor.size_bytes:
            raise ManifestContractError(
                f"sealed artifact size mismatch: {descriptor.relative_path}"
            )
    return manifest
