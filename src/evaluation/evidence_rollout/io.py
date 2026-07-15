"""Strict JSON loading and atomic publication for rollout evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from src.evaluation.evidence_rollout.contracts import EvidenceRolloutDecisionV2
from src.evaluation.evidence_rollout.report import (
    EvidenceRolloutSafeReportV2,
    render_safe_report_markdown,
)
from src.rag.parent_child.project_paths import (
    atomic_write_project_bytes,
    resolve_project_path,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


class EvidenceRolloutArtifactError(RuntimeError):
    """Typed artifact failure with a content-free reason code."""

    def __init__(self, *, code: str) -> None:
        self.code = code
        super().__init__(code)


def canonical_model_bytes(model: BaseModel) -> bytes:
    if not isinstance(model, BaseModel):
        raise TypeError("model must be a validated Pydantic model")
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def load_canonical_json_model(path: Path, model_type: type[ModelT]) -> ModelT:
    """Load one exact canonical model without repairing or normalizing its input."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EvidenceRolloutArtifactError(code="artifact_read_failed") from error
    try:
        model = model_type.model_validate_json(payload)
    except ValidationError as error:
        raise EvidenceRolloutArtifactError(code="artifact_contract_invalid") from error
    if payload != canonical_model_bytes(model):
        raise EvidenceRolloutArtifactError(code="artifact_not_canonical")
    return model


def publish_evidence_rollout_bundle(
    *,
    project_root: Path,
    output_directory: Path,
    decision: EvidenceRolloutDecisionV2,
    report: EvidenceRolloutSafeReportV2,
) -> Path:
    """Publish decision and safe reports together through one staging directory."""

    if not isinstance(decision, EvidenceRolloutDecisionV2):
        raise TypeError("decision must be EvidenceRolloutDecisionV2")
    if not isinstance(report, EvidenceRolloutSafeReportV2):
        raise TypeError("report must be EvidenceRolloutSafeReportV2")
    output = resolve_project_path(
        project_root,
        output_directory,
        must_exist=False,
    )
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = resolve_project_path(project_root, output, must_exist=False)
    stage = output.parent / f".{output.name}.staging-{uuid4().hex}"
    stage = resolve_project_path(project_root, stage, must_exist=False)
    stage.mkdir()
    try:
        atomic_write_project_bytes(
            project_root,
            stage / "activation_decision.json",
            canonical_model_bytes(decision),
            overwrite=False,
        )
        atomic_write_project_bytes(
            project_root,
            stage / "safe_report.json",
            canonical_model_bytes(report),
            overwrite=False,
        )
        atomic_write_project_bytes(
            project_root,
            stage / "safe_report.md",
            render_safe_report_markdown(report),
            overwrite=False,
        )
        stage.replace(output)
    except BaseException:
        if stage.exists():
            if stage.parent != output.parent:
                raise EvidenceRolloutArtifactError(
                    code="staging_directory_boundary_mismatch"
                )
            shutil.rmtree(stage)
        raise
    return output


__all__ = [
    "EvidenceRolloutArtifactError",
    "canonical_model_bytes",
    "load_canonical_json_model",
    "publish_evidence_rollout_bundle",
]
