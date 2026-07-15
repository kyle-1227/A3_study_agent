"""Evaluate a sealed hermetic P0/PG/PR/PGR bundle without granting live proof.

The production live path is the evaluation package's ``LiveEvidenceVariantExecutor``
and requires four concrete adapters. This CLI deliberately accepts only a sealed
hermetic attempt bundle; it can validate contracts and thresholds but can never
authorize activation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.evidence_benchmark_config import (  # noqa: E402
    load_evidence_benchmark_config,
)
from src.config.rag_rollout_config import load_rag_rollout_config  # noqa: E402
from src.evaluation.evidence_rollout.contracts import (  # noqa: E402
    EvidenceEvaluationDatasetV1,
    EvidenceEvaluationRuntimeBindingV1,
    EvidenceVariantAttemptBatchV1,
    HumanSemanticReviewBatchV1,
    load_evidence_rollout_execution_config,
)
from src.evaluation.evidence_rollout.io import (  # noqa: E402
    EvidenceRolloutArtifactError,
    canonical_model_bytes,
    load_canonical_json_model,
    publish_evidence_rollout_bundle,
)
from src.evaluation.evidence_rollout.report import build_safe_report  # noqa: E402
from src.evaluation.evidence_rollout.runner import (  # noqa: E402
    SealedAttemptVariantExecutor,
    run_evidence_rollout_evaluation,
)
from src.rag.parent_child.manifests import GenerationManifest  # noqa: E402
from src.rag.parent_child.project_paths import (  # noqa: E402
    atomic_write_project_bytes,
    require_project_file,
    resolve_project_path,
    resolve_project_root,
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EvidenceRolloutCliFailureV1(_StrictFrozenModel):
    schema_version: Literal["evidence_rollout_cli_failure_v1"]
    status: Literal["blocked"]
    failure_code: str
    failure_type: str


class EvidenceRolloutCliError(RuntimeError):
    """Typed CLI failure whose code contains no evaluation content."""

    def __init__(self, *, code: str) -> None:
        self.code = code
        super().__init__(code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--rollout-config", type=Path, required=True)
    parser.add_argument("--review-protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--human-reviews", type=Path, required=True)
    parser.add_argument("--attempt-batch", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_utf8_text_fingerprint(path: Path) -> str:
    """Hash UTF-8 text with explicit LF semantics across checkout platforms."""

    try:
        text = path.read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceRolloutCliError(code="review_protocol_read_failed") from error
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized:
        raise EvidenceRolloutCliError(code="review_protocol_line_ending_invalid")
    return _sha256(normalized.encode("utf-8"))


def _load_generation_manifest(path: Path) -> tuple[GenerationManifest, str]:
    manifest = load_canonical_json_model(path, GenerationManifest)
    payload = path.read_bytes()
    return manifest, _sha256(payload)


def _validate_external_bindings(
    *,
    review_protocol_path: Path,
    expected_review_protocol_fingerprint: str,
    generation_manifest: GenerationManifest,
    generation_manifest_fingerprint: str,
    binding: EvidenceEvaluationRuntimeBindingV1,
) -> None:
    review_protocol_fingerprint = _canonical_utf8_text_fingerprint(review_protocol_path)
    if review_protocol_fingerprint != expected_review_protocol_fingerprint:
        raise EvidenceRolloutCliError(code="review_protocol_fingerprint_mismatch")
    if generation_manifest.generation_id != binding.generation_id:
        raise EvidenceRolloutCliError(code="generation_id_mismatch")
    if generation_manifest_fingerprint != binding.generation_manifest_fingerprint:
        raise EvidenceRolloutCliError(code="generation_manifest_fingerprint_mismatch")


def run_hermetic_evaluation(
    *,
    project_root: Path,
    execution_config_path: Path,
    benchmark_config_path: Path,
    rollout_config_path: Path,
    review_protocol_path: Path,
    dataset_path: Path,
    runtime_binding_path: Path,
    human_reviews_path: Path,
    attempt_batch_path: Path,
    generation_manifest_path: Path,
    output_directory: Path,
) -> int:
    """Run and publish one strict hermetic decision; return a gate exit code."""

    root = resolve_project_root(project_root)
    execution_path = require_project_file(root, execution_config_path)
    benchmark_path = require_project_file(root, benchmark_config_path)
    rollout_path = require_project_file(root, rollout_config_path)
    protocol_path = require_project_file(root, review_protocol_path)
    dataset_file = require_project_file(root, dataset_path)
    binding_file = require_project_file(root, runtime_binding_path)
    reviews_file = require_project_file(root, human_reviews_path)
    attempts_file = require_project_file(root, attempt_batch_path)
    manifest_file = require_project_file(root, generation_manifest_path)
    output = resolve_project_path(root, output_directory, must_exist=False)

    execution_config = load_evidence_rollout_execution_config(execution_path)
    benchmark_config = load_evidence_benchmark_config(benchmark_path)
    rollout_config = load_rag_rollout_config(rollout_path)
    dataset = load_canonical_json_model(dataset_file, EvidenceEvaluationDatasetV1)
    binding = load_canonical_json_model(
        binding_file,
        EvidenceEvaluationRuntimeBindingV1,
    )
    reviews = load_canonical_json_model(
        reviews_file,
        HumanSemanticReviewBatchV1,
    )
    attempts = load_canonical_json_model(
        attempts_file,
        EvidenceVariantAttemptBatchV1,
    )
    generation_manifest, generation_manifest_fingerprint = _load_generation_manifest(
        manifest_file
    )
    _validate_external_bindings(
        review_protocol_path=protocol_path,
        expected_review_protocol_fingerprint=(
            execution_config.human_review_protocol_fingerprint
        ),
        generation_manifest=generation_manifest,
        generation_manifest_fingerprint=generation_manifest_fingerprint,
        binding=binding,
    )
    decision = asyncio.run(
        run_evidence_rollout_evaluation(
            dataset=dataset,
            execution_config=execution_config,
            benchmark_config=benchmark_config,
            rollout_config=rollout_config,
            binding=binding,
            reviews=reviews,
            executor=SealedAttemptVariantExecutor(attempts),
        )
    )
    report = build_safe_report(decision)
    publish_evidence_rollout_bundle(
        project_root=root,
        output_directory=output,
        decision=decision,
        report=report,
    )
    print(
        json.dumps(
            {
                "activation_allowed": decision.activation_allowed,
                "decision_fingerprint": decision.decision_fingerprint,
                "run_id": decision.run_id,
                "status": decision.status,
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if decision.status == "pass" else 1


def _failure_code(error: Exception) -> str:
    if isinstance(error, (EvidenceRolloutCliError, EvidenceRolloutArtifactError)):
        return error.code
    return "evaluation_cli_failed"


def _safe_failure_type(error: Exception) -> str:
    value = type(error).__name__
    if not value or not value.replace("_", "").isalnum():
        return "UnexpectedException"
    return value[:128]


def _write_failure_artifact(
    *,
    project_root: Path,
    output_directory: Path,
    error: Exception,
) -> None:
    root = resolve_project_root(project_root)
    output = resolve_project_path(root, output_directory, must_exist=False)
    failure_path = output.with_name(output.name + ".failure.json")
    artifact = EvidenceRolloutCliFailureV1(
        schema_version="evidence_rollout_cli_failure_v1",
        status="blocked",
        failure_code=_failure_code(error),
        failure_type=_safe_failure_type(error),
    )
    atomic_write_project_bytes(
        root,
        failure_path,
        canonical_model_bytes(artifact),
        overwrite=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run_hermetic_evaluation(
            project_root=args.project_root,
            execution_config_path=args.execution_config,
            benchmark_config_path=args.benchmark_config,
            rollout_config_path=args.rollout_config,
            review_protocol_path=args.review_protocol,
            dataset_path=args.dataset,
            runtime_binding_path=args.runtime_binding,
            human_reviews_path=args.human_reviews,
            attempt_batch_path=args.attempt_batch,
            generation_manifest_path=args.generation_manifest,
            output_directory=args.output_dir,
        )
    except Exception as error:
        try:
            _write_failure_artifact(
                project_root=args.project_root,
                output_directory=args.output_dir,
                error=error,
            )
        except Exception:
            print(
                "evidence rollout failure artifact publication failed", file=sys.stderr
            )
            return 2
        print("evidence rollout evaluation failed before decision", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
