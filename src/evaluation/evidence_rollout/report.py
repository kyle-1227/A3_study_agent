"""Content-free projections for evidence rollout evaluation artifacts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from src.evaluation.evidence_rollout.contracts import (
    DecisionStatus,
    EvidenceRolloutDecisionV2,
    ExecutionMode,
    Sha256Digest,
)
from src.rag.parent_child.evidence_evaluation import EvidenceActivationMetrics


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EvidenceExecutionStatusSummaryV2(_StrictFrozenModel):
    schema_version: Literal["evidence_execution_status_summary_v2"]
    success: int
    failed: int
    blocked: int
    not_executed: int

    @model_validator(mode="after")
    def validate_non_negative_counts(self) -> Self:
        if any(
            count < 0
            for count in (
                self.success,
                self.failed,
                self.blocked,
                self.not_executed,
            )
        ):
            raise ValueError("execution status counts must be non-negative")
        return self


class EvidenceRolloutSafeReportV2(_StrictFrozenModel):
    """Safe projection; its closed schema cannot carry request or provider content."""

    schema_version: Literal["evidence_rollout_safe_report_v2"]
    run_id: str
    execution_mode: ExecutionMode
    status: DecisionStatus
    activation_allowed: bool
    benchmark_eligible: bool
    rollout_activation_enabled: bool
    reason_codes: list[str]
    dataset_id: str
    generation_id: str
    expected_execution_count: int
    successful_execution_count: int
    reviewed_execution_count: int
    variant_matrix_complete: bool
    execution_statuses: EvidenceExecutionStatusSummaryV2
    dataset_fingerprint: Sha256Digest
    knowledge_graph_data_version: str
    knowledge_graph_artifact_fingerprint: Sha256Digest
    case_binding_inventory_fingerprint: Sha256Digest
    execution_config_fingerprint: Sha256Digest
    benchmark_config_fingerprint: Sha256Digest
    rollout_config_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    generation_manifest_fingerprint: Sha256Digest
    executor_fingerprint: Sha256Digest
    review_protocol_fingerprint: Sha256Digest
    review_bundle_fingerprint: Sha256Digest
    decision_fingerprint: Sha256Digest
    metrics: EvidenceActivationMetrics | None

    @model_validator(mode="after")
    def validate_projection_counts(self) -> Self:
        total = (
            self.execution_statuses.success
            + self.execution_statuses.failed
            + self.execution_statuses.blocked
            + self.execution_statuses.not_executed
        )
        if total != self.expected_execution_count:
            raise ValueError("execution status summary must cover every slot")
        if self.execution_statuses.success != self.successful_execution_count:
            raise ValueError("success summary must match decision count")
        return self


def build_safe_report(
    decision: EvidenceRolloutDecisionV2,
) -> EvidenceRolloutSafeReportV2:
    """Project one validated decision without accepting any content-bearing input."""

    if not isinstance(decision, EvidenceRolloutDecisionV2):
        raise TypeError("decision must be EvidenceRolloutDecisionV2")
    counts = {status: 0 for status in ("success", "failed", "blocked", "not_executed")}
    for record in decision.execution_records:
        counts[record.status] += 1
    return EvidenceRolloutSafeReportV2(
        schema_version="evidence_rollout_safe_report_v2",
        run_id=decision.run_id,
        execution_mode=decision.execution_mode,
        status=decision.status,
        activation_allowed=decision.activation_allowed,
        benchmark_eligible=decision.benchmark_eligible,
        rollout_activation_enabled=decision.rollout_activation_enabled,
        reason_codes=list(decision.reason_codes),
        dataset_id=decision.dataset_id,
        generation_id=decision.generation_id,
        expected_execution_count=decision.expected_execution_count,
        successful_execution_count=decision.successful_execution_count,
        reviewed_execution_count=decision.reviewed_execution_count,
        variant_matrix_complete=decision.variant_matrix_complete,
        execution_statuses=EvidenceExecutionStatusSummaryV2(
            schema_version="evidence_execution_status_summary_v2",
            success=counts["success"],
            failed=counts["failed"],
            blocked=counts["blocked"],
            not_executed=counts["not_executed"],
        ),
        dataset_fingerprint=decision.dataset_fingerprint,
        knowledge_graph_data_version=decision.knowledge_graph_data_version,
        knowledge_graph_artifact_fingerprint=(
            decision.knowledge_graph_artifact_fingerprint
        ),
        case_binding_inventory_fingerprint=(
            decision.case_binding_inventory_fingerprint
        ),
        execution_config_fingerprint=decision.execution_config_fingerprint,
        benchmark_config_fingerprint=decision.benchmark_config_fingerprint,
        rollout_config_fingerprint=decision.rollout_config_fingerprint,
        runtime_fingerprint=decision.runtime_fingerprint,
        generation_manifest_fingerprint=(decision.generation_manifest_fingerprint),
        executor_fingerprint=decision.executor_fingerprint,
        review_protocol_fingerprint=decision.review_protocol_fingerprint,
        review_bundle_fingerprint=decision.review_bundle_fingerprint,
        decision_fingerprint=decision.decision_fingerprint,
        metrics=(
            decision.activation_decision.metrics
            if decision.activation_decision is not None
            else None
        ),
    )


def render_safe_report_markdown(report: EvidenceRolloutSafeReportV2) -> bytes:
    """Render only the closed safe projection; never accept dataset or output text."""

    if not isinstance(report, EvidenceRolloutSafeReportV2):
        raise TypeError("report must be EvidenceRolloutSafeReportV2")
    reasons = ", ".join(report.reason_codes) if report.reason_codes else "none"
    status = report.execution_statuses
    lines = [
        "# Evidence rollout evaluation",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Status: `{report.status}`",
        f"- Execution mode: `{report.execution_mode}`",
        f"- Activation allowed: `{str(report.activation_allowed).lower()}`",
        f"- Benchmark eligible: `{str(report.benchmark_eligible).lower()}`",
        "- Rollout activation enabled: "
        f"`{str(report.rollout_activation_enabled).lower()}`",
        f"- Reason codes: `{reasons}`",
        "",
        "## Completeness",
        "",
        f"- Expected executions: `{report.expected_execution_count}`",
        f"- Successful executions: `{report.successful_execution_count}`",
        f"- Human-reviewed executions: `{report.reviewed_execution_count}`",
        f"- Variant matrix complete: `{str(report.variant_matrix_complete).lower()}`",
        f"- Status counts: success={status.success}, failed={status.failed}, "
        f"blocked={status.blocked}, not_executed={status.not_executed}",
        "",
        "## Bound identities",
        "",
        f"- Dataset: `{report.dataset_id}` / `{report.dataset_fingerprint}`",
        "- Knowledge graph: "
        f"`{report.knowledge_graph_data_version}` / "
        f"`{report.knowledge_graph_artifact_fingerprint}`",
        f"- Case/target inventory: `{report.case_binding_inventory_fingerprint}`",
        f"- Generation: `{report.generation_id}`",
        f"- Generation manifest: `{report.generation_manifest_fingerprint}`",
        f"- Execution config: `{report.execution_config_fingerprint}`",
        f"- Benchmark config: `{report.benchmark_config_fingerprint}`",
        f"- Rollout config: `{report.rollout_config_fingerprint}`",
        f"- Runtime: `{report.runtime_fingerprint}`",
        f"- Executor: `{report.executor_fingerprint}`",
        f"- Review protocol: `{report.review_protocol_fingerprint}`",
        f"- Review bundle: `{report.review_bundle_fingerprint}`",
        f"- Decision: `{report.decision_fingerprint}`",
    ]
    if report.metrics is not None:
        lines.extend(["", "## Aggregate metrics", ""])
        for name, value in report.metrics.model_dump(mode="json").items():
            lines.append(f"- {name}: `{value}`")
    return ("\n".join(lines) + "\n").encode("utf-8")


__all__ = [
    "EvidenceExecutionStatusSummaryV2",
    "EvidenceRolloutSafeReportV2",
    "build_safe_report",
    "render_safe_report_markdown",
]
