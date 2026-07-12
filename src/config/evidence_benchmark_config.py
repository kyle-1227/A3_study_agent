"""Strict activation thresholds for resource-aware evidence orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, model_validator

from src.config._rag_config import StrictRagConfigModel, load_strict_rag_yaml

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


def _freeze_sequence(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


class EvidenceBenchmarkConfig(StrictRagConfigModel):
    """Every threshold required for a joint candidate activation decision."""

    schema_version: Literal["evidence_benchmark_config_v1"]
    required_variants: Annotated[
        tuple[Literal["P0", "PG", "PR", "PGR"], ...],
        BeforeValidator(_freeze_sequence),
    ]
    min_overall_weighted_coverage_lift: UnitFloat
    min_multi_weighted_coverage_lift: UnitFloat
    min_required_gap_reduction: UnitFloat
    max_evidence_precision_loss: UnitFloat
    max_simple_case_coverage_regression: UnitFloat
    max_premature_stop_rate: UnitFloat
    max_over_search_rate: UnitFloat
    min_source_routing_f1: UnitFloat
    min_resource_subject_recall: UnitFloat
    min_assignment_precision: UnitFloat
    min_claim_support_lift: UnitFloat
    min_ungrounded_fact_reduction: UnitFloat
    max_average_retrieval_cost_ratio: PositiveFloat
    max_initial_sufficient_cost_ratio: PositiveFloat
    max_p95_latency_ratio: PositiveFloat
    candidate_failure_policy: Literal["fail_fast"]

    @model_validator(mode="after")
    def validate_factorial_contract(self) -> "EvidenceBenchmarkConfig":
        if self.required_variants != ("P0", "PG", "PR", "PGR"):
            raise ValueError(
                "required_variants must be the canonical P0, PG, PR, PGR order"
            )
        return self


def load_evidence_benchmark_config(config_path: Path) -> EvidenceBenchmarkConfig:
    """Load one explicitly supplied gate file without fallback behavior."""

    return load_strict_rag_yaml(config_path, EvidenceBenchmarkConfig)


__all__ = ["EvidenceBenchmarkConfig", "load_evidence_benchmark_config"]
