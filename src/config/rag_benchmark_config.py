"""Strict benchmark and data-readiness configuration for parent-child RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator

from src.config._rag_config import (
    ConfigPath,
    ConfigPathTuple,
    NonBlankStr,
    NonBlankStrTuple,
    NonNegativeIntTuple,
    PositiveIntTuple,
    StrictRagConfigModel,
    load_strict_rag_yaml,
)


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


def _freeze_sequence(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _validate_unique(values: tuple[object, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


class RrfWeightPair(StrictRagConfigModel):
    vector_weight: PositiveFloat
    bm25_weight: PositiveFloat


class BenchmarkCandidateGrid(StrictRagConfigModel):
    """Bounded two-stage experiment search space; not production defaults."""

    parent_sizes: PositiveIntTuple
    parent_overlaps: NonNegativeIntTuple
    child_sizes: PositiveIntTuple
    child_overlaps: NonNegativeIntTuple
    vector_top_ks: PositiveIntTuple
    bm25_top_ks: PositiveIntTuple
    reranker_top_ns: PositiveIntTuple
    unique_parent_top_ks: PositiveIntTuple
    max_children_per_parent_values: PositiveIntTuple
    max_parents_per_source_values: PositiveIntTuple
    rrf_ks: PositiveIntTuple
    rrf_weight_pairs: Annotated[
        tuple[RrfWeightPair, ...],
        BeforeValidator(_freeze_sequence),
        Field(min_length=1),
    ]
    parent_support_lambdas: Annotated[
        tuple[UnitFloat, ...],
        BeforeValidator(_freeze_sequence),
        Field(min_length=1),
    ]
    full_parent_max_chars_values: PositiveIntTuple
    hit_window_chars_per_side_values: PositiveIntTuple

    @model_validator(mode="after")
    def _validate_grid(self) -> "BenchmarkCandidateGrid":
        sequence_fields = (
            "parent_sizes",
            "parent_overlaps",
            "child_sizes",
            "child_overlaps",
            "vector_top_ks",
            "bm25_top_ks",
            "reranker_top_ns",
            "unique_parent_top_ks",
            "max_children_per_parent_values",
            "max_parents_per_source_values",
            "rrf_ks",
            "rrf_weight_pairs",
            "parent_support_lambdas",
            "full_parent_max_chars_values",
            "hit_window_chars_per_side_values",
        )
        for field_name in sequence_fields:
            values = getattr(self, field_name)
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            _validate_unique(values, field_name)
        for overlap in self.parent_overlaps:
            if overlap >= min(self.parent_sizes):
                raise ValueError("every parent overlap must be below every parent size")
        for overlap in self.child_overlaps:
            if overlap >= min(self.child_sizes):
                raise ValueError("every child overlap must be below every child size")
        if max(self.child_sizes) > min(self.parent_sizes):
            raise ValueError("every child size must fit every parent size candidate")
        return self


class BenchmarkGateConfig(StrictRagConfigModel):
    """Offline eligibility gates kept separate from functional correctness."""

    recall_at_5_min_absolute_gain: UnitFloat
    recall_at_5_ci_lower_bound_min: float
    mrr_min_absolute_gain: UnitFloat
    mrr_ci_lower_bound_min: float
    high_baseline_recall_threshold: UnitFloat
    high_baseline_relative_error_reduction: UnitFloat
    high_baseline_noninferiority_margin: UnitFloat
    per_subject_recall_ci_lower_bound_min: float
    noise_at_5_max_absolute_increase: UnitFloat
    noise_at_5_ci_upper_bound_max: UnitFloat
    p95_latency_max_baseline_ratio: Annotated[float, Field(ge=1.0)]
    p95_latency_absolute_budget_ms: PositiveFloat
    parent_context_max_baseline_ratio: Annotated[float, Field(ge=1.0)]
    answer_correctness_noninferiority_margin: UnitFloat
    citation_support_noninferiority_margin: UnitFloat
    hallucination_max_absolute_increase: UnitFloat

    @model_validator(mode="after")
    def _validate_confidence_bounds(self) -> "BenchmarkGateConfig":
        if self.recall_at_5_ci_lower_bound_min < 0.0:
            raise ValueError("recall_at_5_ci_lower_bound_min must be non-negative")
        if self.mrr_ci_lower_bound_min < 0.0:
            raise ValueError("mrr_ci_lower_bound_min must be non-negative")
        if not -1.0 <= self.per_subject_recall_ci_lower_bound_min <= 1.0:
            raise ValueError(
                "per_subject_recall_ci_lower_bound_min must be between -1 and 1"
            )
        return self


class RagBenchmarkConfig(StrictRagConfigModel):
    """Complete benchmark, gold-data, and readiness-audit contract."""

    schema_version: NonBlankStr
    dataset_schema_version: NonBlankStr
    report_schema_version: NonBlankStr
    primary_subjects: NonBlankStrTuple
    min_global_gold_queries: PositiveInt
    min_subject_gold_queries: PositiveInt
    min_independent_sources: PositiveInt
    low_text_page_chars: PositiveInt
    bootstrap_samples: PositiveInt
    bootstrap_confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    bootstrap_seed: NonNegativeInt
    top_ks: PositiveIntTuple
    parent_top_ks: PositiveIntTuple
    human_gold_paths: ConfigPathTuple
    historical_annotated_paths: ConfigPathTuple
    synthetic_smoke_paths: ConfigPathTuple
    synthetic_smoke_eligible_for_rollout: Literal[False]
    source_group_manifest_path: ConfigPath
    candidate_grid: BenchmarkCandidateGrid
    gates: BenchmarkGateConfig

    @field_validator("primary_subjects")
    @classmethod
    def _validate_primary_subjects(cls, subjects: tuple[str, ...]) -> tuple[str, ...]:
        if not subjects:
            raise ValueError("primary_subjects must not be empty")
        _validate_unique(subjects, "primary_subjects")
        for subject in subjects:
            if subject != subject.casefold():
                raise ValueError("primary_subjects must already be case-folded")
            if (
                subject.startswith("_")
                or subject.endswith("_")
                or "__" in subject
                or not all(
                    character.isalnum() or character == "_" for character in subject
                )
            ):
                raise ValueError("primary_subjects must contain normalized identifiers")
        return subjects

    @model_validator(mode="after")
    def _validate_dataset_inventory(self) -> "RagBenchmarkConfig":
        for field_name in ("top_ks", "parent_top_ks"):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be sorted and unique")
        if 5 not in self.top_ks or 5 not in self.parent_top_ks:
            raise ValueError("top_ks and parent_top_ks must include gate k=5")
        dataset_fields = (
            "human_gold_paths",
            "historical_annotated_paths",
            "synthetic_smoke_paths",
        )
        all_paths: list[Path] = []
        for field_name in dataset_fields:
            paths = getattr(self, field_name)
            if not paths:
                raise ValueError(
                    f"{field_name} must contain at least one explicit path"
                )
            _validate_unique(paths, field_name)
            all_paths.extend(paths)
        if len(all_paths) != len(set(all_paths)):
            raise ValueError("dataset paths must not be shared across dataset classes")
        if self.source_group_manifest_path in set(all_paths):
            raise ValueError(
                "source_group_manifest_path must be separate from query datasets"
            )
        return self


def load_rag_benchmark_config(config_path: Path) -> RagBenchmarkConfig:
    """Load a required RAG benchmark YAML file."""
    return load_strict_rag_yaml(config_path, RagBenchmarkConfig)


__all__ = [
    "BenchmarkCandidateGrid",
    "BenchmarkGateConfig",
    "RagBenchmarkConfig",
    "RrfWeightPair",
    "load_rag_benchmark_config",
]
