from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.rag.readiness import (
    QueryInventoryRecord,
    ReadinessAuditError,
    SourceGroupManifest,
    audit_rag_readiness,
    load_query_inventory,
)


def _manifest(mapping: dict[str, str]) -> SourceGroupManifest:
    return SourceGroupManifest(
        schema_version="source_groups_v1",
        source_groups=mapping,
    )


def test_synthetic_query_cannot_be_rollout_eligible() -> None:
    with pytest.raises(ValidationError):
        QueryInventoryRecord(
            query_id="q1",
            subject="math",
            query="What is a limit?",
            dataset_kind="synthetic_smoke",
            eligible_for_rollout=True,
        )


def test_query_inventory_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "subject": "math",
                "query": "Q",
                "dataset_kind": "human_gold",
                "eligible_for_rollout": True,
                "unexpected": "drift",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReadinessAuditError, match="line 1"):
        load_query_inventory(path)


def test_audit_blocks_missing_groups_and_gold(tmp_path: Path) -> None:
    data = tmp_path / "data"
    subject_dir = data / "math"
    subject_dir.mkdir(parents=True)
    (subject_dir / "notes.txt").write_text("calculus notes", encoding="utf-8")

    report = audit_rag_readiness(
        data_root=data,
        primary_subjects=("math",),
        supported_extensions=(".pdf", ".md", ".txt"),
        source_group_manifest=_manifest({}),
        query_records=(),
        low_text_page_chars=5,
        minimum_independent_sources=3,
        minimum_subject_gold_queries=20,
        minimum_global_gold_queries=50,
    )

    subject = report.subjects[0]
    assert report.production_recommendation_blocked is True
    assert subject.independent_source_count is None
    assert "source_group_manifest_incomplete" in subject.blockers
    assert subject.recommended_additional_gold_queries == 20


def test_audit_passes_explicit_source_and_gold_gates(tmp_path: Path) -> None:
    data = tmp_path / "data"
    subject_dir = data / "math"
    subject_dir.mkdir(parents=True)
    mapping: dict[str, str] = {}
    for index in range(3):
        path = subject_dir / f"source-{index}.txt"
        path.write_text(f"calculus source {index}", encoding="utf-8")
        mapping[f"math/source-{index}.txt"] = f"group-{index}"
    queries = tuple(
        QueryInventoryRecord(
            query_id=f"q-{index}",
            subject="math",
            query=f"Question {index}",
            dataset_kind="human_gold",
            eligible_for_rollout=True,
        )
        for index in range(2)
    )

    report = audit_rag_readiness(
        data_root=data,
        primary_subjects=("math",),
        supported_extensions=(".txt",),
        source_group_manifest=_manifest(mapping),
        query_records=queries,
        low_text_page_chars=5,
        minimum_independent_sources=3,
        minimum_subject_gold_queries=2,
        minimum_global_gold_queries=2,
    )

    assert report.production_recommendation_blocked is False
    assert report.subjects[0].blockers == ()


def test_readiness_script_keeps_missing_datasets_as_visible_blockers(
    tmp_path: Path,
) -> None:
    from scripts.audit_rag_readiness import run_audit

    project = tmp_path / "project"
    data = project / "data" / "math"
    config_dir = project / "config"
    data.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    (data / "notes.txt").write_text("limits", encoding="utf-8")
    (config_dir / "groups.json").write_text(
        json.dumps(
            {
                "schema_version": "source_groups_v1",
                "source_groups": {"math/notes.txt": "notes"},
            }
        ),
        encoding="utf-8",
    )
    benchmark = {
        "schema_version": "rag_benchmark_config_v1",
        "dataset_schema_version": "rag_gold_v1",
        "report_schema_version": "rag_benchmark_report_v1",
        "primary_subjects": ["math"],
        "min_global_gold_queries": 1,
        "min_subject_gold_queries": 1,
        "min_independent_sources": 1,
        "low_text_page_chars": 1,
        "bootstrap_samples": 100,
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": 7,
        "top_ks": [1, 3, 5, 10],
        "parent_top_ks": [1, 3, 5],
        "human_gold_paths": ["data/evaluation/human.jsonl"],
        "historical_annotated_paths": ["data/evaluation/history.jsonl"],
        "synthetic_smoke_paths": ["data/evaluation/smoke.jsonl"],
        "synthetic_smoke_eligible_for_rollout": False,
        "source_group_manifest_path": "config/groups.json",
        "candidate_grid": {
            "parent_sizes": [100],
            "parent_overlaps": [0],
            "child_sizes": [50],
            "child_overlaps": [0],
            "vector_top_ks": [1],
            "bm25_top_ks": [1],
            "reranker_top_ns": [1],
            "unique_parent_top_ks": [1],
            "max_children_per_parent_values": [1],
            "max_parents_per_source_values": [1],
            "rrf_ks": [1],
            "rrf_weight_pairs": [{"vector_weight": 1.0, "bm25_weight": 1.0}],
            "parent_support_lambdas": [0.0],
            "full_parent_max_chars_values": [100],
            "hit_window_chars_per_side_values": [10],
        },
        "gates": {
            "recall_at_5_min_absolute_gain": 0.05,
            "recall_at_5_ci_lower_bound_min": 0.0,
            "mrr_min_absolute_gain": 0.03,
            "mrr_ci_lower_bound_min": 0.0,
            "high_baseline_recall_threshold": 0.9,
            "high_baseline_relative_error_reduction": 0.2,
            "high_baseline_noninferiority_margin": 0.01,
            "per_subject_recall_ci_lower_bound_min": -0.02,
            "noise_at_5_max_absolute_increase": 0.02,
            "noise_at_5_ci_upper_bound_max": 0.03,
            "p95_latency_max_baseline_ratio": 1.25,
            "p95_latency_absolute_budget_ms": 3000.0,
            "parent_context_max_baseline_ratio": 1.35,
            "answer_correctness_noninferiority_margin": 0.02,
            "citation_support_noninferiority_margin": 0.02,
            "hallucination_max_absolute_increase": 0.01,
        },
    }
    (config_dir / "benchmark.yaml").write_text(
        __import__("yaml").safe_dump(benchmark, sort_keys=False), encoding="utf-8"
    )

    artifact = run_audit(
        project_root=project,
        benchmark_config_path=Path("config/benchmark.yaml"),
        data_root=Path("data"),
        output_path=Path("reports/readiness.json"),
    )

    assert artifact.report.production_recommendation_blocked is True
    assert len(artifact.missing_dataset_paths) == 3
    assert (project / "reports" / "readiness.json").is_file()
