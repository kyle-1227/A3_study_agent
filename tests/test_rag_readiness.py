from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.rag_index_config import CatalogConfig
from src.rag.readiness import (
    QueryInventoryRecord,
    ReadinessAuditError,
    SourceGroupManifest,
    audit_rag_readiness,
    load_query_inventory,
)
from src.rag.subject_catalog import SubjectCatalog


def _manifest(mapping: dict[str, str]) -> SourceGroupManifest:
    return SourceGroupManifest(
        schema_version="source_groups_v1",
        source_groups=mapping,
    )


def _catalog(data_root: Path):
    config = CatalogConfig(
        data_root=data_root,
        supported_extensions=(".txt",),
        excluded_exact_names=("ignored",),
        excluded_prefixes=("tmp_",),
        exclude_hidden=True,
        exclude_cache_directories=True,
        cache_directory_names=(".cache", "__pycache__"),
        exclude_unclassified=True,
        unclassified_directory_name="unclassified",
        exclude_needs_ocr=True,
        needs_ocr_directory_name="_needs_ocr",
        normalization_version="subject_id_v1",
        symlink_policy="reject",
    )
    return SubjectCatalog(
        config=config,
        subject_policy_map={"math": "a" * 64},
    ).discover()


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


def test_audit_blocks_missing_groups_and_gold_from_catalog_snapshot(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    subject_dir = data / "math"
    subject_dir.mkdir(parents=True)
    (subject_dir / "notes.txt").write_text("calculus notes", encoding="utf-8")

    report = audit_rag_readiness(
        catalog_snapshot=_catalog(data),
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
        catalog_snapshot=_catalog(data),
        source_group_manifest=_manifest(mapping),
        query_records=queries,
        low_text_page_chars=5,
        minimum_independent_sources=3,
        minimum_subject_gold_queries=2,
        minimum_global_gold_queries=2,
    )

    assert report.production_recommendation_blocked is False
    assert report.subjects[0].blockers == ()


def test_audit_rejects_query_subject_outside_catalog(tmp_path: Path) -> None:
    data = tmp_path / "data"
    subject_dir = data / "math"
    subject_dir.mkdir(parents=True)
    (subject_dir / "notes.txt").write_text("calculus notes", encoding="utf-8")

    with pytest.raises(ReadinessAuditError, match="unknown_subject"):
        audit_rag_readiness(
            catalog_snapshot=_catalog(data),
            source_group_manifest=_manifest({"math/notes.txt": "one"}),
            query_records=(
                QueryInventoryRecord(
                    query_id="other",
                    subject="other",
                    query="Q",
                    dataset_kind="human_gold",
                    eligible_for_rollout=True,
                ),
            ),
            low_text_page_chars=5,
            minimum_independent_sources=1,
            minimum_subject_gold_queries=1,
            minimum_global_gold_queries=1,
        )
