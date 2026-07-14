"""Focused tests for strict local GoldDataset and readiness tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from scripts.audit_rag_readiness import main as audit_main
from scripts.audit_rag_readiness import run_audit
from scripts.prepare_rag_gold_dataset import (
    main as gold_main,
    run_export_readiness_jsonl,
    run_init,
    run_inspect_source,
    run_validate,
)
from src.config.rag_index_config import (
    ChunkPolicyConfig,
    compute_chunk_policy_id,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.gold_dataset import (
    GoldDatasetDraft,
    GoldDatasetPathError,
    load_gold_dataset,
    resolve_project_path,
    validate_gold_dataset,
    write_gold_model,
)
from src.rag.parent_child.config_adapter import resolve_subject_chunk_policy
from src.rag.parent_child.evaluation import GoldDataset, GoldEvidenceSpan, GoldQuery
from src.rag.parent_child.loader import load_cleaned_source, page_range_for_span
from src.rag.parent_child.models import SourceEntry
from src.rag.readiness import load_query_inventory, load_source_group_manifest
from src.rag.subject_catalog import SubjectCatalog


def _retry_payload() -> dict[str, object]:
    return {
        "max_attempts": 1,
        "initial_backoff_seconds": 0.1,
        "max_backoff_seconds": 0.1,
        "multiplier": 1.0,
    }


def _chunk_policy_payload() -> dict[str, object]:
    return {
        "extraction": {
            "algorithm_version": "page_extract_v1",
            "pdf_extraction_method": "configured_pdf_text",
            "text_extraction_method": "configured_utf8_text",
        },
        "page_assembly": {
            "algorithm_version": "page_assembly_v1",
            "page_separator": "\n\f\n",
        },
        "cleaning": {
            "algorithm_version": "page_clean_v2",
            "nul_character_policy": "replace_with_space_v1",
            "normalize_newlines": True,
            "strip_trailing_whitespace": True,
            "strip_outer_blank_lines": True,
            "header_top_lines": 2,
            "footer_bottom_lines": 2,
            "repeated_line_min_pages": 2,
            "repeated_line_min_ratio": 0.5,
            "collapse_blank_lines": True,
            "paragraph_deduplication": False,
        },
        "structure": {
            "detector_version": "structure_detector_v1",
            "pattern_set_version": "patterns_v1",
            "merge_version": "merge_v1",
            "short_unit_chars": 20,
            "major_boundary_levels": [1, 2],
        },
        "atomic_blocks": {
            "policy_version": "atomic_v1",
            "protected_types": ["code", "table", "formula", "list"],
            "hard_max_chars": 200,
        },
        "parent": {
            "algorithm_version": "span_recursive_v1",
            "size": 100,
            "overlap": 0,
            "separators": ["\n\n", "\n", " "],
            "length_policy": "unicode_codepoints",
            "whitespace_policy": "preserve",
        },
        "child": {
            "algorithm_version": "span_recursive_v1",
            "size": 50,
            "overlap": 0,
            "separators": ["\n\n", "\n", " "],
            "length_policy": "unicode_codepoints",
            "whitespace_policy": "preserve",
        },
        "metadata_contract_version": "parent_child_metadata_v1",
    }


def _index_payload() -> dict[str, object]:
    policy = _chunk_policy_payload()
    policy_id = compute_chunk_policy_id(ChunkPolicyConfig.model_validate(policy))
    return {
        "schema_version": "rag_index_config_v1",
        "catalog": {
            "data_root": "data",
            "supported_extensions": [".md"],
            "excluded_exact_names": ["evaluation", "ignored"],
            "excluded_prefixes": ["tmp_"],
            "exclude_hidden": True,
            "exclude_cache_directories": True,
            "cache_directory_names": [".cache", "__pycache__"],
            "exclude_unclassified": True,
            "unclassified_directory_name": "unclassified",
            "exclude_needs_ocr": True,
            "needs_ocr_directory_name": "_needs_ocr",
            "normalization_version": "subject_id_v1",
            "symlink_policy": "reject",
        },
        "storage": {
            "index_root": "indexes",
            "registry_path": "generation_registry.sqlite",
            "collection_name": "a3_children",
            "parent_store_schema_version": "parent_store_v1",
            "registry_schema_version": "generation_registry_v1",
            "owner_marker_schema_version": "generation_owner_v1",
            "registry_busy_timeout_seconds": 1.0,
            "parent_store_busy_timeout_seconds": 1.0,
            "retention_generations": 2,
        },
        "embedding": {
            "provider": "configured_embedding_provider",
            "protocol": "openai_embeddings_v1",
            "model": "configured_embedding_model",
            "response_model": "configured_embedding_model",
            "base_url": "https://embedding.invalid/v1",
            "endpoint_path": "/embeddings",
            "api_key_env": "GOLD_TEST_EMBEDDING_KEY",
            "timeout_seconds": 1.0,
            "retry": _retry_payload(),
            "batch_size": 2,
            "max_in_flight_batches": 1,
            "expected_dimension": 4,
            "distance_metric": "cosine",
            "normalization_contract": "unit_length_v1",
            "document_input_type": "document",
            "query_input_type": "query",
            "input_type_field": "input_type",
            "provider_routing": None,
        },
        "reranker": {
            "provider": "configured_reranker_provider",
            "model": "configured_reranker_model",
            "response_model": "configured_reranker_model",
            "base_url": "https://reranker.invalid/v1",
            "endpoint_path": "/rerank",
            "api_key_env": "GOLD_TEST_RERANKER_KEY",
            "timeout_seconds": 1.0,
            "retry": _retry_payload(),
            "batch_size": 4,
            "protocol": "ranked_index_scores_v1",
            "score_min": 0.0,
            "score_max": 1.0,
            "provider_routing": None,
        },
        "bm25": {
            "tokenizer": "configured_jieba",
            "tokenizer_version": "test_version",
            "dictionary_hash": "a" * 64,
            "artifact_format": "jsonl",
        },
        "chunk_policies": {policy_id: policy},
        "subject_policy_map": {"math": policy_id},
        "retrieval": {
            "vector_top_k": 2,
            "bm25_top_k": 2,
            "rrf_k": 1,
            "vector_weight": 1.0,
            "bm25_weight": 1.0,
            "reranker_top_n": 2,
            "unique_parent_top_k": 1,
            "max_children_per_parent": 1,
            "max_parents_per_source": 1,
            "parent_support_lambda": 0.0,
            "full_parent_max_chars": 100,
            "hit_window_chars_per_side": 10,
            "context_item_max_chars": 100,
            "judge_preview_max_chars": 20,
            "multi_subject_per_subject_top_k": 1,
            "multi_subject_max_parents": 1,
            "cross_branch_rrf_k": 1,
            "subject_coverage_quota": 1,
        },
    }


def _benchmark_payload(*, min_sources: int, min_queries: int) -> dict[str, object]:
    return {
        "schema_version": "rag_benchmark_config_v1",
        "dataset_schema_version": "rag_gold_v1",
        "report_schema_version": "rag_benchmark_report_v1",
        "primary_subjects": ["math"],
        "min_global_gold_queries": min_queries,
        "min_subject_gold_queries": min_queries,
        "min_independent_sources": min_sources,
        "low_text_page_chars": 1,
        "bootstrap_samples": 10,
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": 7,
        "top_ks": [1, 3, 5],
        "parent_top_ks": [1, 3, 5],
        "human_gold_paths": ["data/evaluation/human_gold.jsonl"],
        "historical_annotated_paths": ["data/evaluation/historical.jsonl"],
        "synthetic_smoke_paths": ["data/evaluation/smoke.jsonl"],
        "synthetic_smoke_eligible_for_rollout": False,
        "source_group_manifest_path": "config/source_groups.json",
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
            "p95_latency_absolute_budget_ms": 1.0,
            "parent_context_max_baseline_ratio": 1.35,
            "answer_correctness_noninferiority_margin": 0.02,
            "citation_support_noninferiority_margin": 0.02,
            "hallucination_max_absolute_increase": 0.01,
        },
    }


def _project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project = tmp_path / "project"
    notes = project / "data" / "math" / "limits.md"
    notes.parent.mkdir(parents=True)
    notes.write_text(
        "# Limits\nA limit is a precise local behavior.\n", encoding="utf-8"
    )
    config_dir = project / "config"
    config_dir.mkdir()
    index_path = config_dir / "index.yaml"
    index_path.write_text(
        yaml.safe_dump(_index_payload(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    source_groups_path = config_dir / "source_groups.json"
    source_groups_path.write_text(
        json.dumps(
            {
                "schema_version": "source_groups_v1",
                "source_groups": {"math/limits.md": "calculus_textbook"},
            }
        ),
        encoding="utf-8",
    )
    benchmark_path = config_dir / "benchmark.yaml"
    benchmark_path.write_text(
        yaml.safe_dump(
            _benchmark_payload(min_sources=1, min_queries=1),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return project, index_path, source_groups_path, benchmark_path


def _loaded_document(project: Path, index_path: Path):
    config = resolve_rag_index_config_paths(
        load_rag_index_config(index_path),
        project_root=project,
    )
    snapshot = SubjectCatalog(
        config=config.catalog,
        subject_policy_map=config.subject_policy_map,
    ).discover()
    source = snapshot.source_entries()[0]
    document = load_cleaned_source(
        SourceEntry(
            schema_version="source_entry_v1",
            source_path=source.source_path,
            data_root=snapshot.data_root,
            subject=source.subject_id,
            doc_type="markdown",
        ),
        resolve_subject_chunk_policy(config, source.subject_id).loader_config,
    )
    return config, document


def _gold_dataset(
    project: Path, index_path: Path, *, query_text: str = "What is a limit?"
) -> GoldDataset:
    _, document = _loaded_document(project, index_path)
    start = document.content.index("limit")
    end = start + len("limit")
    page_start, page_end = page_range_for_span(
        document,
        start_char=start,
        end_char=end,
    )
    return GoldDataset(
        schema_version="gold_dataset_v1",
        dataset_id="gold_local_v1",
        queries=(
            GoldQuery(
                schema_version="gold_query_v1",
                query_id="human-q1",
                subject="math",
                query=query_text,
                dataset_kind="human_gold",
                eligible_for_rollout=True,
                gold_spans=(
                    GoldEvidenceSpan(
                        schema_version="gold_evidence_span_v1",
                        gold_span_id="gold_limit",
                        source_group_id="calculus_textbook",
                        source_relpath=document.source_relpath,
                        doc_id=document.doc_id,
                        pagination_kind=document.pagination_kind,
                        page_start=page_start,
                        page_end=page_end,
                        start_char=start,
                        end_char=end,
                        section_path=("Limits",),
                        relevance_grade=3,
                    ),
                ),
            ),
        ),
    )


def _with_dataset_kind(dataset: GoldDataset, dataset_kind: str) -> GoldDataset:
    original = dataset.queries[0]
    return GoldDataset(
        schema_version="gold_dataset_v1",
        dataset_id=dataset.dataset_id,
        queries=(
            GoldQuery(
                schema_version="gold_query_v1",
                query_id=f"{dataset_kind}-q1",
                subject=original.subject,
                query=original.query,
                dataset_kind=dataset_kind,  # type: ignore[arg-type]
                eligible_for_rollout=dataset_kind != "synthetic_smoke",
                gold_spans=original.gold_spans,
            ),
        ),
    )


def test_gold_dataset_validation_proves_cleaned_span_and_rejects_stale_coordinates(
    tmp_path: Path,
) -> None:
    project, index_path, groups_path, _ = _project(tmp_path)
    config, document = _loaded_document(project, index_path)
    dataset = _gold_dataset(project, index_path)

    assert (
        validate_gold_dataset(
            dataset=dataset,
            index_config=config,
            source_groups=load_source_group_manifest(groups_path),
        )
        == dataset
    )

    span = dataset.queries[0].gold_spans[0]
    invalid = dataset.model_copy(
        update={
            "queries": (
                dataset.queries[0].model_copy(
                    update={
                        "gold_spans": (
                            span.model_copy(
                                update={
                                    "end_char": len(document.content) + 1,
                                }
                            ),
                        )
                    }
                ),
            )
        }
    )
    with pytest.raises(Exception, match="Gold span exceeds"):
        validate_gold_dataset(
            dataset=invalid,
            index_config=config,
            source_groups=load_source_group_manifest(groups_path),
        )


def test_gold_draft_validate_and_inspect_use_candidate_page_aware_loader(
    tmp_path: Path,
) -> None:
    project, index_path, groups_path, _ = _project(tmp_path)
    dataset = _gold_dataset(project, index_path)
    draft = GoldDatasetDraft(
        schema_version="gold_dataset_draft_v1",
        dataset_id=dataset.dataset_id,
        queries=dataset.queries,
    )
    draft_path = write_gold_model(
        project_root=project,
        output_path=Path("data/evaluation/draft.json"),
        model=draft,
        overwrite=False,
    )

    validated_path = run_validate(
        project_root=project,
        index_config_path=index_path.relative_to(project),
        source_groups_path=groups_path.relative_to(project),
        input_path=draft_path.relative_to(project),
        output_path=Path("data/evaluation/gold_dataset_v1.json"),
        overwrite=False,
    )
    inspection_path = run_inspect_source(
        project_root=project,
        index_config_path=index_path.relative_to(project),
        source_relpath="math/limits.md",
        output_path=Path("data/evaluation/inspection.json"),
        overwrite=False,
    )

    assert load_gold_dataset(validated_path) == dataset
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    assert inspection["doc_id"] == dataset.queries[0].gold_spans[0].doc_id
    assert inspection["pages"][0]["start_char"] == 0
    assert "A limit" in inspection["cleaned_content"]


def test_gold_dataset_exports_keep_synthetic_ineligible_and_are_derived_only(
    tmp_path: Path,
) -> None:
    project, index_path, _, _ = _project(tmp_path)
    base = _gold_dataset(project, index_path)
    human = _with_dataset_kind(base, "human_gold").queries[0]
    historical = _with_dataset_kind(base, "historical_annotated").queries[0]
    synthetic = _with_dataset_kind(base, "synthetic_smoke").queries[0]
    dataset = GoldDataset(
        schema_version="gold_dataset_v1",
        dataset_id=base.dataset_id,
        queries=(synthetic, historical, human),
    )
    dataset_path = write_gold_model(
        project_root=project,
        output_path=Path("data/evaluation/gold_dataset_v1.json"),
        model=dataset,
        overwrite=False,
    )

    human_path, historical_path, synthetic_path = run_export_readiness_jsonl(
        project_root=project,
        gold_dataset_path=dataset_path.relative_to(project),
        human_output=Path("data/evaluation/human_gold.jsonl"),
        historical_output=Path("data/evaluation/historical_annotated.jsonl"),
        synthetic_output=Path("data/evaluation/synthetic_smoke.jsonl"),
        overwrite=False,
    )

    assert [item.query_id for item in load_query_inventory(human_path)] == [
        "human_gold-q1"
    ]
    assert [item.query_id for item in load_query_inventory(historical_path)] == [
        "historical_annotated-q1"
    ]
    records = load_query_inventory(synthetic_path)
    assert len(records) == 1
    assert records[0].eligible_for_rollout is False
    with pytest.raises(ValidationError):
        GoldQuery(
            schema_version="gold_query_v1",
            query_id="synthetic-invalid",
            subject="math",
            query="Synthetic query",
            dataset_kind="synthetic_smoke",
            eligible_for_rollout=True,
            gold_spans=base.queries[0].gold_spans,
        )


def test_audit_uses_gold_dataset_has_no_query_leak_and_fail_on_blocked(
    tmp_path: Path,
) -> None:
    project, index_path, _, benchmark_path = _project(tmp_path)
    benchmark_path.write_text(
        yaml.safe_dump(
            _benchmark_payload(min_sources=2, min_queries=2),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dataset = _gold_dataset(project, index_path, query_text="secret-query-sentinel")
    dataset_path = write_gold_model(
        project_root=project,
        output_path=Path("data/evaluation/gold_dataset_v1.json"),
        model=dataset,
        overwrite=False,
    )

    artifact = run_audit(
        project_root=project,
        index_config_path=index_path.relative_to(project),
        benchmark_config_path=benchmark_path.relative_to(project),
        gold_dataset_path=dataset_path.relative_to(project),
        output_path=Path("reports/readiness.json"),
        overwrite=False,
    )
    output = project / "reports" / "readiness.json"
    assert artifact.report.production_recommendation_blocked is True
    assert "secret-query-sentinel" not in output.read_text(encoding="utf-8")
    assert (
        audit_main(
            [
                "--project-root",
                str(project),
                "--index-config",
                str(index_path.relative_to(project)),
                "--benchmark-config",
                str(benchmark_path.relative_to(project)),
                "--gold-dataset",
                str(dataset_path.relative_to(project)),
                "--output",
                "reports/readiness-second.json",
                "--fail-on-blocked",
            ]
        )
        == 1
    )


def test_gold_cli_rejects_path_escape_symlink_and_missing_required_arguments(
    tmp_path: Path,
) -> None:
    project, _, _, _ = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = project / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {type(exc).__name__}")

    with pytest.raises(GoldDatasetPathError, match="symlink"):
        resolve_project_path(
            project_root=project,
            value=Path("linked/output.json"),
            must_exist=False,
        )
    assert (
        gold_main(
            [
                "init",
                "--project-root",
                str(project),
                "--dataset-id",
                "gold_local_v1",
                "--output",
                "../outside/gold.json",
            ]
        )
        == 2
    )
    with pytest.raises(SystemExit) as missing:
        gold_main(["init"])
    assert missing.value.code == 2


def test_gold_init_refuses_implicit_overwrite(tmp_path: Path) -> None:
    project, _, _, _ = _project(tmp_path)
    first = run_init(
        project_root=project,
        dataset_id="gold_local_v1",
        output_path=Path("data/evaluation/draft.json"),
        overwrite=False,
    )
    assert first.is_file()
    with pytest.raises(FileExistsError):
        run_init(
            project_root=project,
            dataset_id="gold_local_v1",
            output_path=Path("data/evaluation/draft.json"),
            overwrite=False,
        )
