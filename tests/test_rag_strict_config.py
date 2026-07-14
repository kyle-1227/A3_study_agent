"""Focused tests for strict production RAG configuration contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config._rag_config import (
    RagConfigPathError,
    RagConfigSecretError,
    RagConfigValidationError,
    RagConfigYamlRootError,
    resolve_required_secret,
)
from src.config.rag_benchmark_config import (
    BenchmarkCandidateGrid,
    BenchmarkGateConfig,
    RagBenchmarkConfig,
    load_rag_benchmark_config,
)
from src.config.rag_index_config import (
    CatalogConfig,
    ChunkPolicyConfig,
    RagIndexConfig,
    compute_chunk_policy_id,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.config.rag_rollout_config import (
    RagRolloutConfig,
    load_rag_rollout_config,
)
from src.rag.parent_child.config_adapter import (
    PolicyAdapterError,
    resolve_subject_chunk_policy,
)
from src.rag.parent_child.builder import compute_embedding_fingerprint
from src.rag.parent_child.ids import make_loader_policy_fingerprint
from src.rag.parent_child.runtime_loader import compute_reranker_fingerprint


def _write_yaml(path: Path, payload: object) -> Path:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _retry_payload() -> dict[str, object]:
    return {
        "max_attempts": 2,
        "initial_backoff_seconds": 0.1,
        "max_backoff_seconds": 1.0,
        "multiplier": 2.0,
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
            "header_top_lines": 3,
            "footer_bottom_lines": 3,
            "repeated_line_min_pages": 3,
            "repeated_line_min_ratio": 0.6,
            "collapse_blank_lines": True,
            "paragraph_deduplication": False,
        },
        "structure": {
            "detector_version": "structure_detector_v1",
            "pattern_set_version": "patterns_v1",
            "merge_version": "merge_v1",
            "short_unit_chars": 300,
            "major_boundary_levels": [1, 2],
        },
        "atomic_blocks": {
            "policy_version": "atomic_v1",
            "protected_types": ["code", "table", "formula", "list"],
            "hard_max_chars": 4000,
        },
        "parent": {
            "algorithm_version": "span_recursive_v1",
            "size": 2400,
            "overlap": 200,
            "separators": ["\n\n", "\n", "。", " "],
            "length_policy": "unicode_codepoints",
            "whitespace_policy": "preserve",
        },
        "child": {
            "algorithm_version": "span_recursive_v1",
            "size": 600,
            "overlap": 100,
            "separators": ["\n\n", "\n", "。", " "],
            "length_policy": "unicode_codepoints",
            "whitespace_policy": "preserve",
        },
        "metadata_contract_version": "parent_child_metadata_v1",
    }


def _index_payload(tmp_path: Path) -> dict[str, object]:
    policy_payload = _chunk_policy_payload()
    policy_id = compute_chunk_policy_id(
        ChunkPolicyConfig.model_validate(policy_payload)
    )
    return {
        "schema_version": "rag_index_config_v1",
        "catalog": {
            "data_root": str(tmp_path / "data"),
            "supported_extensions": [".pdf", ".md", ".txt"],
            "excluded_exact_names": ["ignored"],
            "excluded_prefixes": ["tmp_"],
            "exclude_hidden": True,
            "exclude_cache_directories": True,
            "cache_directory_names": ["__pycache__", ".cache"],
            "exclude_unclassified": True,
            "unclassified_directory_name": "unclassified",
            "exclude_needs_ocr": True,
            "needs_ocr_directory_name": "_needs_ocr",
            "normalization_version": "subject_id_v1",
            "symlink_policy": "reject",
        },
        "storage": {
            "index_root": str(tmp_path / "indexes"),
            "registry_path": "generation_registry.sqlite",
            "collection_name": "a3_children",
            "parent_store_schema_version": "parent_store_v1",
            "registry_schema_version": "generation_registry_v1",
            "owner_marker_schema_version": "generation_owner_v1",
            "registry_busy_timeout_seconds": 5.0,
            "parent_store_busy_timeout_seconds": 5.0,
            "retention_generations": 3,
        },
        "embedding": {
            "provider": "vendor_from_config",
            "protocol": "openai_embeddings_v1",
            "model": "embedding_model_from_config",
            "response_model": "embedding_model_from_config",
            "base_url": "https://embedding.invalid/v1",
            "endpoint_path": "/embeddings",
            "api_key_env": "EMBEDDING_KEY_ENV",
            "timeout_seconds": 30.0,
            "retry": _retry_payload(),
            "batch_size": 32,
            "max_in_flight_batches": 1,
            "expected_dimension": 1024,
            "distance_metric": "cosine",
            "normalization_contract": "unit_length_v1",
            "document_input_type": "document",
            "query_input_type": "query",
            "input_type_field": "input_type",
            "provider_routing": None,
        },
        "reranker": {
            "provider": "another_vendor_from_config",
            "model": "reranker_model_from_config",
            "response_model": "reranker_model_from_config",
            "base_url": "https://reranker.invalid/v1",
            "endpoint_path": "/rerank",
            "api_key_env": "RERANKER_KEY_ENV",
            "timeout_seconds": 20.0,
            "retry": _retry_payload(),
            "batch_size": 40,
            "protocol": "ranked_index_scores_v1",
            "score_min": 0.0,
            "score_max": 1.0,
            "provider_routing": None,
        },
        "bm25": {
            "tokenizer": "tokenizer_from_config",
            "tokenizer_version": "tokenizer_v1",
            "dictionary_hash": "b" * 64,
            "artifact_format": "jsonl",
        },
        "chunk_policies": {policy_id: policy_payload},
        "subject_policy_map": {"math": policy_id},
        "retrieval": {
            "vector_top_k": 40,
            "bm25_top_k": 40,
            "rrf_k": 60,
            "vector_weight": 1.0,
            "bm25_weight": 1.0,
            "reranker_top_n": 20,
            "unique_parent_top_k": 5,
            "max_children_per_parent": 2,
            "max_parents_per_source": 2,
            "parent_support_lambda": 0.25,
            "full_parent_max_chars": 2400,
            "hit_window_chars_per_side": 800,
            "context_item_max_chars": 4000,
            "judge_preview_max_chars": 800,
            "multi_subject_per_subject_top_k": 3,
            "multi_subject_max_parents": 8,
            "cross_branch_rrf_k": 60,
            "subject_coverage_quota": 1,
        },
    }


@pytest.mark.parametrize("max_in_flight_batches", (0, 5))
def test_embedding_config_requires_bounded_explicit_concurrency(
    tmp_path: Path,
    max_in_flight_batches: int,
) -> None:
    payload = _index_payload(tmp_path)
    embedding = payload["embedding"]
    assert isinstance(embedding, dict)
    embedding["max_in_flight_batches"] = max_in_flight_batches

    with pytest.raises(ValidationError, match="max_in_flight_batches"):
        RagIndexConfig.model_validate(payload)


def _candidate_grid_payload() -> dict[str, object]:
    return {
        "parent_sizes": [1600, 2400, 3200],
        "parent_overlaps": [0, 200, 300],
        "child_sizes": [400, 600, 800],
        "child_overlaps": [60, 100, 120],
        "vector_top_ks": [20, 40, 80],
        "bm25_top_ks": [20, 40, 80],
        "reranker_top_ns": [10, 20, 40],
        "unique_parent_top_ks": [3, 5, 8],
        "max_children_per_parent_values": [1, 2, 3],
        "max_parents_per_source_values": [1, 2, 3],
        "rrf_ks": [20, 60],
        "rrf_weight_pairs": [
            {"vector_weight": 1.0, "bm25_weight": 1.0},
            {"vector_weight": 2.0, "bm25_weight": 1.0},
            {"vector_weight": 1.0, "bm25_weight": 2.0},
        ],
        "parent_support_lambdas": [0.0, 0.25, 0.5],
        "full_parent_max_chars_values": [1600, 2400, 3200],
        "hit_window_chars_per_side_values": [400, 800, 1200],
    }


def _gate_payload() -> dict[str, object]:
    return {
        "recall_at_5_min_absolute_gain": 0.05,
        "recall_at_5_ci_lower_bound_min": 0.000001,
        "mrr_min_absolute_gain": 0.03,
        "mrr_ci_lower_bound_min": 0.000001,
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
    }


def _benchmark_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": "rag_benchmark_config_v1",
        "dataset_schema_version": "rag_gold_v1",
        "report_schema_version": "rag_benchmark_report_v1",
        "primary_subjects": [
            "big_data",
            "computer",
            "machine_learning",
            "math",
            "python",
        ],
        "min_global_gold_queries": 50,
        "min_subject_gold_queries": 20,
        "min_independent_sources": 3,
        "low_text_page_chars": 100,
        "bootstrap_samples": 10000,
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": 20260710,
        "top_ks": [1, 3, 5, 10],
        "parent_top_ks": [1, 3, 5],
        "human_gold_paths": [str(tmp_path / "human.jsonl")],
        "historical_annotated_paths": [str(tmp_path / "historical.jsonl")],
        "synthetic_smoke_paths": [str(tmp_path / "smoke.jsonl")],
        "synthetic_smoke_eligible_for_rollout": False,
        "source_group_manifest_path": str(tmp_path / "source_groups.json"),
        "candidate_grid": _candidate_grid_payload(),
        "gates": _gate_payload(),
    }


def _rollout_payload() -> dict[str, object]:
    subjects = ["math", "python"]
    stage_rows = [
        ("shadow", 0.0, 0, 24.0, False, subjects),
        ("internal", 100.0, 0, 24.0, True, subjects),
        ("single_subject", 5.0, 200, 24.0, False, ["math"]),
        ("multi_subject", 10.0, 500, 24.0, False, subjects),
        ("expand_1", 25.0, 1000, 48.0, False, subjects),
        ("expand_2", 50.0, 2000, 72.0, False, subjects),
        ("full", 100.0, 2000, 72.0, False, subjects),
    ]
    stages = [
        {
            "stage_id": stage_id,
            "candidate_allocation_percent": allocation,
            "min_evaluable_requests": requests,
            "min_observation_hours": hours,
            "eligible_subjects": eligible_subjects,
            "internal_only": internal_only,
        }
        for stage_id, allocation, requests, hours, internal_only, eligible_subjects in stage_rows
    ]
    return {
        "schema_version": "rag_rollout_config_v1",
        "activation_enabled": False,
        "shadow_enabled": False,
        "primary_subjects": subjects,
        "request_hash_algorithm": "sha256_v1",
        "candidate_failure_policy": "fail_fast",
        "rollback_mode": "explicit_registry_activation",
        "benchmark_eligibility_required": True,
        "stages": stages,
        "stop_conditions": {
            "max_candidate_error_rate": 0.01,
            "max_p95_latency_baseline_ratio": 1.25,
            "max_context_token_baseline_ratio": 1.35,
            "max_recall_at_5_absolute_regression": 0.02,
            "max_answer_correctness_absolute_regression": 0.02,
            "max_citation_support_absolute_regression": 0.02,
            "max_hallucination_absolute_increase": 0.01,
            "max_integrity_failures": 0,
            "max_generation_mismatches": 0,
            "max_parent_hydration_failures": 0,
        },
    }


def test_index_loader_accepts_provider_neutral_explicit_config(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "index.yaml", _index_payload(tmp_path))

    config = load_rag_index_config(config_path)

    assert config.embedding.provider == "vendor_from_config"
    assert config.reranker.provider == "another_vendor_from_config"
    assert config.catalog.data_root == tmp_path / "data"
    assert (
        config.storage.resolved_registry_path()
        == (tmp_path / "indexes" / "generation_registry.sqlite").resolve()
    )


def test_openrouter_provider_routing_is_strict_and_changes_fingerprints(
    tmp_path: Path,
) -> None:
    first_payload = _index_payload(tmp_path)
    first_embedding = first_payload["embedding"]
    first_reranker = first_payload["reranker"]
    assert isinstance(first_embedding, dict)
    assert isinstance(first_reranker, dict)
    first_embedding.update(
        {
            "provider": "openrouter",
            "protocol": "openrouter_embeddings_v1",
            "input_type_field": None,
            "provider_routing": {
                "order": ["parasail"],
                "allow_fallbacks": False,
            },
        }
    )
    first_reranker.update(
        {
            "provider": "openrouter",
            "protocol": "openrouter_ranked_index_scores_v1",
            "provider_routing": {
                "order": ["nvidia"],
                "allow_fallbacks": False,
            },
        }
    )
    first = RagIndexConfig.model_validate(first_payload)

    second_payload = _index_payload(tmp_path)
    second_embedding = second_payload["embedding"]
    second_reranker = second_payload["reranker"]
    assert isinstance(second_embedding, dict)
    assert isinstance(second_reranker, dict)
    second_embedding.update(
        {
            "provider": "openrouter",
            "protocol": "openrouter_embeddings_v1",
            "input_type_field": None,
            "provider_routing": {
                "order": ["nvidia"],
                "allow_fallbacks": False,
            },
        }
    )
    second_reranker.update(
        {
            "provider": "openrouter",
            "protocol": "openrouter_ranked_index_scores_v1",
            "provider_routing": {
                "order": ["parasail"],
                "allow_fallbacks": False,
            },
        }
    )
    second = RagIndexConfig.model_validate(second_payload)

    assert first.embedding.provider_routing is not None
    assert first.embedding.provider_routing.order == ("parasail",)
    assert first.embedding.provider_routing.allow_fallbacks is False
    assert compute_embedding_fingerprint(first) != compute_embedding_fingerprint(second)
    assert compute_reranker_fingerprint(first) != compute_reranker_fingerprint(second)

    missing_routing = _index_payload(tmp_path)
    missing_embedding = missing_routing["embedding"]
    assert isinstance(missing_embedding, dict)
    missing_embedding.update(
        {
            "provider": "openrouter",
            "protocol": "openrouter_embeddings_v1",
            "input_type_field": None,
        }
    )
    del missing_embedding["provider_routing"]
    with pytest.raises(ValidationError):
        RagIndexConfig.model_validate(missing_routing)

    invalid_order = _index_payload(tmp_path)
    invalid_embedding = invalid_order["embedding"]
    assert isinstance(invalid_embedding, dict)
    invalid_embedding.update(
        {
            "provider": "openrouter",
            "protocol": "openrouter_embeddings_v1",
            "input_type_field": None,
            "provider_routing": {
                "order": ["parasail", "nvidia"],
                "allow_fallbacks": False,
            },
        }
    )
    with pytest.raises(ValidationError):
        RagIndexConfig.model_validate(invalid_order)

    invalid_fallback = _index_payload(tmp_path)
    invalid_reranker = invalid_fallback["reranker"]
    assert isinstance(invalid_reranker, dict)
    invalid_reranker.update(
        {
            "provider": "openrouter",
            "protocol": "openrouter_ranked_index_scores_v1",
            "provider_routing": {
                "order": ["nvidia"],
                "allow_fallbacks": True,
            },
        }
    )
    with pytest.raises(ValidationError):
        RagIndexConfig.model_validate(invalid_fallback)

    non_openrouter_routing = _index_payload(tmp_path)
    non_openrouter_embedding = non_openrouter_routing["embedding"]
    assert isinstance(non_openrouter_embedding, dict)
    non_openrouter_embedding["provider_routing"] = {
        "order": ["parasail"],
        "allow_fallbacks": False,
    }
    with pytest.raises(ValidationError):
        RagIndexConfig.model_validate(non_openrouter_routing)


def test_index_loader_rejects_extra_fields_and_bad_cross_constraints(
    tmp_path: Path,
) -> None:
    payload = _index_payload(tmp_path)
    embedding = payload["embedding"]
    assert isinstance(embedding, dict)
    embedding["unexpected"] = "schema drift"
    path = _write_yaml(tmp_path / "extra.yaml", payload)
    with pytest.raises(RagConfigValidationError) as extra_error:
        load_rag_index_config(path)
    assert ("embedding.unexpected", "extra_forbidden") in (
        extra_error.value.validation_errors
    )

    payload = _index_payload(tmp_path)
    retrieval = payload["retrieval"]
    assert isinstance(retrieval, dict)
    retrieval["reranker_top_n"] = 100
    path = _write_yaml(tmp_path / "cross.yaml", payload)
    with pytest.raises(RagConfigValidationError):
        load_rag_index_config(path)


def test_benchmark_loader_requires_all_readiness_paths_and_false_smoke_gate(
    tmp_path: Path,
) -> None:
    path = _write_yaml(tmp_path / "benchmark.yaml", _benchmark_payload(tmp_path))

    config = load_rag_benchmark_config(path)

    assert config.source_group_manifest_path == tmp_path / "source_groups.json"
    assert config.min_independent_sources == 3
    assert config.synthetic_smoke_eligible_for_rollout is False

    missing = _benchmark_payload(tmp_path)
    del missing["source_group_manifest_path"]
    with pytest.raises(RagConfigValidationError) as error:
        load_rag_benchmark_config(_write_yaml(tmp_path / "missing.yaml", missing))
    assert ("source_group_manifest_path", "missing") in error.value.validation_errors

    invalid = _benchmark_payload(tmp_path)
    invalid["synthetic_smoke_eligible_for_rollout"] = True
    with pytest.raises(RagConfigValidationError):
        load_rag_benchmark_config(_write_yaml(tmp_path / "smoke.yaml", invalid))


def test_rollout_loader_keeps_activation_explicit_and_fail_fast(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "rollout.yaml", _rollout_payload())

    config = load_rag_rollout_config(path)

    assert config.activation_enabled is False
    assert config.candidate_failure_policy == "fail_fast"
    assert config.rollback_mode == "explicit_registry_activation"

    invalid = _rollout_payload()
    invalid["candidate_failure_policy"] = "switch_generation"
    with pytest.raises(RagConfigValidationError):
        load_rag_rollout_config(_write_yaml(tmp_path / "unsafe.yaml", invalid))


def test_loaders_require_path_and_yaml_mapping(tmp_path: Path) -> None:
    with pytest.raises(RagConfigPathError):
        load_rag_benchmark_config("benchmark.yaml")

    scalar_path = tmp_path / "scalar.yaml"
    scalar_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(RagConfigYamlRootError):
        load_rag_benchmark_config(scalar_path)


def test_production_models_are_strict_frozen_and_have_no_field_defaults(
    tmp_path: Path,
) -> None:
    model_types = (
        RagIndexConfig,
        CatalogConfig,
        RagBenchmarkConfig,
        BenchmarkCandidateGrid,
        BenchmarkGateConfig,
        RagRolloutConfig,
    )
    for model_type in model_types:
        assert model_type.model_config["extra"] == "forbid"
        assert model_type.model_config["strict"] is True
        assert model_type.model_config["frozen"] is True
        assert all(field.is_required() for field in model_type.model_fields.values())

    config = load_rag_benchmark_config(
        _write_yaml(tmp_path / "benchmark.yaml", _benchmark_payload(tmp_path))
    )
    with pytest.raises(ValidationError):
        config.bootstrap_seed = 7


def test_strict_models_reject_stringified_numeric_values(tmp_path: Path) -> None:
    payload = _benchmark_payload(tmp_path)
    payload["bootstrap_samples"] = "10000"
    path = _write_yaml(tmp_path / "coerced.yaml", payload)

    with pytest.raises(RagConfigValidationError) as error:
        load_rag_benchmark_config(path)

    assert ("bootstrap_samples", "int_type") in error.value.validation_errors


def test_chunk_policy_adapter_keeps_manifest_loader_and_runtime_identity_coherent(
    tmp_path: Path,
) -> None:
    config = RagIndexConfig.model_validate(_index_payload(tmp_path))

    resolved = resolve_subject_chunk_policy(config, "math")

    configured = next(iter(config.chunk_policies.values()))
    expected = compute_chunk_policy_id(configured)
    assert resolved.policy_manifest.policy_id == expected
    assert resolved.parent_child_policy.policy_id == expected
    assert (
        resolved.parent_child_policy.loader_policy_id
        == make_loader_policy_fingerprint(resolved.loader_config)
    )


def test_chunk_policy_adapter_rejects_unknown_subject_and_algorithm(
    tmp_path: Path,
) -> None:
    config = RagIndexConfig.model_validate(_index_payload(tmp_path))
    with pytest.raises(PolicyAdapterError, match="exact configured policy"):
        resolve_subject_chunk_policy(config, "unknown")

    payload = _index_payload(tmp_path)
    _, policy_payload = next(iter(payload["chunk_policies"].items()))
    assert isinstance(policy_payload, dict)
    parent = policy_payload["parent"]
    assert isinstance(parent, dict)
    parent["algorithm_version"] = "unimplemented_v9"
    changed = ChunkPolicyConfig.model_validate(policy_payload)
    changed_id = compute_chunk_policy_id(changed)
    payload["chunk_policies"] = {changed_id: policy_payload}
    payload["subject_policy_map"] = {"math": changed_id}
    changed_config = RagIndexConfig.model_validate(payload)

    with pytest.raises(PolicyAdapterError, match="parent.algorithm_version"):
        resolve_subject_chunk_policy(changed_config, "math")


def test_required_secret_resolution_fails_without_exposing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("A3_TEST_RAG_SECRET", raising=False)
    with pytest.raises(RagConfigSecretError) as missing:
        resolve_required_secret("A3_TEST_RAG_SECRET")
    assert "A3_TEST_RAG_SECRET" in str(missing.value)

    monkeypatch.setenv("A3_TEST_RAG_SECRET", "sensitive-value")
    assert resolve_required_secret("A3_TEST_RAG_SECRET") == "sensitive-value"


def test_index_paths_resolve_only_inside_explicit_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    payload = _index_payload(project_root)
    catalog = payload["catalog"]
    storage = payload["storage"]
    assert isinstance(catalog, dict) and isinstance(storage, dict)
    catalog["data_root"] = "data"
    storage["index_root"] = "indexes"
    config = RagIndexConfig.model_validate(payload)

    resolved = resolve_rag_index_config_paths(config, project_root=project_root)

    assert resolved.catalog.data_root == (project_root / "data").resolve()
    assert resolved.storage.index_root == (project_root / "indexes").resolve()

    catalog["data_root"] = "../outside"
    escaping = RagIndexConfig.model_validate(payload)
    with pytest.raises(ValueError, match="catalog.data_root"):
        resolve_rag_index_config_paths(escaping, project_root=project_root)
