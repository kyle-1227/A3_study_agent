from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.config.rag_index_config import (
    ChunkPolicyConfig,
    RagIndexConfig,
    compute_chunk_policy_id,
)
from src.rag.parent_child.builder import (
    GenerationBuildError,
    GenerationBuildRequest,
    GenerationBuilder,
)
from src.rag.parent_child.generation import validate_sealed_generation
from src.rag.parent_child.registry import (
    GenerationRegistry,
    create_generation_registry,
)
from src.rag.parent_child.retrieval import (
    HybridRetrievalRequest,
    RerankCandidate,
    RerankScore,
)
from src.rag.parent_child.runtime_loader import load_generation_runtime


class _Embedding:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), float(sum(map(ord, text)) % 97), 1.0] for text in texts
        ]

    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 1.0, 1.0]


class _Reranker:
    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankScore, ...]:
        assert query
        return tuple(
            RerankScore(
                schema_version="rerank_score_v1",
                child_id=candidate.child_id,
                score=max(0.0, 1.0 - index * 0.1),
            )
            for index, candidate in enumerate(candidates)
        )


def _chunk_policy() -> dict[str, object]:
    return {
        "extraction": {
            "algorithm_version": "page_extract_v1",
            "pdf_extraction_method": "pymupdf_text_v1",
            "text_extraction_method": "utf8_text_v1",
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
            "header_top_lines": 1,
            "footer_bottom_lines": 1,
            "repeated_line_min_pages": 2,
            "repeated_line_min_ratio": 0.8,
            "collapse_blank_lines": True,
            "paragraph_deduplication": False,
        },
        "structure": {
            "detector_version": "structure_detector_v1",
            "pattern_set_version": "patterns_v1",
            "merge_version": "short_unit_merge_v1",
            "short_unit_chars": 50,
            "major_boundary_levels": [1, 2],
        },
        "atomic_blocks": {
            "policy_version": "atomic_v1",
            "protected_types": ["code", "table", "formula", "list"],
            "hard_max_chars": 500,
        },
        "parent": {
            "algorithm_version": "span_recursive_v1",
            "size": 200,
            "overlap": 20,
            "separators": ["\n\n", "\n", "。", " "],
            "length_policy": "unicode_codepoints",
            "whitespace_policy": "preserve",
        },
        "child": {
            "algorithm_version": "span_recursive_v1",
            "size": 100,
            "overlap": 10,
            "separators": ["\n\n", "\n", "。", " "],
            "length_policy": "unicode_codepoints",
            "whitespace_policy": "preserve",
        },
        "metadata_contract_version": "parent_child_metadata_v1",
    }


def _config(tmp_path: Path) -> RagIndexConfig:
    data_root = (tmp_path / "data").resolve()
    index_root = (tmp_path / "indexes").resolve()
    policy_payload = _chunk_policy()
    policy_id = compute_chunk_policy_id(
        ChunkPolicyConfig.model_validate(policy_payload)
    )
    return RagIndexConfig.model_validate(
        {
            "schema_version": "rag_index_config_v1",
            "catalog": {
                "data_root": data_root,
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
                "index_root": index_root,
                "registry_path": "generation_registry.sqlite",
                "collection_name": "children-v1",
                "parent_store_schema_version": "parent_store_v1",
                "registry_schema_version": "generation_registry_v1",
                "owner_marker_schema_version": "generation_owner_v1",
                "registry_busy_timeout_seconds": 2.0,
                "parent_store_busy_timeout_seconds": 2.0,
                "retention_generations": 3,
            },
            "embedding": {
                "provider": "configured-test-provider",
                "protocol": "openai_embeddings_v1",
                "model": "configured-test-model",
                "response_model": "configured-test-model",
                "base_url": "https://provider.invalid/v1",
                "endpoint_path": "/embeddings",
                "api_key_env": "TEST_EMBEDDING_KEY",
                "timeout_seconds": 5.0,
                "retry": {
                    "max_attempts": 2,
                    "initial_backoff_seconds": 0.1,
                    "max_backoff_seconds": 1.0,
                    "multiplier": 2.0,
                },
                "batch_size": 8,
                "max_in_flight_batches": 1,
                "expected_dimension": 3,
                "distance_metric": "cosine",
                "normalization_contract": "none_v1",
                "document_input_type": "document",
                "query_input_type": "query",
                "input_type_field": None,
                "provider_routing": None,
            },
            "reranker": {
                "provider": "configured-test-provider",
                "model": "configured-reranker",
                "response_model": "configured-reranker",
                "base_url": "https://provider.invalid/v1",
                "endpoint_path": "/rerank",
                "api_key_env": "TEST_RERANKER_KEY",
                "timeout_seconds": 5.0,
                "retry": {
                    "max_attempts": 2,
                    "initial_backoff_seconds": 0.1,
                    "max_backoff_seconds": 1.0,
                    "multiplier": 2.0,
                },
                "batch_size": 10,
                "protocol": "ranked_index_scores_v1",
                "score_min": 0.0,
                "score_max": 1.0,
                "provider_routing": None,
            },
            "bm25": {
                "tokenizer": "whitespace-test",
                "tokenizer_version": "v1",
                "dictionary_hash": "b" * 64,
                "artifact_format": "jsonl",
            },
            "chunk_policies": {policy_id: policy_payload},
            "subject_policy_map": {"math": policy_id},
            "retrieval": {
                "vector_top_k": 2,
                "bm25_top_k": 2,
                "rrf_k": 20,
                "vector_weight": 1.0,
                "bm25_weight": 1.0,
                "reranker_transport_fallback_mode": "disabled",
                "reranker_top_n": 4,
                "unique_parent_top_k": 2,
                "max_children_per_parent": 2,
                "max_parents_per_source": 2,
                "parent_support_lambda": 0.25,
                "full_parent_max_chars": 200,
                "hit_window_chars_per_side": 50,
                "context_item_max_chars": 400,
                "judge_preview_max_chars": 200,
                "multi_subject_per_subject_top_k": 1,
                "multi_subject_max_parents": 2,
                "cross_branch_rrf_k": 20,
                "subject_coverage_quota": 1,
            },
        }
    )


def _open_registry(config: RagIndexConfig) -> GenerationRegistry:
    config.storage.index_root.mkdir(parents=True, exist_ok=True)
    path = create_generation_registry(
        config.storage.resolved_registry_path(),
        schema_version=config.storage.registry_schema_version,
        busy_timeout_seconds=config.storage.registry_busy_timeout_seconds,
    )
    return GenerationRegistry.open(
        path,
        index_root=config.storage.index_root,
        expected_schema_version=config.storage.registry_schema_version,
        marker_schema_version=config.storage.owner_marker_schema_version,
        busy_timeout_seconds=config.storage.registry_busy_timeout_seconds,
    )


def test_builder_seals_ready_generation_without_activation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = config.catalog.data_root / "math" / "notes.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Limits\n\n" + "The limit theorem has examples and proof details. " * 20,
        encoding="utf-8",
    )
    with _open_registry(config) as registry:
        result = GenerationBuilder(
            config=config,
            registry=registry,
            embedding_provider=_Embedding(),
            bm25_tokenizer=lambda text: tuple(text.split()),
            build_clock=lambda: datetime(2026, 7, 11, tzinfo=UTC),
        ).build(
            GenerationBuildRequest(
                schema_version="generation_build_request_v1",
                generation_id="gen-builder-a",
                code_revision="test-revision",
            )
        )

        assert result.registry_state == "READY"
        assert result.activated is False
        assert registry.get_generation("gen-builder-a").state == "READY"
        assert registry.deployment().primary_generation_id is None
        assert result.manifest.validation_passed is True
        assert result.manifest.integrity.all_zero()
        final = config.storage.index_root / "gen-builder-a"
        assert (final / "manifest.json").is_file()
        assert (final / "parents.sqlite").is_file()
        assert (final / "chroma_children").is_dir()
        assert (final / "bm25" / "math.jsonl").is_file()


def test_builder_records_failed_generation_without_sealing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = config.catalog.data_root / "math" / "notes.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Limits\n\nSome valid source text.", encoding="utf-8")
    with _open_registry(config) as registry:
        builder = GenerationBuilder(
            config=config,
            registry=registry,
            embedding_provider=_Embedding(),
            bm25_tokenizer=lambda _text: (),
            build_clock=lambda: datetime(2026, 7, 11, tzinfo=UTC),
        )
        with pytest.raises(GenerationBuildError, match="BM25"):
            builder.build(
                GenerationBuildRequest(
                    schema_version="generation_build_request_v1",
                    generation_id="gen-builder-failed",
                    code_revision="test-revision",
                )
            )

        record = registry.get_generation("gen-builder-failed")
        assert record.state == "FAILED"
        assert record.failure_code == "bm25"
        assert not (config.storage.index_root / "gen-builder-failed").exists()
        assert (config.storage.index_root / ".staging" / "gen-builder-failed").exists()
        registry.cleanup_generation("gen-builder-failed")
        assert registry.get_generation("gen-builder-failed").state == "DELETED"
        assert not (
            config.storage.index_root / ".staging" / "gen-builder-failed"
        ).exists()


def test_ready_generation_loads_exact_runtime_and_retrieves(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = config.catalog.data_root / "math" / "notes.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Limits\n\n" + "The limit theorem has examples and proof details. " * 20,
        encoding="utf-8",
    )
    embedding = _Embedding()

    def tokenizer(text: str) -> tuple[str, ...]:
        return tuple(text.split())

    with _open_registry(config) as registry:
        GenerationBuilder(
            config=config,
            registry=registry,
            embedding_provider=embedding,
            bm25_tokenizer=tokenizer,
            build_clock=lambda: datetime(2026, 7, 11, tzinfo=UTC),
        ).build(
            GenerationBuildRequest(
                schema_version="generation_build_request_v1",
                generation_id="gen-runtime-a",
                code_revision="test-revision",
            )
        )
        runtime = load_generation_runtime(
            config=config,
            registry_record=registry.get_generation("gen-runtime-a"),
            query_embedding_provider=embedding,
            reranker=_Reranker(),
            bm25_tokenizer=tokenizer,
        )
        try:
            result = runtime.retriever().retrieve(
                HybridRetrievalRequest(
                    schema_version="hybrid_retrieval_request_v1",
                    request_id="request-a",
                    query="limit",
                    subject="math",
                    generation_id="gen-runtime-a",
                )
            )
            assert runtime.available_subjects == ("math",)
            assert runtime.cross_branch_rrf_k == config.retrieval.cross_branch_rrf_k
            assert (
                runtime.judge_preview_max_chars
                == config.retrieval.judge_preview_max_chars
            )
            assert result.status == "ok"
            assert result.hydrated_parents
            assert result.retrieval_fingerprint
        finally:
            runtime.close()
        record = registry.get_generation("gen-runtime-a")
        assert record.manifest_sha256 is not None
        manifest = validate_sealed_generation(
            config.storage.index_root,
            "gen-runtime-a",
            expected_manifest_sha256=record.manifest_sha256,
            expected_marker_schema_version=(config.storage.owner_marker_schema_version),
        )
        assert manifest.generation_id == "gen-runtime-a"
        runtime_root = config.storage.index_root / ".runtime_chroma"
        assert runtime_root.is_dir()
        assert tuple(runtime_root.iterdir()) == ()
