"""Generate one strict, local Parent-Child RAG index configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import cast

import yaml  # type: ignore[import-untyped]


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config._rag_config import load_strict_rag_yaml  # noqa: E402
from src.config.rag_index_config import (  # noqa: E402
    Bm25Config,
    CatalogConfig,
    ChunkPolicyConfig,
    EmbeddingConfig,
    RagIndexConfig,
    RerankerConfig,
    RetrievalConfig,
    RetryConfig,
    StorageConfig,
    compute_chunk_policy_id,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child.project_paths import (  # noqa: E402
    atomic_write_project_bytes,
    require_project_directory,
    require_project_file,
    resolve_project_path,
    resolve_project_root,
)
from src.rag.parent_child.tokenizer import resolve_jieba_runtime_identity  # noqa: E402
from src.rag.subject_catalog import SubjectCatalog  # noqa: E402


class IndexConfigInitializationError(ValueError):
    """One explicit config-generator input cannot form a strict RAG config."""


def _portable_relative_path(
    *,
    project_root: Path,
    value: Path,
    must_be_directory: bool,
) -> Path:
    """Validate one project-contained path and retain a portable representation."""

    root = resolve_project_root(project_root)
    if must_be_directory:
        resolved = require_project_directory(root, value)
    else:
        resolved = resolve_project_path(root, value, must_exist=False)
        if resolved.exists() and not resolved.is_dir():
            raise IndexConfigInitializationError(
                "index_root must be a directory when it exists"
            )
    return resolved.relative_to(root)


def _portable_registry_path(
    *,
    project_root: Path,
    index_root: Path,
    registry_path: Path,
) -> Path:
    """Validate a registry path relative to its explicit index-root boundary."""

    root = resolve_project_root(project_root)
    index_root_resolved = resolve_project_path(root, index_root, must_exist=False)
    registry_candidate = (
        registry_path
        if registry_path.is_absolute()
        else index_root_resolved / registry_path
    )
    registry_resolved = resolve_project_path(root, registry_candidate, must_exist=False)
    if registry_resolved.exists() and not registry_resolved.is_file():
        raise IndexConfigInitializationError(
            "registry_path must be a regular file when it exists"
        )
    if not registry_resolved.is_relative_to(index_root_resolved):
        raise IndexConfigInitializationError(
            "registry_path must remain within index_root"
        )
    return registry_resolved.relative_to(index_root_resolved)


def _validate_experimental_runtime_catalog(config: RagIndexConfig) -> None:
    """Require the configured local corpus scope before writing runtime config."""

    exclusions: list[str] = []
    if "evaluation" not in config.catalog.excluded_exact_names:
        exclusions.append("evaluation")
    if not config.catalog.exclude_hidden:
        exclusions.append("hidden directories")
    if not config.catalog.exclude_cache_directories:
        exclusions.append("cache directories")
    if not config.catalog.cache_directory_names:
        exclusions.append("configured cache directory names")
    if not config.catalog.exclude_unclassified:
        exclusions.append("unclassified")
    if not config.catalog.exclude_needs_ocr:
        exclusions.append("_needs_ocr")
    if exclusions:
        raise IndexConfigInitializationError(
            "catalog exclusion contract is incomplete: " + ", ".join(exclusions)
        )
    if config.catalog.symlink_policy != "reject":
        raise IndexConfigInitializationError("catalog symlink_policy must be 'reject'")
    snapshot = SubjectCatalog(
        config=config.catalog,
        subject_policy_map=config.subject_policy_map,
    ).discover()
    if not snapshot.subject_ids():
        raise IndexConfigInitializationError(
            "catalog must discover at least one subject"
        )


def portable_runtime_config_from_source(
    *,
    project_root: Path,
    source_config_path: Path,
    data_root: Path,
    index_root: Path,
    registry_path: Path,
) -> RagIndexConfig:
    """Copy a strict template while requiring explicit portable local locations.

    Provider, model, endpoint, policy, and retrieval values come only from the
    strictly validated source configuration. This function never invents them.
    """

    root = resolve_project_root(project_root)
    template = load_rag_index_config(require_project_file(root, source_config_path))
    portable_data_root = _portable_relative_path(
        project_root=root,
        value=data_root,
        must_be_directory=True,
    )
    portable_index_root = _portable_relative_path(
        project_root=root,
        value=index_root,
        must_be_directory=False,
    )
    portable_registry_path = _portable_registry_path(
        project_root=root,
        index_root=portable_index_root,
        registry_path=registry_path,
    )
    payload = template.model_dump(mode="json")
    catalog = dict(payload["catalog"])
    storage = dict(payload["storage"])
    catalog["data_root"] = portable_data_root.as_posix()
    storage["index_root"] = portable_index_root.as_posix()
    storage["registry_path"] = portable_registry_path.as_posix()
    payload["catalog"] = catalog
    payload["storage"] = storage
    runtime = RagIndexConfig.model_validate(payload)
    _validate_experimental_runtime_catalog(
        resolve_rag_index_config_paths(runtime, project_root=root)
    )
    return runtime


def _strict_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("value must be exactly 'true' or 'false'")


def _json_string_sequence(value: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a JSON array of strings"
        ) from exc
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) for item in parsed
    ):
        raise argparse.ArgumentTypeError("value must be a JSON array of strings")
    return tuple(parsed)


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("value must be a JSON object") from exc
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def _assignment(value: str) -> tuple[str, str]:
    name, separator, assigned_value = value.partition("=")
    if not separator or not name or not assigned_value:
        raise argparse.ArgumentTypeError("value must use non-empty NAME=VALUE syntax")
    if name != name.strip() or assigned_value != assigned_value.strip():
        raise argparse.ArgumentTypeError(
            "NAME=VALUE cannot contain surrounding whitespace"
        )
    return name, assigned_value


def _assignment_mapping(
    values: list[tuple[str, str]],
    *,
    argument_name: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in values:
        if name in result:
            raise IndexConfigInitializationError(
                f"{argument_name} contains duplicate name '{name}'"
            )
        result[name] = value
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--schema-version", required=True)

    catalog = parser.add_argument_group("catalog")
    catalog.add_argument("--data-root", type=Path, required=True)
    catalog.add_argument(
        "--supported-extensions", type=_json_string_sequence, required=True
    )
    catalog.add_argument(
        "--excluded-exact-names", type=_json_string_sequence, required=True
    )
    catalog.add_argument(
        "--excluded-prefixes", type=_json_string_sequence, required=True
    )
    catalog.add_argument("--exclude-hidden", type=_strict_bool, required=True)
    catalog.add_argument(
        "--exclude-cache-directories", type=_strict_bool, required=True
    )
    catalog.add_argument(
        "--cache-directory-names", type=_json_string_sequence, required=True
    )
    catalog.add_argument("--exclude-unclassified", type=_strict_bool, required=True)
    catalog.add_argument("--unclassified-directory-name", required=True)
    catalog.add_argument("--exclude-needs-ocr", type=_strict_bool, required=True)
    catalog.add_argument("--needs-ocr-directory-name", required=True)
    catalog.add_argument(
        "--normalization-version",
        choices=("subject_id_v1",),
        required=True,
    )
    catalog.add_argument(
        "--catalog-symlink-policy",
        choices=("reject",),
        required=True,
    )

    storage = parser.add_argument_group("storage")
    storage.add_argument("--index-root", type=Path, required=True)
    storage.add_argument("--registry-path", type=Path, required=True)
    storage.add_argument("--collection-name", required=True)
    storage.add_argument("--parent-store-schema-version", required=True)
    storage.add_argument("--registry-schema-version", required=True)
    storage.add_argument("--owner-marker-schema-version", required=True)
    storage.add_argument("--registry-busy-timeout-seconds", type=float, required=True)
    storage.add_argument(
        "--parent-store-busy-timeout-seconds", type=float, required=True
    )
    storage.add_argument("--retention-generations", type=int, required=True)

    embedding = parser.add_argument_group("embedding")
    embedding.add_argument("--embedding-provider", required=True)
    embedding.add_argument(
        "--embedding-protocol",
        choices=("openai_embeddings_v1", "openrouter_embeddings_v1"),
        required=True,
    )
    embedding.add_argument("--embedding-model", required=True)
    embedding.add_argument("--embedding-response-model", required=True)
    embedding.add_argument("--embedding-base-url", required=True)
    embedding.add_argument("--embedding-endpoint-path", required=True)
    embedding.add_argument("--embedding-api-key-env", required=True)
    embedding.add_argument("--embedding-timeout-seconds", type=float, required=True)
    embedding.add_argument("--embedding-retry-max-attempts", type=int, required=True)
    embedding.add_argument(
        "--embedding-retry-initial-backoff-seconds",
        type=float,
        required=True,
    )
    embedding.add_argument(
        "--embedding-retry-max-backoff-seconds",
        type=float,
        required=True,
    )
    embedding.add_argument("--embedding-retry-multiplier", type=float, required=True)
    embedding.add_argument("--embedding-batch-size", type=int, required=True)
    embedding.add_argument(
        "--embedding-max-in-flight-batches",
        type=int,
        required=True,
    )
    embedding.add_argument("--embedding-expected-dimension", type=int, required=True)
    embedding.add_argument(
        "--embedding-distance-metric",
        choices=("cosine", "l2", "ip"),
        required=True,
    )
    embedding.add_argument("--embedding-normalization-contract", required=True)
    embedding.add_argument("--embedding-document-input-type", required=True)
    embedding.add_argument("--embedding-query-input-type", required=True)
    input_type_group = embedding.add_mutually_exclusive_group(required=True)
    input_type_group.add_argument("--embedding-input-type-field")
    input_type_group.add_argument(
        "--embedding-no-input-type-field",
        action="store_const",
        const=None,
        dest="embedding_input_type_field",
    )
    embedding_routing_group = embedding.add_mutually_exclusive_group(required=True)
    embedding_routing_group.add_argument(
        "--embedding-provider-routing",
        type=_json_object,
    )
    embedding_routing_group.add_argument(
        "--embedding-no-provider-routing",
        action="store_const",
        const=None,
        dest="embedding_provider_routing",
    )

    reranker = parser.add_argument_group("reranker")
    reranker.add_argument("--reranker-provider", required=True)
    reranker.add_argument("--reranker-model", required=True)
    reranker.add_argument("--reranker-response-model", required=True)
    reranker.add_argument("--reranker-base-url", required=True)
    reranker.add_argument("--reranker-endpoint-path", required=True)
    reranker.add_argument("--reranker-api-key-env", required=True)
    reranker.add_argument("--reranker-timeout-seconds", type=float, required=True)
    reranker.add_argument("--reranker-retry-max-attempts", type=int, required=True)
    reranker.add_argument(
        "--reranker-retry-initial-backoff-seconds",
        type=float,
        required=True,
    )
    reranker.add_argument(
        "--reranker-retry-max-backoff-seconds",
        type=float,
        required=True,
    )
    reranker.add_argument("--reranker-retry-multiplier", type=float, required=True)
    reranker.add_argument("--reranker-batch-size", type=int, required=True)
    reranker.add_argument(
        "--reranker-protocol",
        choices=(
            "ranked_index_scores_v1",
            "openrouter_ranked_index_scores_v1",
        ),
        required=True,
    )
    reranker.add_argument("--reranker-score-min", type=float, required=True)
    reranker.add_argument("--reranker-score-max", type=float, required=True)
    reranker_routing_group = reranker.add_mutually_exclusive_group(required=True)
    reranker_routing_group.add_argument(
        "--reranker-provider-routing",
        type=_json_object,
    )
    reranker_routing_group.add_argument(
        "--reranker-no-provider-routing",
        action="store_const",
        const=None,
        dest="reranker_provider_routing",
    )

    bm25 = parser.add_argument_group("bm25")
    bm25.add_argument(
        "--bm25-tokenizer",
        choices=("jieba_builtin_precise_v1",),
        required=True,
    )
    bm25.add_argument("--bm25-artifact-format", choices=("jsonl",), required=True)

    policies = parser.add_argument_group("chunk policies")
    policies.add_argument(
        "--chunk-policy", action="append", type=_assignment, required=True
    )
    policies.add_argument(
        "--subject-policy", action="append", type=_assignment, required=True
    )

    retrieval = parser.add_argument_group("retrieval")
    retrieval.add_argument("--vector-top-k", type=int, required=True)
    retrieval.add_argument("--bm25-top-k", type=int, required=True)
    retrieval.add_argument("--rrf-k", type=int, required=True)
    retrieval.add_argument("--vector-weight", type=float, required=True)
    retrieval.add_argument("--bm25-weight", type=float, required=True)
    retrieval.add_argument("--reranker-top-n", type=int, required=True)
    retrieval.add_argument("--unique-parent-top-k", type=int, required=True)
    retrieval.add_argument("--max-children-per-parent", type=int, required=True)
    retrieval.add_argument("--max-parents-per-source", type=int, required=True)
    retrieval.add_argument("--parent-support-lambda", type=float, required=True)
    retrieval.add_argument("--full-parent-max-chars", type=int, required=True)
    retrieval.add_argument("--hit-window-chars-per-side", type=int, required=True)
    retrieval.add_argument("--context-item-max-chars", type=int, required=True)
    retrieval.add_argument("--judge-preview-max-chars", type=int, required=True)
    retrieval.add_argument("--multi-subject-per-subject-top-k", type=int, required=True)
    retrieval.add_argument("--multi-subject-max-parents", type=int, required=True)
    retrieval.add_argument("--cross-branch-rrf-k", type=int, required=True)
    retrieval.add_argument("--subject-coverage-quota", type=int, required=True)
    return parser


def _retry_from_args(args: argparse.Namespace, prefix: str) -> RetryConfig:
    return RetryConfig(
        max_attempts=getattr(args, f"{prefix}_retry_max_attempts"),
        initial_backoff_seconds=getattr(
            args,
            f"{prefix}_retry_initial_backoff_seconds",
        ),
        max_backoff_seconds=getattr(args, f"{prefix}_retry_max_backoff_seconds"),
        multiplier=getattr(args, f"{prefix}_retry_multiplier"),
    )


def _catalog_from_args(args: argparse.Namespace, root: Path) -> CatalogConfig:
    return CatalogConfig(
        data_root=require_project_directory(root, args.data_root),
        supported_extensions=args.supported_extensions,
        excluded_exact_names=args.excluded_exact_names,
        excluded_prefixes=args.excluded_prefixes,
        exclude_hidden=args.exclude_hidden,
        exclude_cache_directories=args.exclude_cache_directories,
        cache_directory_names=args.cache_directory_names,
        exclude_unclassified=args.exclude_unclassified,
        unclassified_directory_name=args.unclassified_directory_name,
        exclude_needs_ocr=args.exclude_needs_ocr,
        needs_ocr_directory_name=args.needs_ocr_directory_name,
        normalization_version=args.normalization_version,
        symlink_policy=args.catalog_symlink_policy,
    )


def _storage_from_args(args: argparse.Namespace, root: Path) -> StorageConfig:
    index_root = resolve_project_path(root, args.index_root, must_exist=False)
    if index_root.exists() and not index_root.is_dir():
        raise IndexConfigInitializationError(
            "index_root must be a directory when it exists"
        )
    registry_path = resolve_project_path(root, args.registry_path, must_exist=False)
    if registry_path.exists() and not registry_path.is_file():
        raise IndexConfigInitializationError(
            "registry_path must be a regular file when it exists"
        )
    if not registry_path.is_relative_to(index_root):
        raise IndexConfigInitializationError(
            "registry_path must remain within index_root"
        )
    return StorageConfig(
        index_root=index_root,
        registry_path=registry_path,
        collection_name=args.collection_name,
        parent_store_schema_version=args.parent_store_schema_version,
        registry_schema_version=args.registry_schema_version,
        owner_marker_schema_version=args.owner_marker_schema_version,
        registry_busy_timeout_seconds=args.registry_busy_timeout_seconds,
        parent_store_busy_timeout_seconds=args.parent_store_busy_timeout_seconds,
        retention_generations=args.retention_generations,
    )


def _embedding_from_args(args: argparse.Namespace) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=args.embedding_provider,
        protocol=args.embedding_protocol,
        model=args.embedding_model,
        response_model=args.embedding_response_model,
        base_url=args.embedding_base_url,
        endpoint_path=args.embedding_endpoint_path,
        api_key_env=args.embedding_api_key_env,
        timeout_seconds=args.embedding_timeout_seconds,
        retry=_retry_from_args(args, "embedding"),
        batch_size=args.embedding_batch_size,
        max_in_flight_batches=args.embedding_max_in_flight_batches,
        expected_dimension=args.embedding_expected_dimension,
        distance_metric=args.embedding_distance_metric,
        normalization_contract=args.embedding_normalization_contract,
        document_input_type=args.embedding_document_input_type,
        query_input_type=args.embedding_query_input_type,
        input_type_field=args.embedding_input_type_field,
        provider_routing=args.embedding_provider_routing,
    )


def _reranker_from_args(args: argparse.Namespace) -> RerankerConfig:
    return RerankerConfig(
        provider=args.reranker_provider,
        model=args.reranker_model,
        response_model=args.reranker_response_model,
        base_url=args.reranker_base_url,
        endpoint_path=args.reranker_endpoint_path,
        api_key_env=args.reranker_api_key_env,
        timeout_seconds=args.reranker_timeout_seconds,
        retry=_retry_from_args(args, "reranker"),
        batch_size=args.reranker_batch_size,
        protocol=args.reranker_protocol,
        score_min=args.reranker_score_min,
        score_max=args.reranker_score_max,
        provider_routing=args.reranker_provider_routing,
    )


def _retrieval_from_args(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        rrf_k=args.rrf_k,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        reranker_top_n=args.reranker_top_n,
        unique_parent_top_k=args.unique_parent_top_k,
        max_children_per_parent=args.max_children_per_parent,
        max_parents_per_source=args.max_parents_per_source,
        parent_support_lambda=args.parent_support_lambda,
        full_parent_max_chars=args.full_parent_max_chars,
        hit_window_chars_per_side=args.hit_window_chars_per_side,
        context_item_max_chars=args.context_item_max_chars,
        judge_preview_max_chars=args.judge_preview_max_chars,
        multi_subject_per_subject_top_k=args.multi_subject_per_subject_top_k,
        multi_subject_max_parents=args.multi_subject_max_parents,
        cross_branch_rrf_k=args.cross_branch_rrf_k,
        subject_coverage_quota=args.subject_coverage_quota,
    )


def _load_named_policies(
    *,
    project_root: Path,
    assignments: list[tuple[str, str]],
) -> dict[str, ChunkPolicyConfig]:
    paths = _assignment_mapping(assignments, argument_name="--chunk-policy")
    loaded: dict[str, ChunkPolicyConfig] = {}
    for name in sorted(paths):
        fragment_path = require_project_file(project_root, Path(paths[name]))
        loaded[name] = load_strict_rag_yaml(fragment_path, ChunkPolicyConfig)
    return loaded


def build_initialized_config(
    *,
    project_root: Path,
    schema_version: str,
    catalog: CatalogConfig,
    storage: StorageConfig,
    embedding: EmbeddingConfig,
    reranker: RerankerConfig,
    bm25_tokenizer: str,
    bm25_artifact_format: str,
    named_policies: dict[str, ChunkPolicyConfig],
    subject_policy_names: dict[str, str],
    retrieval: RetrievalConfig,
) -> RagIndexConfig:
    """Build and fully validate one config from explicit local inputs."""

    root = resolve_project_root(project_root)
    if catalog.symlink_policy != "reject":
        raise IndexConfigInitializationError("catalog symlink_policy must be 'reject'")
    if not named_policies:
        raise IndexConfigInitializationError(
            "at least one named chunk policy is required"
        )
    if not subject_policy_names:
        raise IndexConfigInitializationError(
            "at least one subject-policy mapping is required"
        )

    unknown_policy_names = sorted(
        set(subject_policy_names.values()) - set(named_policies)
    )
    unused_policy_names = sorted(
        set(named_policies) - set(subject_policy_names.values())
    )
    if unknown_policy_names:
        raise IndexConfigInitializationError(
            "subject-policy references unknown policy names: "
            + ", ".join(unknown_policy_names)
        )
    if unused_policy_names:
        raise IndexConfigInitializationError(
            "chunk-policy names are unused: " + ", ".join(unused_policy_names)
        )

    policy_ids_by_name = {
        name: compute_chunk_policy_id(policy) for name, policy in named_policies.items()
    }
    requested_subject_policy_ids = {
        subject: policy_ids_by_name[policy_name]
        for subject, policy_name in subject_policy_names.items()
    }
    snapshot = SubjectCatalog(
        config=catalog,
        subject_policy_map=requested_subject_policy_ids,
    ).discover()
    used_policy_ids = set(snapshot.subject_policy_map.values())
    chunk_policies = {
        policy_id: named_policies[name]
        for name, policy_id in sorted(policy_ids_by_name.items())
        if policy_id in used_policy_ids
    }

    jieba_identity = resolve_jieba_runtime_identity()
    config = RagIndexConfig(
        schema_version=schema_version,
        catalog=catalog,
        storage=storage,
        embedding=embedding,
        reranker=reranker,
        bm25=Bm25Config(
            tokenizer=bm25_tokenizer,
            tokenizer_version=jieba_identity.tokenizer_version,
            dictionary_hash=jieba_identity.dictionary_hash,
            artifact_format=bm25_artifact_format,
        ),
        chunk_policies=dict(sorted(chunk_policies.items())),
        subject_policy_map=dict(sorted(snapshot.subject_policy_map.items())),
        retrieval=retrieval,
    )
    return resolve_rag_index_config_paths(config, project_root=root)


def write_initialized_config(
    *,
    project_root: Path,
    output_path: Path,
    config: RagIndexConfig,
    overwrite: bool,
) -> Path:
    """Write stable YAML, then re-load it through the strict config loader."""

    root = resolve_project_root(project_root)
    output = resolve_project_path(root, output_path, must_exist=False)
    payload = config.model_dump(mode="json")
    encoded = yaml.safe_dump(
        payload,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).encode("utf-8")
    written = atomic_write_project_bytes(
        root,
        output,
        encoded,
        overwrite=overwrite,
    )
    reloaded = load_rag_index_config(written)
    if reloaded != config:
        raise IndexConfigInitializationError(
            "written index config failed strict round-trip validation"
        )
    return written


def write_portable_runtime_config(
    *,
    project_root: Path,
    output_path: Path,
    config: RagIndexConfig,
    overwrite: bool,
) -> Path:
    """Write a strict runtime config while preserving POSIX relative path text."""

    root = resolve_project_root(project_root)
    output = resolve_project_path(root, output_path, must_exist=False)
    payload = config.model_dump(mode="json")
    catalog = dict(payload["catalog"])
    storage = dict(payload["storage"])
    catalog["data_root"] = config.catalog.data_root.as_posix()
    storage["index_root"] = config.storage.index_root.as_posix()
    storage["registry_path"] = config.storage.registry_path.as_posix()
    payload["catalog"] = catalog
    payload["storage"] = storage
    encoded = yaml.safe_dump(
        payload,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).encode("utf-8")
    written = atomic_write_project_bytes(
        root,
        output,
        encoded,
        overwrite=overwrite,
    )
    if load_rag_index_config(written) != config:
        raise IndexConfigInitializationError(
            "written portable runtime config failed strict round-trip validation"
        )
    return written


def run_init(args: argparse.Namespace) -> Path:
    """Generate the config described by a fully parsed CLI namespace."""

    root = resolve_project_root(cast(Path, args.project_root))
    named_policies = _load_named_policies(
        project_root=root,
        assignments=cast(list[tuple[str, str]], args.chunk_policy),
    )
    subject_policy_names = _assignment_mapping(
        cast(list[tuple[str, str]], args.subject_policy),
        argument_name="--subject-policy",
    )
    config = build_initialized_config(
        project_root=root,
        schema_version=args.schema_version,
        catalog=_catalog_from_args(args, root),
        storage=_storage_from_args(args, root),
        embedding=_embedding_from_args(args),
        reranker=_reranker_from_args(args),
        bm25_tokenizer=args.bm25_tokenizer,
        bm25_artifact_format=args.bm25_artifact_format,
        named_policies=named_policies,
        subject_policy_names=subject_policy_names,
        retrieval=_retrieval_from_args(args),
    )
    return write_initialized_config(
        project_root=root,
        output_path=cast(Path, args.output),
        config=config,
        overwrite=cast(bool, args.overwrite),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run_init(args)
    print(f"Strict local RAG index config generated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
