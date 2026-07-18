"""Strict, provider-neutral configuration for production RAG generations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, Field, field_validator, model_validator

from src.config._rag_config import (
    ConfigPath,
    NonBlankStr,
    NonBlankStrTuple,
    NonEmptyStr,
    NonEmptyStrTuple,
    StrictRagConfigModel,
    load_strict_rag_yaml,
)


def _subject_id(value: str) -> str:
    if value != value.casefold():
        raise ValueError("subject identifier must already be case-folded")
    if value.startswith("_") or value.endswith("_") or "__" in value:
        raise ValueError("subject identifier has invalid underscore placement")
    if not all(character.isalnum() or character == "_" for character in value):
        raise ValueError("subject identifier contains unsupported characters")
    return value


def _base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    return value


def _endpoint_path(value: str) -> str:
    if not value.startswith("/") or value.startswith("//"):
        raise ValueError("endpoint_path must begin with exactly one '/'")
    if "?" in value or "#" in value or ".." in value.split("/"):
        raise ValueError("endpoint_path must not contain traversal, query, or fragment")
    return value


def _unique(values: tuple[object, ...], *, field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


_PROVIDER_ROUTING_SLUG = re.compile(
    r"^[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?(?:/[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?)*$"
)


def _provider_routing_slug(value: str) -> str:
    """Validate one provider-routing slug without normalizing it."""

    if not _PROVIDER_ROUTING_SLUG.fullmatch(value):
        raise ValueError("provider routing slug must be lower-case and slash-safe")
    return value


SubjectId = Annotated[NonBlankStr, AfterValidator(_subject_id)]
PolicyId = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BaseUrl = Annotated[NonBlankStr, AfterValidator(_base_url)]
EndpointPath = Annotated[NonBlankStr, AfterValidator(_endpoint_path)]
ProviderRoutingSlug = Annotated[
    NonBlankStr,
    AfterValidator(_provider_routing_slug),
]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CatalogConfig(StrictRagConfigModel):
    """Filesystem discovery policy for the source corpus."""

    data_root: ConfigPath
    supported_extensions: NonBlankStrTuple
    excluded_exact_names: NonBlankStrTuple
    excluded_prefixes: NonBlankStrTuple
    exclude_hidden: bool
    exclude_cache_directories: bool
    cache_directory_names: NonBlankStrTuple
    exclude_unclassified: bool
    unclassified_directory_name: NonBlankStr
    exclude_needs_ocr: bool
    needs_ocr_directory_name: NonBlankStr
    normalization_version: Literal["subject_id_v1"]
    symlink_policy: Literal["reject", "allow_internal"]

    @model_validator(mode="after")
    def _validate_discovery_policy(self) -> "CatalogConfig":
        _unique(self.supported_extensions, field_name="supported_extensions")
        _unique(self.excluded_exact_names, field_name="excluded_exact_names")
        _unique(self.excluded_prefixes, field_name="excluded_prefixes")
        _unique(self.cache_directory_names, field_name="cache_directory_names")
        for extension in self.supported_extensions:
            if not extension.startswith(".") or extension != extension.casefold():
                raise ValueError(
                    "supported_extensions must be lower-case suffixes beginning with '.'"
                )
        return self


class StorageConfig(StrictRagConfigModel):
    """Immutable generation storage locations and identifiers."""

    index_root: ConfigPath
    registry_path: ConfigPath
    collection_name: Annotated[str, Field(min_length=3, pattern=r"^[A-Za-z0-9_-]+$")]
    parent_store_schema_version: NonBlankStr
    registry_schema_version: NonBlankStr
    owner_marker_schema_version: NonBlankStr
    registry_busy_timeout_seconds: PositiveFloat
    parent_store_busy_timeout_seconds: PositiveFloat
    retention_generations: PositiveInt

    @model_validator(mode="after")
    def _validate_registry_containment(self) -> "StorageConfig":
        if self.index_root.is_absolute():
            index_root = self.index_root.resolve()
            registry_path = self.registry_path
            if not registry_path.is_absolute():
                registry_path = index_root / registry_path
            if not registry_path.resolve().is_relative_to(index_root):
                raise ValueError("registry_path must resolve within index_root")
            return self
        if self.registry_path.is_absolute():
            raise ValueError(
                "registry_path must be relative when index_root is project-relative"
            )
        if ".." in self.registry_path.parts:
            raise ValueError("relative registry_path must not contain parent traversal")
        return self

    def resolved_registry_path(self) -> Path:
        """Return the validated registry location under ``index_root``."""
        if not self.index_root.is_absolute():
            raise ValueError(
                "storage paths must be resolved against an explicit project_root first"
            )
        index_root = self.index_root.resolve()
        if self.registry_path.is_absolute():
            return self.registry_path.resolve()
        return (index_root / self.registry_path).resolve()


class RetryConfig(StrictRagConfigModel):
    """Explicit bounded retry policy; it never changes provider or model."""

    max_attempts: PositiveInt
    initial_backoff_seconds: PositiveFloat
    max_backoff_seconds: PositiveFloat
    multiplier: Annotated[float, Field(ge=1.0)]

    @model_validator(mode="after")
    def _validate_backoff(self) -> "RetryConfig":
        if self.max_attempts > 10:
            raise ValueError("max_attempts must not exceed 10")
        if self.max_backoff_seconds > 60.0:
            raise ValueError("max_backoff_seconds must not exceed 60 seconds")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "max_backoff_seconds must be at least initial_backoff_seconds"
            )
        return self


class ProviderRoutingConfig(StrictRagConfigModel):
    """One explicit no-fallback OpenRouter provider route."""

    order: Annotated[tuple[ProviderRoutingSlug, ...], Field(min_length=1, max_length=1)]
    allow_fallbacks: Literal[False]

    @field_validator("order", mode="before")
    @classmethod
    def _freeze_order(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


class EmbeddingConfig(StrictRagConfigModel):
    """Explicit embedding endpoint configuration and response protocol."""

    provider: NonBlankStr
    protocol: Literal["openai_embeddings_v1", "openrouter_embeddings_v1"]
    model: NonBlankStr
    response_model: NonBlankStr
    base_url: BaseUrl
    endpoint_path: EndpointPath
    api_key_env: Annotated[
        str,
        Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
    ]
    timeout_seconds: PositiveFloat
    retry: RetryConfig
    batch_size: PositiveInt
    max_in_flight_batches: Annotated[PositiveInt, Field(le=4)]
    expected_dimension: PositiveInt
    distance_metric: Literal["cosine", "l2", "ip"]
    normalization_contract: NonBlankStr
    document_input_type: NonBlankStr
    query_input_type: NonBlankStr
    input_type_field: NonBlankStr | None
    provider_routing: ProviderRoutingConfig | None

    @model_validator(mode="after")
    def _validate_provider_specific_protocol(self) -> "EmbeddingConfig":
        """Keep provider-specific wire contracts explicit and fail-fast."""

        if self.protocol != "openrouter_embeddings_v1":
            if self.provider == "openrouter":
                raise ValueError(
                    "provider 'openrouter' requires openrouter_embeddings_v1"
                )
            if self.provider_routing is not None:
                raise ValueError(
                    "non-openrouter embedding providers require provider_routing to be null"
                )
            return self
        if self.provider != "openrouter":
            raise ValueError(
                "openrouter_embeddings_v1 requires provider to be 'openrouter'"
            )
        if self.provider_routing is None:
            raise ValueError(
                "openrouter_embeddings_v1 requires explicit provider_routing"
            )
        if self.input_type_field is not None:
            raise ValueError(
                "openrouter_embeddings_v1 requires input_type_field to be null"
            )
        return self


class RerankerConfig(StrictRagConfigModel):
    """Provider-neutral reranker endpoint and response contract."""

    provider: NonBlankStr
    model: NonBlankStr
    response_model: NonBlankStr
    base_url: BaseUrl
    endpoint_path: EndpointPath
    api_key_env: Annotated[
        str,
        Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
    ]
    timeout_seconds: PositiveFloat
    retry: RetryConfig
    batch_size: PositiveInt
    protocol: NonBlankStr
    score_min: float
    score_max: float
    provider_routing: ProviderRoutingConfig | None

    @model_validator(mode="after")
    def _validate_score_contract(self) -> "RerankerConfig":
        if self.score_min != 0.0 or self.score_max != 1.0:
            raise ValueError("reranker score contract must be exactly [0.0, 1.0]")
        if self.protocol == "openrouter_ranked_index_scores_v1":
            if self.provider != "openrouter":
                raise ValueError(
                    "openrouter_ranked_index_scores_v1 requires provider to be "
                    "'openrouter'"
                )
            if self.provider_routing is None:
                raise ValueError(
                    "openrouter_ranked_index_scores_v1 requires explicit "
                    "provider_routing"
                )
        elif self.provider == "openrouter":
            raise ValueError(
                "provider 'openrouter' requires openrouter_ranked_index_scores_v1"
            )
        elif self.provider_routing is not None:
            raise ValueError(
                "non-openrouter reranker providers require provider_routing to be null"
            )
        elif (
            self.protocol == "ranked_index_scores_v1"
            and self.response_model != self.model
        ):
            raise ValueError(
                "ranked_index_scores_v1 requires response_model to equal model"
            )
        return self


class Bm25Config(StrictRagConfigModel):
    """Safe per-subject BM25 artifact contract."""

    tokenizer: NonBlankStr
    tokenizer_version: NonBlankStr
    dictionary_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    artifact_format: Literal["jsonl"]


class ExtractionPolicyConfig(StrictRagConfigModel):
    algorithm_version: NonBlankStr
    pdf_extraction_method: NonBlankStr
    text_extraction_method: NonBlankStr


def _validate_source_relpath(value: str) -> str:
    if "\\" in value or "\x00" in value:
        raise ValueError(
            "OCR source_relpaths must use POSIX separators and contain no NUL"
        )
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(
            "OCR source_relpaths must not contain empty, dot, or parent parts"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise ValueError("OCR source_relpaths must be contained relative POSIX paths")
    if path.as_posix() != value:
        raise ValueError("OCR source_relpaths must be canonical")
    if path.suffix.casefold() != ".pdf":
        raise ValueError("OCR source_relpaths must identify PDF sources")
    return value


class TesseractLanguageAssetConfig(StrictRagConfigModel):
    """One ordered Tesseract language asset identity."""

    language: Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]
    traineddata_sha256: Sha256Hex


class TesseractOcrPolicyConfig(StrictRagConfigModel):
    """Explicit, fingerprinted Tesseract runtime and source selection."""

    schema_version: Literal["tesseract_ocr_policy_v1"]
    engine_protocol: Literal["tesseract_cli_tsv_v1"]
    extraction_method: Literal["tesseract_cli_tsv_v1"]
    source_relpaths: tuple[NonBlankStr, ...] = Field(min_length=1)
    binary_path: ConfigPath
    binary_sha256: Sha256Hex
    runtime_manifest_sha256: Sha256Hex
    expected_version: NonBlankStr
    renderer_protocol: Literal["pymupdf_pixmap_png_v1"]
    pymupdf_version: NonBlankStr
    mupdf_version: NonBlankStr
    tessdata_dir: ConfigPath
    language_assets: tuple[TesseractLanguageAssetConfig, ...] = Field(min_length=1)
    dpi: Annotated[int, Field(ge=72, le=600)]
    render_colorspace: Literal["rgb"]
    render_alpha: Literal[False]
    render_annotations: Literal[True]
    oem: Literal[1]
    psm: Literal[3]
    thread_limit: Literal[1]
    timeout_seconds: Annotated[float, Field(gt=0.0, le=300.0)]
    output_format: Literal["tsv_lines_v1"]
    empty_page_policy: Literal["allow_empty_physical_page_v1"]

    @field_validator("source_relpaths", "language_assets", mode="before")
    @classmethod
    def _freeze_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("source_relpaths")
    @classmethod
    def _validate_source_relpaths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(_validate_source_relpath(item) for item in value)
        _unique(validated, field_name="source_relpaths")
        if validated != tuple(sorted(validated)):
            raise ValueError("OCR source_relpaths must be sorted")
        return validated

    @field_validator("language_assets")
    @classmethod
    def _validate_language_assets(
        cls, value: tuple[TesseractLanguageAssetConfig, ...]
    ) -> tuple[TesseractLanguageAssetConfig, ...]:
        languages = tuple(asset.language for asset in value)
        _unique(languages, field_name="language_assets")
        return value


class OcrExtractionPolicyConfig(StrictRagConfigModel):
    """Page extraction V2 with an exact, no-fallback OCR source inventory."""

    algorithm_version: Literal["page_extract_tesseract_v2"]
    pdf_extraction_method: Literal["configured_pdf_text"]
    text_extraction_method: Literal["configured_utf8_text"]
    pdf_ocr: TesseractOcrPolicyConfig


class PageAssemblyPolicyConfig(StrictRagConfigModel):
    algorithm_version: NonBlankStr
    page_separator: NonEmptyStr


class CleaningPolicyConfig(StrictRagConfigModel):
    algorithm_version: Literal["page_clean_v2"]
    nul_character_policy: Literal["replace_with_space_v1", "reject"]
    normalize_newlines: bool
    strip_trailing_whitespace: bool
    strip_outer_blank_lines: bool
    header_top_lines: PositiveInt
    footer_bottom_lines: PositiveInt
    repeated_line_min_pages: PositiveInt
    repeated_line_min_ratio: UnitFloat
    collapse_blank_lines: bool
    paragraph_deduplication: bool


class StructurePolicyConfig(StrictRagConfigModel):
    detector_version: NonBlankStr
    pattern_set_version: NonBlankStr
    merge_version: NonBlankStr
    short_unit_chars: PositiveInt
    major_boundary_levels: Annotated[
        tuple[PositiveInt, ...],
        # YAML has sequences rather than native Python tuples.
        # NonBlankStrTuple cannot be reused because the item type is numeric.
        Field(min_length=1),
    ]

    @field_validator("major_boundary_levels", mode="before")
    @classmethod
    def _freeze_boundary_levels(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_boundary_levels(self) -> "StructurePolicyConfig":
        _unique(self.major_boundary_levels, field_name="major_boundary_levels")
        if self.major_boundary_levels != tuple(sorted(self.major_boundary_levels)):
            raise ValueError("major_boundary_levels must be sorted")
        expected = tuple(range(1, max(self.major_boundary_levels) + 1))
        if self.major_boundary_levels != expected:
            raise ValueError(
                "major_boundary_levels must be contiguous and start at level 1"
            )
        return self


class AtomicBlockPolicyConfig(StrictRagConfigModel):
    policy_version: NonBlankStr
    protected_types: NonBlankStrTuple
    hard_max_chars: PositiveInt

    @model_validator(mode="after")
    def _validate_protected_types(self) -> "AtomicBlockPolicyConfig":
        _unique(self.protected_types, field_name="protected_types")
        return self


class RecursiveChunkConfig(StrictRagConfigModel):
    algorithm_version: NonBlankStr
    size: PositiveInt
    overlap: NonNegativeInt
    separators: NonEmptyStrTuple
    length_policy: NonBlankStr
    whitespace_policy: NonBlankStr

    @model_validator(mode="after")
    def _validate_overlap(self) -> "RecursiveChunkConfig":
        if self.overlap >= self.size:
            raise ValueError("overlap must be less than size")
        _unique(self.separators, field_name="separators")
        return self


class ChunkPolicyConfig(StrictRagConfigModel):
    """All output-affecting parent/child chunk policy inputs."""

    extraction: ExtractionPolicyConfig | OcrExtractionPolicyConfig
    page_assembly: PageAssemblyPolicyConfig
    cleaning: CleaningPolicyConfig
    structure: StructurePolicyConfig
    atomic_blocks: AtomicBlockPolicyConfig
    parent: RecursiveChunkConfig
    child: RecursiveChunkConfig
    metadata_contract_version: NonBlankStr

    @model_validator(mode="after")
    def _validate_parent_child_bounds(self) -> "ChunkPolicyConfig":
        if self.child.size > self.parent.size:
            raise ValueError("child size must not exceed parent size")
        if self.parent.size > self.atomic_blocks.hard_max_chars:
            raise ValueError("parent size must not exceed atomic block hard_max_chars")
        return self


def chunk_policy_manifest_payload(
    policy: ChunkPolicyConfig,
) -> dict[str, object]:
    """Return the canonical output-affecting V1 policy-manifest payload."""

    extraction = policy.extraction.model_dump(mode="json")
    if isinstance(policy.extraction, OcrExtractionPolicyConfig):
        pdf_ocr = dict(extraction["pdf_ocr"])
        # Installation paths are operational details. Their exact runtime
        # identities are still sealed through the configured SHA-256 values.
        pdf_ocr.pop("binary_path")
        pdf_ocr.pop("tessdata_dir")
        extraction["pdf_ocr"] = pdf_ocr

    return {
        "schema_version": "policy_manifest_v1",
        "canonicalization_version": "canonical_json_v1",
        "id_algorithm_version": "parent_child_id_v1",
        "extraction": extraction,
        "page_assembly": policy.page_assembly.model_dump(mode="json"),
        "cleaning": policy.cleaning.model_dump(mode="json"),
        "structure": policy.structure.model_dump(mode="json"),
        "atomic_blocks": policy.atomic_blocks.model_dump(mode="json"),
        "parent_split": policy.parent.model_dump(mode="json"),
        "child_split": policy.child.model_dump(mode="json"),
        "metadata_contract_version": policy.metadata_contract_version,
    }


def compute_chunk_policy_id(policy: ChunkPolicyConfig) -> str:
    """Recompute the policy ID from canonical output-affecting configuration."""

    encoded = json.dumps(
        chunk_policy_manifest_payload(policy),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RetrievalConfig(StrictRagConfigModel):
    """Explicit hybrid retrieval, aggregation, and expansion controls."""

    vector_top_k: PositiveInt
    bm25_top_k: PositiveInt
    rrf_k: PositiveInt
    vector_weight: PositiveFloat
    bm25_weight: PositiveFloat
    reranker_top_n: PositiveInt
    unique_parent_top_k: PositiveInt
    max_children_per_parent: PositiveInt
    max_parents_per_source: PositiveInt
    parent_support_lambda: UnitFloat
    full_parent_max_chars: PositiveInt
    hit_window_chars_per_side: PositiveInt
    context_item_max_chars: PositiveInt
    judge_preview_max_chars: PositiveInt
    multi_subject_per_subject_top_k: PositiveInt
    multi_subject_max_parents: PositiveInt
    cross_branch_rrf_k: PositiveInt
    subject_coverage_quota: PositiveInt

    @model_validator(mode="after")
    def _validate_retrieval_bounds(self) -> "RetrievalConfig":
        candidate_count = self.vector_top_k + self.bm25_top_k
        if self.reranker_top_n > candidate_count:
            raise ValueError("reranker_top_n must not exceed vector_top_k + bm25_top_k")
        if self.unique_parent_top_k > self.reranker_top_n:
            raise ValueError("unique_parent_top_k must not exceed reranker_top_n")
        if self.full_parent_max_chars > self.context_item_max_chars:
            raise ValueError(
                "full_parent_max_chars must not exceed context_item_max_chars"
            )
        if self.multi_subject_per_subject_top_k > self.multi_subject_max_parents:
            raise ValueError(
                "multi_subject_per_subject_top_k must not exceed "
                "multi_subject_max_parents"
            )
        if self.subject_coverage_quota > self.multi_subject_per_subject_top_k:
            raise ValueError("subject_coverage_quota must not exceed per-subject top_k")
        return self


class RagIndexConfig(StrictRagConfigModel):
    """Complete strict configuration for a parent-child RAG generation."""

    schema_version: NonBlankStr
    catalog: CatalogConfig
    storage: StorageConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    bm25: Bm25Config
    chunk_policies: dict[PolicyId, ChunkPolicyConfig]
    subject_policy_map: dict[SubjectId, PolicyId]
    retrieval: RetrievalConfig

    @model_validator(mode="after")
    def _validate_complete_policy_mapping(self) -> "RagIndexConfig":
        if not self.chunk_policies:
            raise ValueError("chunk_policies must not be empty")
        if not self.subject_policy_map:
            raise ValueError("subject_policy_map must not be empty")
        configured_policy_ids = set(self.chunk_policies)
        referenced_policy_ids = set(self.subject_policy_map.values())
        unknown = referenced_policy_ids - configured_policy_ids
        unused = configured_policy_ids - referenced_policy_ids
        if unknown or unused:
            raise ValueError(
                "subject_policy_map and chunk_policies must reference the same policy IDs"
            )
        mismatched_policy_ids = tuple(
            sorted(
                policy_id
                for policy_id, policy in self.chunk_policies.items()
                if compute_chunk_policy_id(policy) != policy_id
            )
        )
        if mismatched_policy_ids:
            raise ValueError(
                "chunk policy IDs do not match canonical output-affecting config: "
                + ", ".join(mismatched_policy_ids)
            )

        largest_child = max(
            policy.child.size for policy in self.chunk_policies.values()
        )
        expanded_window_chars = (
            2 * self.retrieval.hit_window_chars_per_side + largest_child
        )
        if expanded_window_chars > self.retrieval.context_item_max_chars:
            raise ValueError(
                "hit window plus largest child must fit context_item_max_chars"
            )
        if self.retrieval.reranker_top_n > self.reranker.batch_size:
            raise ValueError("reranker_top_n must not exceed reranker.batch_size")
        maximum_flat_candidates = (
            self.retrieval.vector_top_k + self.retrieval.bm25_top_k
        )
        if maximum_flat_candidates > self.reranker.batch_size:
            raise ValueError(
                "reranker.batch_size must cover vector_top_k + bm25_top_k "
                "for strict Flat Baseline retrieval"
            )
        minimum_coverage_parents = (
            len(self.subject_policy_map) * self.retrieval.subject_coverage_quota
        )
        if self.retrieval.multi_subject_max_parents < minimum_coverage_parents:
            raise ValueError(
                "multi_subject_max_parents cannot satisfy all subject coverage quotas"
            )
        return self


def load_rag_index_config(config_path: Path) -> RagIndexConfig:
    """Load a required production RAG index YAML file."""
    return load_strict_rag_yaml(config_path, RagIndexConfig)


def resolve_rag_index_config_paths(
    config: RagIndexConfig,
    *,
    project_root: Path,
) -> RagIndexConfig:
    """Resolve configured roots under one explicit project containment boundary."""

    root = project_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project_root must be a directory")

    def resolve_contained(value: Path, *, field_name: str) -> Path:
        candidate = value if value.is_absolute() else root / value
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise ValueError(f"{field_name} must resolve within project_root")
        return resolved

    def reject_symlink_components(value: Path, *, field_name: str) -> None:
        candidate = value if value.is_absolute() else root / value
        if ".." in candidate.parts:
            raise ValueError(f"{field_name} must not contain parent traversal")
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be lexically contained in project_root"
            ) from exc
        cursor = root
        for part in relative.parts:
            cursor /= part
            try:
                path_status = cursor.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ValueError(f"{field_name} could not be inspected") from exc
            attributes = getattr(path_status, "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
            if cursor.is_symlink() or attributes & reparse_flag:
                raise ValueError(
                    f"{field_name} must not traverse a symlink or reparse point"
                )

    payload = config.model_dump(mode="python")
    catalog = dict(payload["catalog"])
    storage = dict(payload["storage"])
    reject_symlink_components(
        config.catalog.data_root,
        field_name="catalog.data_root",
    )
    reject_symlink_components(
        config.storage.index_root,
        field_name="storage.index_root",
    )
    catalog["data_root"] = resolve_contained(
        config.catalog.data_root,
        field_name="catalog.data_root",
    )
    storage["index_root"] = resolve_contained(
        config.storage.index_root,
        field_name="storage.index_root",
    )
    registry_path = config.storage.registry_path
    logical_registry_path = (
        registry_path
        if registry_path.is_absolute()
        else config.storage.index_root / registry_path
    )
    reject_symlink_components(
        logical_registry_path,
        field_name="storage.registry_path",
    )
    if registry_path.is_absolute():
        resolved_registry = registry_path.resolve(strict=False)
        if not resolved_registry.is_relative_to(storage["index_root"]):
            raise ValueError("storage.registry_path must remain inside index_root")
        storage["registry_path"] = resolved_registry
    chunk_policies = dict(payload["chunk_policies"])
    configured_ocr_sources: set[str] = set()
    for policy_id, policy in config.chunk_policies.items():
        if not isinstance(policy.extraction, OcrExtractionPolicyConfig):
            continue
        runtime = policy.extraction.pdf_ocr
        binary_path = resolve_contained(
            runtime.binary_path,
            field_name="chunk_policies.extraction.pdf_ocr.binary_path",
        )
        tessdata_dir = resolve_contained(
            runtime.tessdata_dir,
            field_name="chunk_policies.extraction.pdf_ocr.tessdata_dir",
        )
        reject_symlink_components(
            runtime.binary_path,
            field_name="chunk_policies.extraction.pdf_ocr.binary_path",
        )
        reject_symlink_components(
            runtime.tessdata_dir,
            field_name="chunk_policies.extraction.pdf_ocr.tessdata_dir",
        )
        if not binary_path.is_file():
            raise ValueError("configured OCR binary_path must be an existing file")
        if not tessdata_dir.is_dir():
            raise ValueError(
                "configured OCR tessdata_dir must be an existing directory"
            )
        for asset in runtime.language_assets:
            traineddata = tessdata_dir / f"{asset.language}.traineddata"
            reject_symlink_components(
                traineddata,
                field_name="configured OCR traineddata file",
            )
            if not traineddata.is_file():
                raise ValueError(
                    "configured OCR language is missing its traineddata file"
                )
        for source_relpath in runtime.source_relpaths:
            if source_relpath in configured_ocr_sources:
                raise ValueError("OCR source_relpaths must be globally unique")
            configured_ocr_sources.add(source_relpath)
            relative = PurePosixPath(source_relpath)
            subject = relative.parts[0]
            if config.subject_policy_map.get(subject) != policy_id:
                raise ValueError(
                    "OCR source subject must reference the containing chunk policy"
                )
            source_path = Path(catalog["data_root"]) / Path(*relative.parts)
            reject_symlink_components(
                source_path,
                field_name="configured OCR source path",
            )
            resolved_source = source_path.resolve(strict=False)
            if (
                not resolved_source.is_relative_to(Path(catalog["data_root"]))
                or not resolved_source.is_file()
            ):
                raise ValueError(
                    "configured OCR source must be an existing file inside data_root"
                )
        source_subjects = {
            PurePosixPath(item).parts[0] for item in runtime.source_relpaths
        }
        mapped_subjects = {
            subject
            for subject, mapped_policy_id in config.subject_policy_map.items()
            if mapped_policy_id == policy_id
        }
        if mapped_subjects != source_subjects:
            raise ValueError(
                "OCR policy subjects must exactly match its source_relpaths subjects"
            )
        policy_payload = dict(chunk_policies[policy_id])
        extraction = dict(policy_payload["extraction"])
        pdf_ocr = dict(extraction["pdf_ocr"])
        pdf_ocr["binary_path"] = binary_path
        pdf_ocr["tessdata_dir"] = tessdata_dir
        extraction["pdf_ocr"] = pdf_ocr
        policy_payload["extraction"] = extraction
        chunk_policies[policy_id] = policy_payload
    payload["catalog"] = catalog
    payload["storage"] = storage
    payload["chunk_policies"] = chunk_policies
    return RagIndexConfig.model_validate(payload)


__all__ = [
    "AtomicBlockPolicyConfig",
    "Bm25Config",
    "CatalogConfig",
    "ChunkPolicyConfig",
    "CleaningPolicyConfig",
    "EmbeddingConfig",
    "ExtractionPolicyConfig",
    "OcrExtractionPolicyConfig",
    "PageAssemblyPolicyConfig",
    "ProviderRoutingConfig",
    "RagIndexConfig",
    "RecursiveChunkConfig",
    "RerankerConfig",
    "RetryConfig",
    "RetrievalConfig",
    "StorageConfig",
    "StructurePolicyConfig",
    "TesseractLanguageAssetConfig",
    "TesseractOcrPolicyConfig",
    "chunk_policy_manifest_payload",
    "compute_chunk_policy_id",
    "load_rag_index_config",
    "resolve_rag_index_config_paths",
]
