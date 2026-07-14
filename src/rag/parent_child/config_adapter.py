"""Fail-fast conversion from strict index config to runtime chunk contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.config.rag_index_config import (
    ChunkPolicyConfig,
    OcrExtractionPolicyConfig,
    RagIndexConfig,
    chunk_policy_manifest_payload,
    compute_chunk_policy_id,
)
from src.rag.parent_child._storage_io import canonical_json_bytes, sha256_bytes
from src.rag.parent_child.ids import make_loader_policy_fingerprint
from src.rag.parent_child.manifests import PolicyManifest
from src.rag.parent_child.models import (
    PageAwareLoaderConfig,
    ParentChildPolicy,
    TesseractLanguageAsset,
    TesseractOcrRuntimeConfig,
)
from src.rag.parent_child.tesseract_ocr import validate_tesseract_runtime


class PolicyAdapterError(RuntimeError):
    """Configured policy cannot be executed by the V1 implementation."""


class ResolvedChunkPolicy(BaseModel):
    """Coherent manifest, loader, and splitter contracts for one policy ID."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    policy_manifest: PolicyManifest
    loader_config: PageAwareLoaderConfig
    parent_child_policy: ParentChildPolicy


def _require_supported_algorithms(policy: ChunkPolicyConfig) -> None:
    expected_extraction = (
        "page_extract_tesseract_v2"
        if isinstance(policy.extraction, OcrExtractionPolicyConfig)
        else "page_extract_v1"
    )
    supported = {
        "extraction.algorithm_version": (
            policy.extraction.algorithm_version,
            expected_extraction,
        ),
        "page_assembly.algorithm_version": (
            policy.page_assembly.algorithm_version,
            "page_assembly_v1",
        ),
        "cleaning.algorithm_version": (
            policy.cleaning.algorithm_version,
            "page_clean_v2",
        ),
        "parent.algorithm_version": (
            policy.parent.algorithm_version,
            "span_recursive_v1",
        ),
        "child.algorithm_version": (
            policy.child.algorithm_version,
            "span_recursive_v1",
        ),
        "parent.length_policy": (
            policy.parent.length_policy,
            "unicode_codepoints",
        ),
        "child.length_policy": (
            policy.child.length_policy,
            "unicode_codepoints",
        ),
        "parent.whitespace_policy": (
            policy.parent.whitespace_policy,
            "preserve",
        ),
        "child.whitespace_policy": (
            policy.child.whitespace_policy,
            "preserve",
        ),
        "metadata_contract_version": (
            policy.metadata_contract_version,
            "parent_child_metadata_v1",
        ),
    }
    mismatches = tuple(
        name for name, (actual, expected) in supported.items() if actual != expected
    )
    if mismatches:
        raise PolicyAdapterError(
            "unsupported output-affecting policy contracts: " + ", ".join(mismatches)
        )

    supported_atomic_types = {"code", "table", "formula", "list"}
    unknown_atomic_types = set(policy.atomic_blocks.protected_types) - (
        supported_atomic_types
    )
    if unknown_atomic_types:
        raise PolicyAdapterError(
            "unsupported atomic block types: " + ", ".join(sorted(unknown_atomic_types))
        )


def resolve_chunk_policy(
    index_config: RagIndexConfig,
    policy_id: str,
) -> ResolvedChunkPolicy:
    """Resolve a declared policy and revalidate its canonical manifest identity."""

    try:
        configured = index_config.chunk_policies[policy_id]
    except KeyError as exc:
        raise PolicyAdapterError("requested policy_id is not configured") from exc
    expected_policy_id = compute_chunk_policy_id(configured)
    if expected_policy_id != policy_id:
        raise PolicyAdapterError("configured policy_id failed canonical recomputation")
    _require_supported_algorithms(configured)

    manifest_payload = chunk_policy_manifest_payload(configured)
    policy_manifest = PolicyManifest.model_validate(
        {**manifest_payload, "policy_id": policy_id}
    )
    cleaning_policy_id = sha256_bytes(
        canonical_json_bytes(configured.cleaning.model_dump(mode="json"))
    )
    pdf_ocr: TesseractOcrRuntimeConfig | None = None
    if isinstance(configured.extraction, OcrExtractionPolicyConfig):
        runtime = configured.extraction.pdf_ocr
        pdf_ocr = TesseractOcrRuntimeConfig(
            schema_version="tesseract_ocr_runtime_v1",
            engine_protocol=runtime.engine_protocol,
            extraction_method=runtime.extraction_method,
            source_relpaths=runtime.source_relpaths,
            binary_path=runtime.binary_path,
            binary_sha256=runtime.binary_sha256,
            runtime_manifest_sha256=runtime.runtime_manifest_sha256,
            expected_version=runtime.expected_version,
            renderer_protocol=runtime.renderer_protocol,
            pymupdf_version=runtime.pymupdf_version,
            mupdf_version=runtime.mupdf_version,
            tessdata_dir=runtime.tessdata_dir,
            language_assets=tuple(
                TesseractLanguageAsset(
                    language=asset.language,
                    traineddata_sha256=asset.traineddata_sha256,
                )
                for asset in runtime.language_assets
            ),
            dpi=runtime.dpi,
            render_colorspace=runtime.render_colorspace,
            render_alpha=runtime.render_alpha,
            render_annotations=runtime.render_annotations,
            oem=runtime.oem,
            psm=runtime.psm,
            thread_limit=runtime.thread_limit,
            timeout_seconds=runtime.timeout_seconds,
            output_format=runtime.output_format,
            empty_page_policy=runtime.empty_page_policy,
        )
    loader = PageAwareLoaderConfig(
        schema_version="page_aware_loader_policy_v1",
        extraction_algorithm_version=configured.extraction.algorithm_version,
        page_assembly_algorithm_version=configured.page_assembly.algorithm_version,
        cleaning_algorithm_version=configured.cleaning.algorithm_version,
        cleaning_policy_id=cleaning_policy_id,
        nul_character_policy=configured.cleaning.nul_character_policy,
        page_separator=configured.page_assembly.page_separator,
        normalize_newlines=configured.cleaning.normalize_newlines,
        strip_trailing_whitespace=configured.cleaning.strip_trailing_whitespace,
        strip_outer_blank_lines=configured.cleaning.strip_outer_blank_lines,
        header_top_lines=configured.cleaning.header_top_lines,
        footer_bottom_lines=configured.cleaning.footer_bottom_lines,
        repeated_line_min_pages=configured.cleaning.repeated_line_min_pages,
        repeated_line_min_ratio=configured.cleaning.repeated_line_min_ratio,
        collapse_blank_lines=configured.cleaning.collapse_blank_lines,
        paragraph_deduplication=configured.cleaning.paragraph_deduplication,
        supported_extensions=index_config.catalog.supported_extensions,
        pdf_extraction_method=configured.extraction.pdf_extraction_method,
        text_extraction_method=configured.extraction.text_extraction_method,
        pdf_ocr=pdf_ocr,
    )
    protected = set(configured.atomic_blocks.protected_types)
    runtime_policy = ParentChildPolicy(
        schema_version="parent_child_policy_v1",
        canonicalization_version="canonical_json_v1",
        id_algorithm_version="parent_child_id_v1",
        metadata_contract_version="parent_child_metadata_v1",
        policy_id=policy_id,
        structure_detector_version=configured.structure.detector_version,
        structure_pattern_set_version=configured.structure.pattern_set_version,
        structure_merge_version=configured.structure.merge_version,
        short_unit_chars=configured.structure.short_unit_chars,
        parent_split_algorithm="span_recursive_v1",
        child_split_algorithm="span_recursive_v1",
        loader_policy_id=make_loader_policy_fingerprint(loader),
        cleaning_policy_id=cleaning_policy_id,
        parent_size=configured.parent.size,
        parent_overlap=configured.parent.overlap,
        parent_hard_max=configured.atomic_blocks.hard_max_chars,
        child_size=configured.child.size,
        child_overlap=configured.child.overlap,
        child_hard_max=configured.atomic_blocks.hard_max_chars,
        parent_separators=configured.parent.separators,
        child_separators=configured.child.separators,
        major_section_max_level=max(configured.structure.major_boundary_levels),
        atomic_fenced_code_blocks="code" in protected,
        atomic_markdown_tables="table" in protected,
        atomic_list_blocks="list" in protected,
        atomic_display_math="formula" in protected,
    )
    return ResolvedChunkPolicy(
        policy_manifest=policy_manifest,
        loader_config=loader,
        parent_child_policy=runtime_policy,
    )


def resolve_subject_chunk_policy(
    index_config: RagIndexConfig,
    subject: str,
) -> ResolvedChunkPolicy:
    """Resolve an exact configured subject; unknown subjects never widen scope."""

    try:
        policy_id = index_config.subject_policy_map[subject]
    except KeyError as exc:
        raise PolicyAdapterError("subject has no exact configured policy") from exc
    return resolve_chunk_policy(index_config, policy_id)


def validate_configured_ocr_runtimes(index_config: RagIndexConfig) -> None:
    """Validate every unique OCR runtime before any corpus processing."""

    validated: set[
        tuple[
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            tuple[tuple[str, str], ...],
        ]
    ] = set()
    for policy_id in sorted(index_config.chunk_policies):
        runtime = resolve_chunk_policy(index_config, policy_id).loader_config.pdf_ocr
        if runtime is None:
            continue
        identity = (
            str(runtime.binary_path),
            str(runtime.tessdata_dir),
            runtime.binary_sha256,
            runtime.runtime_manifest_sha256,
            runtime.expected_version,
            runtime.renderer_protocol,
            runtime.pymupdf_version,
            runtime.mupdf_version,
            tuple(
                (asset.language, asset.traineddata_sha256)
                for asset in runtime.language_assets
            ),
        )
        if identity in validated:
            continue
        validate_tesseract_runtime(runtime)
        validated.add(identity)


def validate_configured_ocr_source_inventory(
    index_config: RagIndexConfig, active_source_relpaths: tuple[str, ...]
) -> None:
    """Require every configured OCR source to be active in SubjectCatalog."""

    expected = {
        source_relpath
        for policy_id in sorted(index_config.chunk_policies)
        for runtime in (
            resolve_chunk_policy(index_config, policy_id).loader_config.pdf_ocr,
        )
        if runtime is not None
        for source_relpath in runtime.source_relpaths
    }
    missing = expected - set(active_source_relpaths)
    if missing:
        raise PolicyAdapterError(
            "configured OCR source inventory is not active in SubjectCatalog: "
            + ", ".join(sorted(missing))
        )


__all__ = [
    "PolicyAdapterError",
    "ResolvedChunkPolicy",
    "resolve_chunk_policy",
    "resolve_subject_chunk_policy",
    "validate_configured_ocr_runtimes",
    "validate_configured_ocr_source_inventory",
]
