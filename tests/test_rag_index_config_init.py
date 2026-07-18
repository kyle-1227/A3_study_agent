from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
import pytest
import yaml

from scripts.init_rag_index_config import (
    _parser,
    main,
    portable_runtime_config_from_source,
    write_portable_runtime_config,
)
from src.config.rag_index_config import (
    ChunkPolicyConfig,
    compute_chunk_policy_id,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child.project_paths import ProjectPathError
from src.rag.parent_child.tesseract_ocr import (
    compute_tesseract_runtime_manifest_sha256,
)
from src.rag.parent_child.tokenizer import resolve_jieba_runtime_identity
from src.rag.subject_catalog import SubjectPolicyMapError


def _policy_payload(*, short_unit_chars: int = 300) -> dict[str, object]:
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
            "header_top_lines": 1,
            "footer_bottom_lines": 1,
            "repeated_line_min_pages": 2,
            "repeated_line_min_ratio": 0.5,
            "collapse_blank_lines": True,
            "paragraph_deduplication": False,
        },
        "structure": {
            "detector_version": "structure_detector_v1",
            "pattern_set_version": "patterns_v1",
            "merge_version": "merge_v1",
            "short_unit_chars": short_unit_chars,
            "major_boundary_levels": [1, 2],
        },
        "atomic_blocks": {
            "policy_version": "atomic_v1",
            "protected_types": ["code", "table", "formula", "list"],
            "hard_max_chars": 3200,
        },
        "parent": {
            "algorithm_version": "span_recursive_v1",
            "size": 1600,
            "overlap": 100,
            "separators": ["\n\n", "\n", "。", " "],
            "length_policy": "unicode_codepoints",
            "whitespace_policy": "preserve",
        },
        "child": {
            "algorithm_version": "span_recursive_v1",
            "size": 400,
            "overlap": 50,
            "separators": ["\n\n", "\n", "。", " "],
            "length_policy": "unicode_codepoints",
            "whitespace_policy": "preserve",
        },
        "metadata_contract_version": "parent_child_metadata_v1",
    }


def _write_policy(path: Path, *, short_unit_chars: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            _policy_payload(short_unit_chars=short_unit_chars),
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    source = project_root / "data" / "math" / "notes.txt"
    source.parent.mkdir(parents=True)
    source.write_text("limits", encoding="utf-8")
    policy_path = _write_policy(project_root / "policies" / "standard.yaml")
    return project_root, policy_path


def _arguments(project_root: Path, policy_path: Path) -> list[str]:
    return [
        "--project-root",
        str(project_root),
        "--output",
        "config/rag/index.local.yaml",
        "--schema-version",
        "rag_index_config_v1",
        "--data-root",
        "data",
        "--supported-extensions",
        '[".txt"]',
        "--excluded-exact-names",
        '["evaluation"]',
        "--excluded-prefixes",
        "[]",
        "--exclude-hidden",
        "true",
        "--exclude-cache-directories",
        "true",
        "--cache-directory-names",
        '["__pycache__", ".cache"]',
        "--exclude-unclassified",
        "true",
        "--unclassified-directory-name",
        "unclassified",
        "--exclude-needs-ocr",
        "true",
        "--needs-ocr-directory-name",
        "_needs_ocr",
        "--normalization-version",
        "subject_id_v1",
        "--catalog-symlink-policy",
        "reject",
        "--index-root",
        "indexes",
        "--registry-path",
        "indexes/generation_registry.sqlite",
        "--collection-name",
        "a3_children",
        "--parent-store-schema-version",
        "parent_store_v1",
        "--registry-schema-version",
        "generation_registry_v1",
        "--owner-marker-schema-version",
        "generation_owner_v1",
        "--registry-busy-timeout-seconds",
        "5.0",
        "--parent-store-busy-timeout-seconds",
        "5.0",
        "--retention-generations",
        "3",
        "--embedding-provider",
        "test_embedding_provider",
        "--embedding-protocol",
        "openai_embeddings_v1",
        "--embedding-model",
        "test_embedding_model",
        "--embedding-response-model",
        "test_embedding_model",
        "--embedding-base-url",
        "https://embedding.invalid/v1",
        "--embedding-endpoint-path",
        "/embeddings",
        "--embedding-api-key-env",
        "TEST_EMBEDDING_API_KEY",
        "--embedding-timeout-seconds",
        "10.0",
        "--embedding-retry-max-attempts",
        "2",
        "--embedding-retry-initial-backoff-seconds",
        "0.1",
        "--embedding-retry-max-backoff-seconds",
        "1.0",
        "--embedding-retry-multiplier",
        "2.0",
        "--embedding-batch-size",
        "20",
        "--embedding-max-in-flight-batches",
        "1",
        "--embedding-expected-dimension",
        "1024",
        "--embedding-distance-metric",
        "cosine",
        "--embedding-normalization-contract",
        "unit_vector_v1",
        "--embedding-document-input-type",
        "document",
        "--embedding-query-input-type",
        "query",
        "--embedding-no-input-type-field",
        "--embedding-no-provider-routing",
        "--reranker-provider",
        "test_reranker_provider",
        "--reranker-model",
        "test_reranker_model",
        "--reranker-response-model",
        "test_reranker_model",
        "--reranker-base-url",
        "https://reranker.invalid/v1",
        "--reranker-endpoint-path",
        "/rerank",
        "--reranker-api-key-env",
        "TEST_RERANKER_API_KEY",
        "--reranker-timeout-seconds",
        "10.0",
        "--reranker-retry-max-attempts",
        "2",
        "--reranker-retry-initial-backoff-seconds",
        "0.1",
        "--reranker-retry-max-backoff-seconds",
        "1.0",
        "--reranker-retry-multiplier",
        "2.0",
        "--reranker-batch-size",
        "40",
        "--reranker-recovery-mode",
        "strict_bisect_v1",
        "--reranker-recovery-max-total-requests",
        "9",
        "--reranker-recovery-max-split-depth",
        "2",
        "--reranker-recovery-min-batch-size",
        "5",
        "--reranker-recovery-max-response-bytes",
        "1048576",
        "--reranker-protocol",
        "ranked_index_scores_v1",
        "--reranker-score-min",
        "0.0",
        "--reranker-score-max",
        "1.0",
        "--reranker-no-provider-routing",
        "--bm25-tokenizer",
        "jieba_builtin_precise_v1",
        "--bm25-artifact-format",
        "jsonl",
        "--chunk-policy",
        f"standard={policy_path.relative_to(project_root).as_posix()}",
        "--subject-policy",
        "math=standard",
        "--vector-top-k",
        "20",
        "--bm25-top-k",
        "20",
        "--rrf-k",
        "60",
        "--vector-weight",
        "1.0",
        "--bm25-weight",
        "1.0",
        "--reranker-top-n",
        "20",
        "--unique-parent-top-k",
        "3",
        "--max-children-per-parent",
        "1",
        "--max-parents-per-source",
        "1",
        "--parent-support-lambda",
        "0.25",
        "--full-parent-max-chars",
        "1600",
        "--hit-window-chars-per-side",
        "400",
        "--context-item-max-chars",
        "2400",
        "--judge-preview-max-chars",
        "400",
        "--multi-subject-per-subject-top-k",
        "3",
        "--multi-subject-max-parents",
        "3",
        "--cross-branch-rrf-k",
        "60",
        "--subject-coverage-quota",
        "1",
    ]


def test_init_generates_stable_strict_config_without_reading_api_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, policy_path = _project(tmp_path)
    arguments = _arguments(project_root, policy_path)
    monkeypatch.setenv("TEST_EMBEDDING_API_KEY", "secret-must-not-be-written")

    assert main(arguments) == 0

    output = project_root / "config" / "rag" / "index.local.yaml"
    first = output.read_bytes()
    config = load_rag_index_config(output)
    identity = resolve_jieba_runtime_identity()
    expected_policy = ChunkPolicyConfig.model_validate(_policy_payload())
    expected_policy_id = compute_chunk_policy_id(expected_policy)

    assert config.subject_policy_map == {"math": expected_policy_id}
    assert set(config.chunk_policies) == {expected_policy_id}
    assert config.catalog.data_root == (project_root / "data").resolve()
    assert config.bm25.tokenizer_version == identity.tokenizer_version
    assert config.bm25.dictionary_hash == identity.dictionary_hash
    assert b"secret-must-not-be-written" not in first
    assert b"TEST_EMBEDDING_API_KEY" in first

    assert main([*arguments, "--overwrite"]) == 0
    assert output.read_bytes() == first
    with pytest.raises(FileExistsError):
        main(arguments)


def test_legacy_chunk_policy_id_is_byte_stable() -> None:
    policy = ChunkPolicyConfig.model_validate(_policy_payload())
    assert compute_chunk_policy_id(policy) == (
        "87e4eb5997cf6525fb839f55914af68b0f34a484d4c67c55f24e0f8b9311be72"
    )


def test_init_accepts_explicit_openrouter_embedding_protocol(
    tmp_path: Path,
) -> None:
    project_root, policy_path = _project(tmp_path)
    arguments = _arguments(project_root, policy_path)
    arguments[arguments.index("--embedding-provider") + 1] = "openrouter"
    arguments[arguments.index("--embedding-protocol") + 1] = "openrouter_embeddings_v1"
    routing_index = arguments.index("--embedding-no-provider-routing")
    arguments[routing_index : routing_index + 1] = [
        "--embedding-provider-routing",
        '{"order":["parasail"],"allow_fallbacks":false}',
    ]

    assert main(arguments) == 0

    config = load_rag_index_config(project_root / "config" / "rag" / "index.local.yaml")
    assert config.embedding.provider == "openrouter"
    assert config.embedding.protocol == "openrouter_embeddings_v1"
    assert config.embedding.input_type_field is None
    assert config.embedding.provider_routing is not None
    assert config.embedding.provider_routing.order == ("parasail",)


def test_init_assigns_distinct_explicit_policies_to_catalog_subjects(
    tmp_path: Path,
) -> None:
    project_root, policy_path = _project(tmp_path)
    python_source = project_root / "data" / "python" / "notes.txt"
    python_source.parent.mkdir(parents=True)
    python_source.write_text("functions", encoding="utf-8")
    secondary_path = _write_policy(
        project_root / "policies" / "secondary.yaml",
        short_unit_chars=301,
    )

    assert (
        main(
            [
                *_arguments(project_root, policy_path),
                "--chunk-policy",
                f"secondary={secondary_path.relative_to(project_root).as_posix()}",
                "--subject-policy",
                "python=secondary",
            ]
        )
        == 0
    )
    config = load_rag_index_config(project_root / "config" / "rag" / "index.local.yaml")

    assert tuple(config.subject_policy_map) == ("math", "python")
    assert config.subject_policy_map["math"] != config.subject_policy_map["python"]


def test_init_fails_when_catalog_subject_lacks_a_policy_mapping(tmp_path: Path) -> None:
    project_root, policy_path = _project(tmp_path)
    python_source = project_root / "data" / "python" / "notes.txt"
    python_source.parent.mkdir(parents=True)
    python_source.write_text("functions", encoding="utf-8")

    with pytest.raises(SubjectPolicyMapError, match="missing policy"):
        main(_arguments(project_root, policy_path))


def test_init_rejects_project_escape_and_symlink_policy_fragment(
    tmp_path: Path,
) -> None:
    project_root, policy_path = _project(tmp_path)
    escaping = _arguments(project_root, policy_path)
    output_index = escaping.index("config/rag/index.local.yaml")
    escaping[output_index] = "../outside.yaml"
    with pytest.raises(ProjectPathError, match="inside project_root"):
        main(escaping)

    linked_policy = project_root / "policies" / "linked.yaml"
    try:
        linked_policy.symlink_to(policy_path)
    except OSError:
        pytest.skip("symlink creation is unavailable in this environment")
    symlinked = _arguments(project_root, policy_path)
    policy_index = symlinked.index(
        f"standard={policy_path.relative_to(project_root).as_posix()}"
    )
    symlinked[policy_index] = "standard=policies/linked.yaml"
    with pytest.raises(ProjectPathError, match="symlink or reparse"):
        main(symlinked)


def test_init_cli_requires_explicit_provider_and_policy_inputs() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args([])


def test_init_cli_requires_explicit_embedding_concurrency(tmp_path: Path) -> None:
    project_root, policy_path = _project(tmp_path)
    arguments = _arguments(project_root, policy_path)
    index = arguments.index("--embedding-max-in-flight-batches")
    del arguments[index : index + 2]

    with pytest.raises(SystemExit):
        main(arguments)


def test_init_cli_requires_explicit_provider_routing(tmp_path: Path) -> None:
    project_root, policy_path = _project(tmp_path)
    arguments = _arguments(project_root, policy_path)
    arguments.remove("--embedding-no-provider-routing")

    with pytest.raises(SystemExit):
        main(arguments)


def test_portable_runtime_config_round_trips_absolute_ocr_paths(
    tmp_path: Path,
) -> None:
    project_root, policy_path = _project(tmp_path)
    assert main(_arguments(project_root, policy_path)) == 0
    source_config_path = project_root / "config" / "rag" / "index.local.yaml"
    payload = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    runtime_root = project_root / ".runtime_tools" / "tesseract"
    tessdata = runtime_root / "tessdata"
    (tessdata / "configs").mkdir(parents=True)
    binary = runtime_root / "tesseract.exe"
    binary.write_bytes(b"portable-binary")
    (runtime_root / "engine.dll").write_bytes(b"portable-dll")
    (tessdata / "configs" / "tsv").write_text(
        "tessedit_create_tsv 1\n",
        encoding="utf-8",
    )
    language_hashes: list[dict[str, str]] = []
    for language in ("chi_sim", "eng"):
        traineddata = tessdata / f"{language}.traineddata"
        traineddata.write_bytes(f"portable-{language}".encode())
        language_hashes.append(
            {
                "language": language,
                "traineddata_sha256": hashlib.sha256(
                    traineddata.read_bytes()
                ).hexdigest(),
            }
        )
    ocr_source = project_root / "data" / "math" / "book.pdf"
    ocr_source.write_bytes(b"%PDF-portable-test")

    chunk_policies = payload["chunk_policies"]
    assert isinstance(chunk_policies, dict)
    policy_payload = dict(next(iter(chunk_policies.values())))
    policy_payload["extraction"] = {
        "algorithm_version": "page_extract_tesseract_v2",
        "pdf_extraction_method": "configured_pdf_text",
        "text_extraction_method": "configured_utf8_text",
        "pdf_ocr": {
            "schema_version": "tesseract_ocr_policy_v1",
            "engine_protocol": "tesseract_cli_tsv_v1",
            "extraction_method": "tesseract_cli_tsv_v1",
            "source_relpaths": ["math/book.pdf"],
            "binary_path": str(binary.resolve()),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "runtime_manifest_sha256": (
                compute_tesseract_runtime_manifest_sha256(binary, tessdata)
            ),
            "expected_version": "5.5.0",
            "renderer_protocol": "pymupdf_pixmap_png_v1",
            "pymupdf_version": fitz.VersionBind,
            "mupdf_version": fitz.VersionFitz,
            "tessdata_dir": str(tessdata.resolve()),
            "language_assets": language_hashes,
            "dpi": 300,
            "render_colorspace": "rgb",
            "render_alpha": False,
            "render_annotations": True,
            "oem": 1,
            "psm": 3,
            "thread_limit": 1,
            "timeout_seconds": 30.0,
            "output_format": "tsv_lines_v1",
            "empty_page_policy": "allow_empty_physical_page_v1",
        },
    }
    policy = ChunkPolicyConfig.model_validate(policy_payload)
    policy_id = compute_chunk_policy_id(policy)
    payload["chunk_policies"] = {policy_id: policy.model_dump(mode="json")}
    payload["subject_policy_map"] = {"math": policy_id}
    catalog = payload["catalog"]
    assert isinstance(catalog, dict)
    catalog["supported_extensions"] = [".pdf", ".txt"]
    source_config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

    runtime = portable_runtime_config_from_source(
        project_root=project_root,
        source_config_path=source_config_path.relative_to(project_root),
        data_root=Path("data"),
        index_root=Path("indexes"),
        registry_path=Path("generation_registry.sqlite"),
    )
    output = write_portable_runtime_config(
        project_root=project_root,
        output_path=Path("config/rag/index.runtime.yaml"),
        config=runtime,
        overwrite=False,
    )
    written_payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    written_policy = next(iter(written_payload["chunk_policies"].values()))
    written_ocr = written_policy["extraction"]["pdf_ocr"]
    assert written_ocr["binary_path"] == ".runtime_tools/tesseract/tesseract.exe"
    assert written_ocr["tessdata_dir"] == ".runtime_tools/tesseract/tessdata"
    assert not Path(written_payload["catalog"]["data_root"]).is_absolute()
    assert not Path(written_payload["storage"]["index_root"]).is_absolute()
    assert resolve_rag_index_config_paths(
        load_rag_index_config(output),
        project_root=project_root,
    ) == resolve_rag_index_config_paths(runtime, project_root=project_root)
