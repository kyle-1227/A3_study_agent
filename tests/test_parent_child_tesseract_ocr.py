from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import fitz
import pytest
from pydantic import ValidationError

from src.config.rag_index_config import CatalogConfig, TesseractOcrPolicyConfig
from src.rag.parent_child import config_adapter
from src.rag.parent_child import loader as loader_module
from src.rag.parent_child import tesseract_ocr
from src.rag.parent_child.exceptions import (
    OcrProtocolError,
    OcrRuntimeIdentityError,
)
from src.rag.parent_child.ids import make_loader_policy_fingerprint
from src.rag.parent_child.models import (
    PageAwareLoaderConfig,
    SourceEntry,
    TesseractLanguageAsset,
    TesseractOcrRuntimeConfig,
)
from src.rag.parent_child.tesseract_ocr import (
    TesseractCliOcr,
    compute_tesseract_runtime_manifest_sha256,
    parse_tesseract_tsv,
    validate_tesseract_runtime,
)
from src.rag.subject_catalog import SubjectCatalog


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_config(tmp_path: Path) -> TesseractOcrRuntimeConfig:
    runtime_root = tmp_path / "runtime"
    tessdata = runtime_root / "tessdata"
    (tessdata / "configs").mkdir(parents=True)
    binary = runtime_root / "tesseract.exe"
    binary.write_bytes(b"strict-test-binary")
    (runtime_root / "engine.dll").write_bytes(b"strict-test-dll")
    (tessdata / "configs" / "tsv").write_text(
        "tessedit_create_tsv 1\n",
        encoding="utf-8",
    )
    chi_sim = tessdata / "chi_sim.traineddata"
    eng = tessdata / "eng.traineddata"
    chi_sim.write_bytes(b"strict-chi-sim")
    eng.write_bytes(b"strict-eng")
    return TesseractOcrRuntimeConfig(
        schema_version="tesseract_ocr_runtime_v1",
        engine_protocol="tesseract_cli_tsv_v1",
        extraction_method="tesseract_cli_tsv_v1",
        source_relpaths=("computer/book.pdf",),
        binary_path=binary.resolve(),
        binary_sha256=_sha256(binary),
        runtime_manifest_sha256=compute_tesseract_runtime_manifest_sha256(
            binary.resolve(), tessdata.resolve()
        ),
        expected_version="5.5.0",
        renderer_protocol="pymupdf_pixmap_png_v1",
        pymupdf_version=fitz.VersionBind,
        mupdf_version=fitz.VersionFitz,
        tessdata_dir=tessdata.resolve(),
        language_assets=(
            TesseractLanguageAsset(
                language="chi_sim",
                traineddata_sha256=_sha256(chi_sim),
            ),
            TesseractLanguageAsset(
                language="eng",
                traineddata_sha256=_sha256(eng),
            ),
        ),
        dpi=300,
        render_colorspace="rgb",
        render_alpha=False,
        render_annotations=True,
        oem=1,
        psm=3,
        thread_limit=1,
        timeout_seconds=5.0,
        output_format="tsv_lines_v1",
        empty_page_policy="allow_empty_physical_page_v1",
    )


def _valid_identity_output(
    command: tuple[str, ...], *, timeout_seconds: float
) -> bytes:
    assert timeout_seconds > 0
    if command[-1] == "--version":
        return b"tesseract v5.5.0\r\n"
    if command[-1] == "--list-langs":
        return (
            b'List of available languages in "strict-test" (2):\r\nchi_sim\r\neng\r\n'
        )
    raise AssertionError("unexpected identity command")


def _tsv_row(
    *,
    line: int,
    word: int,
    text: str,
    level: int = 5,
    page: int = 1,
) -> str:
    return "\t".join(
        (
            str(level),
            str(page),
            "1",
            "1",
            str(line),
            str(word),
            "0",
            "0",
            "10",
            "10",
            "95.0",
            text,
        )
    )


def _tsv_payload(*rows: str, include_page_row: bool = True) -> bytes:
    header = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\t"
        "top\twidth\theight\tconf\ttext"
    )
    page_row = "\t".join(
        ("1", "1", "0", "0", "0", "0", "0", "0", "100", "100", "-1", "")
    )
    body = (page_row, *rows) if include_page_row else rows
    return ("\n".join((header, *body)) + "\n").encode()


def test_runtime_manifest_and_all_assets_are_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config(tmp_path)
    monkeypatch.setattr(tesseract_ocr, "_identity_command", _valid_identity_output)

    validate_tesseract_runtime(config)
    assert compute_tesseract_runtime_manifest_sha256(
        config.binary_path,
        config.tessdata_dir,
    ) == compute_tesseract_runtime_manifest_sha256(
        config.binary_path,
        config.tessdata_dir,
    )

    (config.binary_path.parent / "engine.dll").write_bytes(b"changed-dll")
    with pytest.raises(OcrRuntimeIdentityError, match="runtime manifest"):
        validate_tesseract_runtime(config)


def test_runtime_rejects_binary_traineddata_and_version_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config(tmp_path)
    monkeypatch.setattr(tesseract_ocr, "_identity_command", _valid_identity_output)

    config.binary_path.write_bytes(b"changed-binary")
    with pytest.raises(OcrRuntimeIdentityError, match="binary SHA-256"):
        validate_tesseract_runtime(config)

    config = _runtime_config(tmp_path / "traineddata")
    (config.tessdata_dir / "chi_sim.traineddata").write_bytes(b"changed-language")
    with pytest.raises(OcrRuntimeIdentityError, match="traineddata SHA-256"):
        validate_tesseract_runtime(config)

    config = _runtime_config(tmp_path / "version")

    def wrong_version(command: tuple[str, ...], *, timeout_seconds: float) -> bytes:
        if command[-1] == "--version":
            return b"tesseract v5.4.0\n"
        return _valid_identity_output(command, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(tesseract_ocr, "_identity_command", wrong_version)
    with pytest.raises(OcrRuntimeIdentityError, match="version mismatch"):
        validate_tesseract_runtime(config)

    renderer_drift = TesseractOcrRuntimeConfig.model_validate(
        {
            **config.model_dump(mode="python"),
            "pymupdf_version": "0.0.0",
        }
    )
    with pytest.raises(OcrRuntimeIdentityError, match="PyMuPDF version"):
        validate_tesseract_runtime(renderer_drift)


@pytest.mark.parametrize(
    "language_output, error_pattern",
    (
        (b"chi_sim\neng\n", "header"),
        (
            b'List of available languages in "strict-test" (3):\nchi_sim\neng\n',
            "count",
        ),
        (
            b'List of available languages in "strict-test" (2):\nchi_sim\nchi_sim\n',
            "duplicates",
        ),
    ),
)
def test_runtime_rejects_malformed_language_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language_output: bytes,
    error_pattern: str,
) -> None:
    config = _runtime_config(tmp_path)

    def identity_output(command: tuple[str, ...], *, timeout_seconds: float) -> bytes:
        if command[-1] == "--list-langs":
            return language_output
        return _valid_identity_output(command, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(tesseract_ocr, "_identity_command", identity_output)
    with pytest.raises(OcrRuntimeIdentityError, match=error_pattern):
        validate_tesseract_runtime(config)


def test_runtime_manifest_rejects_linked_or_reparse_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config(tmp_path)
    original = tesseract_ocr._is_link_or_reparse

    def identify_reparse(path: Path) -> bool:
        if path.name == "configs":
            return True
        return original(path)

    monkeypatch.setattr(tesseract_ocr, "_is_link_or_reparse", identify_reparse)
    with pytest.raises(OcrRuntimeIdentityError, match="config directory"):
        compute_tesseract_runtime_manifest_sha256(
            config.binary_path,
            config.tessdata_dir,
        )


def test_tsv_parser_reconstructs_lines_and_rejects_duplicate_word_order() -> None:
    payload = _tsv_payload(
        _tsv_row(line=1, word=1, text="Python"),
        _tsv_row(line=1, word=2, text="generator"),
        _tsv_row(line=2, word=1, text="memory"),
    )
    assert parse_tesseract_tsv(payload) == "Python generator\nmemory"

    chinese = _tsv_payload(
        _tsv_row(line=1, word=1, text="生成"),
        _tsv_row(line=1, word=2, text="器"),
        _tsv_row(line=1, word=3, text="更"),
        _tsv_row(line=1, word=4, text="节省"),
        _tsv_row(line=1, word=5, text="内存"),
        _tsv_row(line=1, word=6, text="。"),
    )
    assert parse_tesseract_tsv(chinese) == "生成器更节省内存。"

    duplicate = _tsv_payload(
        _tsv_row(line=1, word=1, text="first"),
        _tsv_row(line=1, word=1, text="duplicate"),
    )
    with pytest.raises(OcrProtocolError, match="word ordering"):
        parse_tesseract_tsv(duplicate)

    with pytest.raises(OcrProtocolError, match="level"):
        parse_tesseract_tsv(
            _tsv_payload(_tsv_row(line=1, word=1, text="invalid", level=6))
        )
    with pytest.raises(OcrProtocolError, match="hierarchy"):
        parse_tesseract_tsv(
            _tsv_payload(_tsv_row(line=1, word=1, text="invalid", page=2))
        )
    with pytest.raises(OcrProtocolError, match="single-page row"):
        parse_tesseract_tsv(
            _tsv_payload(
                _tsv_row(line=1, word=1, text="invalid"),
                include_page_row=False,
            )
        )


def test_tesseract_timeout_is_explicit_and_has_no_success_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config(tmp_path)
    monkeypatch.setattr(tesseract_ocr, "validate_tesseract_runtime", lambda _: None)
    monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "must-not-reach-subprocess")
    observed_environment: dict[str, str] | None = None

    def timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal observed_environment
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_environment = environment
        raise subprocess.TimeoutExpired(cmd="tesseract", timeout=1.0)

    monkeypatch.setattr(tesseract_ocr.subprocess, "run", timeout)
    engine = TesseractCliOcr(config)
    with pytest.raises(OcrProtocolError, match="request failed"):
        engine.recognize_png(b"real-png-placeholder")
    assert observed_environment is not None
    assert "RAG_EMBEDDING_API_KEY" not in observed_environment
    assert observed_environment["OMP_THREAD_LIMIT"] == "1"


def test_loader_does_not_fallback_when_configured_ocr_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_config(tmp_path)
    data_root = tmp_path / "data"
    source = data_root / "computer" / "book.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-strict-test")
    fallback_called = False

    def fail_ocr(path: Path, config: TesseractOcrRuntimeConfig) -> tuple[str, ...]:
        assert path == source.resolve()
        assert config == runtime
        raise OcrProtocolError("configured OCR failed")

    def forbidden_fallback(path: Path) -> tuple[str, ...]:
        nonlocal fallback_called
        fallback_called = True
        return ("forbidden",)

    monkeypatch.setattr(loader_module, "extract_pdf_pages_with_tesseract", fail_ocr)
    monkeypatch.setattr(loader_module, "_extract_pdf_pages", forbidden_fallback)
    loader_config = PageAwareLoaderConfig(
        schema_version="page_aware_loader_policy_v1",
        extraction_algorithm_version="page_extract_tesseract_v2",
        page_assembly_algorithm_version="page_assembly_v1",
        cleaning_algorithm_version="page_clean_v2",
        cleaning_policy_id="a" * 64,
        nul_character_policy="replace_with_space_v1",
        page_separator="\n\f\n",
        normalize_newlines=True,
        strip_trailing_whitespace=True,
        strip_outer_blank_lines=True,
        header_top_lines=1,
        footer_bottom_lines=1,
        repeated_line_min_pages=2,
        repeated_line_min_ratio=0.5,
        collapse_blank_lines=True,
        paragraph_deduplication=False,
        supported_extensions=(".pdf", ".md", ".txt"),
        pdf_extraction_method="configured_pdf_text",
        text_extraction_method="configured_utf8_text",
        pdf_ocr=runtime,
    )
    with pytest.raises(OcrProtocolError, match="configured OCR failed"):
        loader_module.load_cleaned_source(
            SourceEntry(
                schema_version="source_entry_v1",
                source_path=source.resolve(),
                data_root=data_root.resolve(),
                subject="computer",
                doc_type="pdf",
            ),
            loader_config,
        )
    assert fallback_called is False


def test_ocr_inventory_requires_an_active_catalog_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_config(tmp_path)
    resolved = SimpleNamespace(loader_config=SimpleNamespace(pdf_ocr=runtime))
    index_config = SimpleNamespace(chunk_policies={"policy": object()})
    monkeypatch.setattr(
        config_adapter,
        "resolve_chunk_policy",
        lambda _config, _policy_id: resolved,
    )

    with pytest.raises(config_adapter.PolicyAdapterError, match="not active"):
        config_adapter.validate_configured_ocr_source_inventory(
            index_config,
            ("computer/other.pdf",),
        )
    config_adapter.validate_configured_ocr_source_inventory(
        index_config,
        ("computer/book.pdf",),
    )


def test_runtime_validation_identity_includes_runtime_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _runtime_config(tmp_path)
    second = TesseractOcrRuntimeConfig.model_validate(
        {
            **first.model_dump(mode="python"),
            "runtime_manifest_sha256": "f" * 64,
        }
    )
    by_policy = {
        "first": SimpleNamespace(loader_config=SimpleNamespace(pdf_ocr=first)),
        "second": SimpleNamespace(loader_config=SimpleNamespace(pdf_ocr=second)),
    }
    index_config = SimpleNamespace(
        chunk_policies={"first": object(), "second": object()}
    )
    monkeypatch.setattr(
        config_adapter,
        "resolve_chunk_policy",
        lambda _config, policy_id: by_policy[policy_id],
    )
    validated: list[str] = []
    monkeypatch.setattr(
        config_adapter,
        "validate_tesseract_runtime",
        lambda runtime: validated.append(runtime.runtime_manifest_sha256),
    )

    config_adapter.validate_configured_ocr_runtimes(index_config)
    assert validated == [first.runtime_manifest_sha256, second.runtime_manifest_sha256]


def test_runtime_validation_does_not_deduplicate_distinct_installations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _runtime_config(tmp_path / "first")
    second = _runtime_config(tmp_path / "second")
    by_policy = {
        "first": SimpleNamespace(loader_config=SimpleNamespace(pdf_ocr=first)),
        "second": SimpleNamespace(loader_config=SimpleNamespace(pdf_ocr=second)),
    }
    index_config = SimpleNamespace(
        chunk_policies={"first": object(), "second": object()}
    )
    monkeypatch.setattr(
        config_adapter,
        "resolve_chunk_policy",
        lambda _config, policy_id: by_policy[policy_id],
    )
    validated: list[Path] = []
    monkeypatch.setattr(
        config_adapter,
        "validate_tesseract_runtime",
        lambda runtime: validated.append(runtime.binary_path),
    )

    config_adapter.validate_configured_ocr_runtimes(index_config)
    assert validated == [first.binary_path, second.binary_path]


def test_ocr_inventory_uses_real_subject_catalog_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    active_source = data_root / "computer" / "active.pdf"
    excluded_source = data_root / "computer" / "_needs_ocr" / "book.pdf"
    active_source.parent.mkdir(parents=True)
    excluded_source.parent.mkdir(parents=True)
    active_source.write_bytes(b"active")
    excluded_source.write_bytes(b"excluded")
    catalog = SubjectCatalog(
        config=CatalogConfig(
            data_root=data_root,
            supported_extensions=(".pdf",),
            excluded_exact_names=(),
            excluded_prefixes=(),
            exclude_hidden=True,
            exclude_cache_directories=True,
            cache_directory_names=("__pycache__",),
            exclude_unclassified=True,
            unclassified_directory_name="unclassified",
            exclude_needs_ocr=True,
            needs_ocr_directory_name="_needs_ocr",
            normalization_version="subject_id_v1",
            symlink_policy="reject",
        ),
        subject_policy_map={"computer": "policy"},
    ).discover()
    runtime = _runtime_config(tmp_path / "runtime")
    excluded_runtime = TesseractOcrRuntimeConfig.model_validate(
        {
            **runtime.model_dump(mode="python"),
            "source_relpaths": ("computer/_needs_ocr/book.pdf",),
        }
    )
    resolved = SimpleNamespace(loader_config=SimpleNamespace(pdf_ocr=excluded_runtime))
    index_config = SimpleNamespace(chunk_policies={"policy": object()})
    monkeypatch.setattr(
        config_adapter,
        "resolve_chunk_policy",
        lambda _config, _policy_id: resolved,
    )

    with pytest.raises(config_adapter.PolicyAdapterError, match="not active"):
        config_adapter.validate_configured_ocr_source_inventory(
            index_config,
            tuple(source.source_relpath for source in catalog.source_entries()),
        )


def test_legacy_loader_fingerprint_is_byte_stable() -> None:
    loader_config = PageAwareLoaderConfig(
        schema_version="page_aware_loader_policy_v1",
        extraction_algorithm_version="page_extract_v1",
        page_assembly_algorithm_version="page_assembly_v1",
        cleaning_algorithm_version="page_clean_v2",
        cleaning_policy_id=(
            "e6f07426e2871f15d5e0337ebe270e786cab9919034a140b217458aec5a08043"
        ),
        nul_character_policy="replace_with_space_v1",
        page_separator="\n\f\n",
        normalize_newlines=True,
        strip_trailing_whitespace=True,
        strip_outer_blank_lines=True,
        header_top_lines=1,
        footer_bottom_lines=1,
        repeated_line_min_pages=2,
        repeated_line_min_ratio=0.5,
        collapse_blank_lines=True,
        paragraph_deduplication=False,
        supported_extensions=(".pdf", ".md", ".txt"),
        pdf_extraction_method="configured_pdf_text",
        text_extraction_method="configured_utf8_text",
        pdf_ocr=None,
    )
    assert make_loader_policy_fingerprint(loader_config) == (
        "f0aaa609a95de63b3c6e864fb833874258785c6d04af6721572b9dc68edc394d"
    )


def test_loader_algorithm_and_ocr_runtime_must_match(tmp_path: Path) -> None:
    runtime = _runtime_config(tmp_path)
    ocr_payload = {
        "schema_version": "page_aware_loader_policy_v1",
        "extraction_algorithm_version": "page_extract_tesseract_v2",
        "page_assembly_algorithm_version": "page_assembly_v1",
        "cleaning_algorithm_version": "page_clean_v2",
        "cleaning_policy_id": "a" * 64,
        "nul_character_policy": "replace_with_space_v1",
        "page_separator": "\n\f\n",
        "normalize_newlines": True,
        "strip_trailing_whitespace": True,
        "strip_outer_blank_lines": True,
        "header_top_lines": 1,
        "footer_bottom_lines": 1,
        "repeated_line_min_pages": 2,
        "repeated_line_min_ratio": 0.5,
        "collapse_blank_lines": True,
        "paragraph_deduplication": False,
        "supported_extensions": (".pdf", ".md", ".txt"),
        "pdf_extraction_method": "configured_pdf_text",
        "text_extraction_method": "configured_utf8_text",
        "pdf_ocr": runtime.model_dump(mode="python"),
    }
    PageAwareLoaderConfig.model_validate(ocr_payload)

    missing_runtime = dict(ocr_payload)
    missing_runtime["pdf_ocr"] = None
    with pytest.raises(ValidationError, match="requires an OCR runtime"):
        PageAwareLoaderConfig.model_validate(missing_runtime)

    unexpected_runtime = dict(ocr_payload)
    unexpected_runtime["extraction_algorithm_version"] = "page_extract_v1"
    with pytest.raises(ValidationError, match="requires pdf_ocr to be null"):
        PageAwareLoaderConfig.model_validate(unexpected_runtime)


@pytest.mark.parametrize(
    "source_relpath",
    (
        "computer//book.pdf",
        "computer/./book.pdf",
        "computer/../book.pdf",
        "computer/book\x00.pdf",
    ),
)
def test_ocr_policy_rejects_noncanonical_source_relpaths(
    tmp_path: Path,
    source_relpath: str,
) -> None:
    runtime = _runtime_config(tmp_path)
    payload = runtime.model_dump(mode="python")
    payload["schema_version"] = "tesseract_ocr_policy_v1"
    payload["source_relpaths"] = (source_relpath,)
    with pytest.raises(ValidationError, match="source_relpaths"):
        TesseractOcrPolicyConfig.model_validate(payload)
