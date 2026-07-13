from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import fitz
import pytest
from pydantic import ValidationError

from src.rag.parent_child._storage_io import (
    ArtifactPathError,
    atomic_write_bytes,
    model_json_bytes,
    resolve_under_root,
    sha256_bytes,
    sha256_path,
)
from src.rag.parent_child.bm25_artifact import (
    Bm25CorpusRow,
    compute_tokenizer_fingerprint,
    read_subject_bm25_artifact,
    write_subject_bm25_artifact,
)
from src.rag.parent_child.exceptions import AtomicSpanTooLargeError
from src.rag.parent_child.generation import GenerationWorkspace
from src.rag.parent_child.ids import make_loader_policy_fingerprint
from src.rag.parent_child.loader import load_cleaned_source
from src.rag.parent_child.manifests import (
    ArtifactDescriptor,
    Bm25ManifestIdentity,
    EmbeddingManifestIdentity,
    GenerationCounts,
    GenerationIntegrityCounts,
    GenerationManifest,
    build_completeness_report,
)
from src.rag.parent_child.models import (
    PageAwareLoaderConfig,
    ParentChildPolicy,
    SourceEntry,
)
from src.rag.parent_child.parent_store import (
    MissingParentError,
    ParentStore,
    create_parent_store,
)
from src.rag.parent_child.registry import (
    GenerationCleanupError,
    GenerationRegistry,
    create_generation_registry,
)
from src.rag.parent_child.splitter import (
    build_parent_child_bundle,
    detect_protected_atomic_spans,
)


def _loader_config() -> PageAwareLoaderConfig:
    return PageAwareLoaderConfig(
        schema_version="page_aware_loader_policy_v1",
        extraction_algorithm_version="page_extract_v1",
        page_assembly_algorithm_version="page_assembly_v1",
        cleaning_algorithm_version="page_clean_v1",
        cleaning_policy_id="page_clean_v1",
        page_separator="\n\f\n",
        normalize_newlines=True,
        strip_trailing_whitespace=True,
        strip_outer_blank_lines=True,
        header_top_lines=3,
        footer_bottom_lines=3,
        repeated_line_min_pages=3,
        repeated_line_min_ratio=0.6,
        collapse_blank_lines=True,
        paragraph_deduplication=False,
        supported_extensions=(".pdf", ".md", ".txt"),
        pdf_extraction_method="pymupdf_text_v1",
        text_extraction_method="utf8_text_v1",
    )


def test_storage_path_resolution_rejects_symlink_escape(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    outside = tmp_path / "outside"
    storage_root.mkdir()
    outside.mkdir()
    link = storage_root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {type(exc).__name__}")

    with pytest.raises(ArtifactPathError, match="symlink"):
        resolve_under_root(storage_root, "linked/artifact.json", must_exist=False)


def test_storage_path_resolution_rejects_detected_symlink_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    real_is_symlink = Path.is_symlink

    def detected_symlink(path: Path) -> bool:
        return path.name == "linked" or real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", detected_symlink)
    with pytest.raises(ArtifactPathError, match="symlink"):
        resolve_under_root(storage_root, "linked/artifact.json", must_exist=False)


def _policy(loader: PageAwareLoaderConfig, **overrides: object) -> ParentChildPolicy:
    values: dict[str, object] = {
        "schema_version": "parent_child_policy_v1",
        "canonicalization_version": "canonical_json_v1",
        "id_algorithm_version": "parent_child_id_v1",
        "metadata_contract_version": "parent_child_metadata_v1",
        "policy_id": "a" * 64,
        "structure_detector_version": "structure_detector_v1",
        "structure_pattern_set_version": "patterns_v1",
        "structure_merge_version": "short_unit_merge_v1",
        "short_unit_chars": 30,
        "parent_split_algorithm": "span_recursive_v1",
        "child_split_algorithm": "span_recursive_v1",
        "loader_policy_id": make_loader_policy_fingerprint(loader),
        "cleaning_policy_id": loader.cleaning_policy_id,
        "parent_size": 90,
        "parent_overlap": 15,
        "parent_hard_max": 180,
        "child_size": 45,
        "child_overlap": 8,
        "child_hard_max": 100,
        "parent_separators": ("\n\n", "\n", "。", " "),
        "child_separators": ("\n\n", "\n", "。", " "),
        "major_section_max_level": 2,
        "atomic_fenced_code_blocks": True,
        "atomic_markdown_tables": True,
        "atomic_list_blocks": True,
        "atomic_display_math": True,
    }
    values.update(overrides)
    return ParentChildPolicy.model_validate(values)


def _text_bundle(tmp_path: Path, *, generation_id: str = "gen-a"):
    data_root = tmp_path / "data"
    source_path = data_root / "math" / "notes.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "# Limits\n\n" + "Limit definition and examples. " * 8 + "\n\n"
        "# Derivatives\n\n" + "Derivative rules and examples. " * 8,
        encoding="utf-8",
    )
    loader = _loader_config()
    source = load_cleaned_source(
        SourceEntry(
            schema_version="source_entry_v1",
            source_path=source_path,
            data_root=data_root,
            subject="math",
            doc_type="notes",
        ),
        loader,
    )
    return build_parent_child_bundle(source, _policy(loader), generation_id)


def test_page_aware_pdf_preserves_empty_page_ordinal(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    source_path = data_root / "math" / "pages.pdf"
    source_path.parent.mkdir(parents=True)
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "First physical page")
    document.new_page()
    third = document.new_page()
    third.insert_text((72, 72), "Third physical page")
    document.save(source_path)
    document.close()

    source = load_cleaned_source(
        SourceEntry(
            schema_version="source_entry_v1",
            source_path=source_path,
            data_root=data_root,
            subject="math",
            doc_type="notes",
        ),
        _loader_config(),
    )

    assert source.pagination_kind == "physical"
    assert tuple(page.page_number for page in source.source_pages) == (1, 2, 3)
    assert source.source_pages[1].cleaned_text == ""
    assert tuple(span.page_number for span in source.page_spans) == (1, 2, 3)


def test_page_aware_cleaning_removes_only_repeated_page_edge_noise(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    source_path = data_root / "math" / "headers.pdf"
    source_path.parent.mkdir(parents=True)
    document = fitz.open()
    for number in range(1, 4):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"Course handout\nUnique body {number}\nShared body phrase\nPage footer",
        )
    document.save(source_path)
    document.close()

    config = _loader_config().model_copy(
        update={"header_top_lines": 1, "footer_bottom_lines": 1}
    )
    source = load_cleaned_source(
        SourceEntry(
            schema_version="source_entry_v1",
            source_path=source_path,
            data_root=data_root,
            subject="math",
            doc_type="notes",
        ),
        config,
    )

    assert "Course handout" not in source.content
    assert "Page footer" not in source.content
    assert source.content.count("Shared body phrase") == 3
    assert tuple(page.page_number for page in source.source_pages) == (1, 2, 3)


def test_parent_child_bundle_is_exact_stable_and_generation_independent(
    tmp_path: Path,
) -> None:
    first = _text_bundle(tmp_path, generation_id="gen-a")
    second = build_parent_child_bundle(first.source, first.policy, "gen-b")

    assert tuple(parent.parent_id for parent in first.parents) == tuple(
        parent.parent_id for parent in second.parents
    )
    assert tuple(child.metadata.child_id for child in first.children) == tuple(
        child.metadata.child_id for child in second.children
    )
    for parent in first.parents:
        assert (
            parent.content == first.source.content[parent.start_char : parent.end_char]
        )
    parent_by_id = {parent.parent_id: parent for parent in first.parents}
    for child in first.children:
        parent = parent_by_id[child.metadata.parent_id]
        assert (
            child.content
            == parent.content[
                child.metadata.child_start_in_parent : child.metadata.child_end_in_parent
            ]
        )
        metadata = child.metadata.to_chroma_metadata()
        assert all(
            isinstance(value, (str, int, float, bool)) for value in metadata.values()
        )
        assert "content" not in metadata


def test_schema_drift_and_oversized_atomic_block_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        PageAwareLoaderConfig.model_validate(
            {**_loader_config().model_dump(), "unexpected": True}
        )

    data_root = tmp_path / "atomic-data"
    source_path = data_root / "python" / "code.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("```python\n" + ("x = 1\n" * 30) + "```", encoding="utf-8")
    loader = _loader_config()
    source = load_cleaned_source(
        SourceEntry(
            schema_version="source_entry_v1",
            source_path=source_path,
            data_root=data_root,
            subject="python",
            doc_type="notes",
        ),
        loader,
    )
    policy = _policy(
        loader,
        parent_size=40,
        parent_overlap=5,
        parent_hard_max=80,
        child_size=30,
        child_overlap=5,
        child_hard_max=60,
    )
    with pytest.raises(AtomicSpanTooLargeError):
        build_parent_child_bundle(source, policy, "gen-atomic")


def test_unclosed_fence_does_not_protect_the_rest_of_a_document(tmp_path: Path) -> None:
    data_root = tmp_path / "unclosed-fence-data"
    source_path = data_root / "python" / "notes.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "```not-a-real-code-fence\n"
        + ("Ordinary extracted prose must remain splittable.\n" * 40),
        encoding="utf-8",
    )
    loader = _loader_config()
    source = load_cleaned_source(
        SourceEntry(
            schema_version="source_entry_v1",
            source_path=source_path,
            data_root=data_root,
            subject="python",
            doc_type="notes",
        ),
        loader,
    )
    policy = _policy(
        loader,
        parent_size=90,
        parent_overlap=15,
        parent_hard_max=180,
        child_size=45,
        child_overlap=8,
        child_hard_max=100,
    )

    protected = detect_protected_atomic_spans(source, policy)
    bundle = build_parent_child_bundle(source, policy, "gen-unclosed-fence")

    assert protected == ()
    assert bundle.parents
    assert bundle.children


def test_parent_store_and_bm25_artifact_round_trip(tmp_path: Path) -> None:
    bundle = _text_bundle(tmp_path)
    generation_root = tmp_path / "generation"
    create_parent_store(
        generation_root,
        "parents.sqlite",
        bundle.parents,
        store_schema_version="parent_store_v1",
        expected_generation_id=bundle.generation_id,
        busy_timeout_seconds=1.0,
    )
    with ParentStore.open_readonly(
        generation_root,
        "parents.sqlite",
        expected_schema_version="parent_store_v1",
        expected_generation_id=bundle.generation_id,
        busy_timeout_seconds=1.0,
    ) as store:
        store.verify_integrity()
        hydrated = store.get_many([bundle.parents[0].parent_id])
        assert hydrated == (bundle.parents[0],)
        with pytest.raises(MissingParentError):
            store.get_many(["parent_" + "f" * 40])

    rows = tuple(
        Bm25CorpusRow(
            schema_version="bm25_row_v1",
            generation_id=bundle.generation_id,
            subject="math",
            child_id=child.metadata.child_id,
            tokens=tuple(child.content.split()),
        )
        for child in bundle.children
    )
    dictionary_sha256 = "b" * 64
    manifest = write_subject_bm25_artifact(
        generation_root,
        "bm25/math.jsonl",
        "bm25/math.manifest.json",
        rows,
        manifest_schema_version="bm25_manifest_v1",
        expected_generation_id=bundle.generation_id,
        expected_subject="math",
        tokenizer_name="whitespace-test",
        tokenizer_version="v1",
        dictionary_sha256=dictionary_sha256,
    )
    expected_fingerprint = compute_tokenizer_fingerprint(
        tokenizer_name="whitespace-test",
        tokenizer_version="v1",
        dictionary_sha256=dictionary_sha256,
    )
    loaded_manifest, loaded_rows = read_subject_bm25_artifact(
        generation_root,
        "bm25/math.manifest.json",
        expected_manifest_schema_version="bm25_manifest_v1",
        expected_generation_id=bundle.generation_id,
        expected_subject="math",
        expected_tokenizer_fingerprint=expected_fingerprint,
    )
    assert loaded_manifest == manifest
    assert loaded_rows == tuple(sorted(rows, key=lambda item: item.child_id))


def _artifact_descriptor(root: Path, artifact_type: str, relative_path: str):
    path = root / relative_path
    size = (
        path.stat().st_size
        if path.is_file()
        else sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    )
    return ArtifactDescriptor(
        artifact_type=artifact_type,
        relative_path=relative_path,
        sha256=sha256_path(path),
        schema_version="artifact_v1",
        size_bytes=size,
    )


def _seal_empty_generation(index_root: Path, generation_id: str):
    workspace = GenerationWorkspace.create(
        index_root,
        generation_id,
        marker_schema_version="owner_v1",
    )
    artifact_rows = (
        ("chroma_children", "chroma_children/data.bin"),
        ("parent_store", "parents.sqlite"),
        ("bm25_corpus", "bm25/math.jsonl"),
        ("bm25_manifest", "bm25/math.manifest.json"),
        ("policy_manifest", "policy_manifest.json"),
        ("subject_manifest", "subject_manifest.json"),
        ("build_report", "build_report.json"),
    )
    for _, relative_path in artifact_rows:
        atomic_write_bytes(
            workspace.staging_path,
            relative_path,
            f"{generation_id}:{relative_path}".encode(),
            overwrite=False,
        )
    descriptors = tuple(
        _artifact_descriptor(workspace.staging_path, artifact_type, relative_path)
        for artifact_type, relative_path in artifact_rows
    )
    integrity = GenerationIntegrityCounts(
        duplicate_parent_count=0,
        duplicate_child_count=0,
        orphan_child_count=0,
        unreferenced_parent_count=0,
        generation_mismatch_count=0,
        policy_mismatch_count=0,
        subject_mismatch_count=0,
        bm25_mismatch_count=0,
        chroma_mismatch_count=0,
    )
    report = build_completeness_report(
        (),
        (),
        {},
        (),
        {},
        report_schema_version="validation_v1",
        expected_generation_id=generation_id,
        source_count=0,
    )
    counts = GenerationCounts(
        source_count=0,
        subject_count=0,
        parent_count=0,
        child_count=0,
        bm25_child_count=0,
    )
    by_type = {descriptor.artifact_type: descriptor for descriptor in descriptors}
    manifest = GenerationManifest(
        schema_version="generation_manifest_v1",
        generation_id=generation_id,
        build_state="ready",
        code_revision="test-revision",
        build_time_utc=datetime.now(UTC),
        collection_name="children",
        artifacts=descriptors,
        embedding=EmbeddingManifestIdentity(
            provider="test-provider",
            model="test-model",
            base_url_identity="https://invalid.test",
            input_types=("document", "query"),
            fingerprint="a" * 64,
            dimension=3,
            distance_metric="cosine",
        ),
        bm25=Bm25ManifestIdentity(
            tokenizer_name="test-tokenizer",
            tokenizer_version="v1",
            dictionary_sha256="b" * 64,
            tokenizer_fingerprint="c" * 64,
            artifact_format="jsonl",
        ),
        subject_manifest_sha256=by_type["subject_manifest"].sha256,
        policy_manifest_sha256=by_type["policy_manifest"].sha256,
        subject_fingerprint="d" * 64,
        policy_fingerprint="e" * 64,
        source_fingerprint="f" * 64,
        parent_id_set_sha256=report.parent_id_set_sha256,
        child_id_set_sha256=report.child_id_set_sha256,
        counts=counts,
        integrity=integrity,
        validation_report_sha256=sha256_bytes(model_json_bytes(report)),
        validation_passed=True,
    )
    return workspace.seal(manifest, report)


def test_generation_registry_activation_rollback_and_cleanup(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    index_root.mkdir()
    registry_path = create_generation_registry(
        index_root / "generation_registry.sqlite",
        schema_version="registry_v1",
        busy_timeout_seconds=1.0,
    )
    with GenerationRegistry.open(
        registry_path,
        index_root=index_root,
        expected_schema_version="registry_v1",
        marker_schema_version="owner_v1",
        busy_timeout_seconds=1.0,
    ) as registry:
        sealed = []
        for generation_id in ("gen-a", "gen-b", "gen-c"):
            registry.register_building(generation_id)
            registry.transition(generation_id, "VALIDATING")
            result = _seal_empty_generation(index_root, generation_id)
            registry.mark_ready(
                generation_id,
                manifest_sha256=result.manifest_sha256,
            )
            sealed.append(result)

        first = registry.activate("gen-a")
        second = registry.activate("gen-b")
        rolled_back = registry.rollback()
        assert first.revision == 1
        assert second.replaced_generation_id == "gen-a"
        assert rolled_back.generation_id == "gen-a"
        assert rolled_back.action == "rollback"
        with pytest.raises(GenerationCleanupError):
            registry.cleanup_generation("gen-b")

        registry.cleanup_generation("gen-c")
        assert registry.get_generation("gen-c").state == "DELETED"
        assert not (index_root / "gen-c").exists()


def test_failed_after_seal_generation_can_be_owned_cleanup(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    index_root.mkdir()
    registry_path = create_generation_registry(
        index_root / "generation_registry.sqlite",
        schema_version="registry_v1",
        busy_timeout_seconds=1.0,
    )
    with GenerationRegistry.open(
        registry_path,
        index_root=index_root,
        expected_schema_version="registry_v1",
        marker_schema_version="owner_v1",
        busy_timeout_seconds=1.0,
    ) as registry:
        registry.register_building("gen-after-seal")
        registry.transition("gen-after-seal", "VALIDATING")
        _seal_empty_generation(index_root, "gen-after-seal")
        registry.mark_failed(
            "gen-after-seal",
            failure_code="registry_ready_failed",
            failure_type="InjectedFailure",
        )

        registry.cleanup_generation("gen-after-seal")

        assert registry.get_generation("gen-after-seal").state == "DELETED"
        assert not (index_root / "gen-after-seal").exists()
