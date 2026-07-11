"""Strict, local-only authoring support for policy-independent RAG gold data.

The persisted :class:`GoldDataset` remains the only benchmark truth.  A draft
format exists solely so an empty, reviewable annotation file can be created
without fabricating a query or evidence span.  Validation always reloads the
configured page-aware source and proves every submitted coordinate before it
can be sealed as ``gold_dataset_v1``.
"""

from __future__ import annotations

import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config.rag_index_config import RagIndexConfig
from src.rag.chunking.structure_detector import detect_document_sections
from src.rag.parent_child._storage_io import model_json_bytes
from src.rag.parent_child.project_paths import (
    ProjectPathError,
    atomic_write_project_bytes,
    resolve_project_path as _resolve_project_path,
)
from src.rag.parent_child.config_adapter import resolve_subject_chunk_policy
from src.rag.parent_child.evaluation import GoldDataset, GoldEvidenceSpan, GoldQuery
from src.rag.parent_child.loader import load_cleaned_source, page_range_for_span
from src.rag.parent_child.models import CleanedSourceDocument, SourceEntry
from src.rag.readiness import QueryInventoryRecord, SourceGroupManifest
from src.rag.subject_catalog import (
    SourceCatalogEntry,
    SubjectCatalog,
    SubjectCatalogSnapshot,
)


class GoldDatasetAuthoringError(RuntimeError):
    """A local GoldDataset operation could not prove its strict contract."""


class GoldDatasetPathError(GoldDatasetAuthoringError, ProjectPathError):
    """A CLI path is outside the project root or crosses a reparse point."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty and already stripped")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must not contain path separators or NUL")
    return value


class GoldDatasetDraft(_StrictFrozenModel):
    """Editable local precursor which deliberately permits an empty query list."""

    schema_version: Literal["gold_dataset_draft_v1"]
    dataset_id: str
    queries: tuple[GoldQuery, ...]

    @field_validator("dataset_id")
    @classmethod
    def _validate_dataset_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="dataset_id")

    @model_validator(mode="after")
    def _validate_unique_query_ids(self) -> GoldDatasetDraft:
        query_ids = tuple(query.query_id for query in self.queries)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("queries must not contain duplicate query_id values")
        return self


class GoldInspectionPage(_StrictFrozenModel):
    """One local annotation page in cleaned-document offset coordinates."""

    page_number: int = Field(ge=1)
    start_char: int = Field(ge=0)
    content_end_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    cleaned_text: str

    @model_validator(mode="after")
    def _validate_offsets(self) -> GoldInspectionPage:
        if not self.start_char <= self.content_end_char <= self.end_char:
            raise ValueError("page offsets must be ordered")
        if self.content_end_char - self.start_char != len(self.cleaned_text):
            raise ValueError("cleaned_text length must match the page content span")
        return self


class GoldInspectionSection(_StrictFrozenModel):
    """One deterministic structure-detector section for local span review."""

    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    section_path: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_offsets(self) -> GoldInspectionSection:
        if self.end_char <= self.start_char:
            raise ValueError("section span must be non-empty")
        return self


class GoldSourceInspection(_StrictFrozenModel):
    """Content-bearing local annotation aid; never a safe operational report."""

    schema_version: Literal["gold_source_inspection_v1"]
    source_relpath: str
    doc_id: str
    subject: str
    pagination_kind: Literal["physical", "logical"]
    pages: tuple[GoldInspectionPage, ...] = Field(min_length=1)
    sections: tuple[GoldInspectionSection, ...] = Field(min_length=1)
    cleaned_content: str = Field(min_length=1)


def resolve_project_path(
    *,
    project_root: Path,
    value: Path,
    must_exist: bool,
) -> Path:
    """Resolve one project-contained path while rejecting symlink/junction hops."""

    if not isinstance(project_root, Path) or not isinstance(value, Path):
        raise TypeError("project_root and value must be pathlib.Path instances")
    if any(part == ".." for part in value.parts):
        raise GoldDatasetPathError("configured paths must not contain parent traversal")
    try:
        return _resolve_project_path(project_root, value, must_exist=must_exist)
    except ProjectPathError as exc:
        raise GoldDatasetPathError(str(exc)) from exc


def project_relative_path(*, project_root: Path, path: Path) -> str:
    """Return a contained path label after strict project-boundary validation."""

    root = resolve_project_path(
        project_root=project_root,
        value=project_root,
        must_exist=True,
    )
    resolved = resolve_project_path(
        project_root=root,
        value=path,
        must_exist=False,
    )
    return resolved.relative_to(root).as_posix()


def _load_json_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldDatasetAuthoringError(
            f"unable to read GoldDataset JSON: {type(exc).__name__}"
        ) from exc


def load_gold_dataset_draft_or_final(path: Path) -> GoldDatasetDraft | GoldDataset:
    """Load only the two explicit local authoring schemas without repair."""

    document = _load_json_text(path)
    try:
        payload = json.loads(document)
    except json.JSONDecodeError as exc:
        raise GoldDatasetAuthoringError(
            f"unable to load GoldDataset JSON: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise GoldDatasetAuthoringError("GoldDataset JSON root must be an object")
    schema_version = payload.get("schema_version")
    try:
        if schema_version == "gold_dataset_draft_v1":
            return GoldDatasetDraft.model_validate_json(document)
        if schema_version == "gold_dataset_v1":
            return GoldDataset.model_validate_json(document)
    except Exception as exc:
        raise GoldDatasetAuthoringError(
            f"GoldDataset contract validation failed: {type(exc).__name__}"
        ) from exc
    raise GoldDatasetAuthoringError("unsupported GoldDataset schema_version")


def load_gold_dataset(path: Path) -> GoldDataset:
    """Load a final, non-draft GoldDataset without accepting alternate schemas."""

    try:
        return GoldDataset.model_validate_json(_load_json_text(path))
    except Exception as exc:
        raise GoldDatasetAuthoringError(
            f"final GoldDataset contract validation failed: {type(exc).__name__}"
        ) from exc


def _doc_type_for_source(source: SourceCatalogEntry) -> str:
    doc_types = {
        ".pdf": "pdf",
        ".md": "markdown",
        ".txt": "text",
    }
    try:
        return doc_types[source.extension]
    except KeyError as exc:
        raise GoldDatasetAuthoringError(
            "configured source extension has no page-aware GoldDataset loader"
        ) from exc


def _catalog_snapshot(index_config: RagIndexConfig) -> SubjectCatalogSnapshot:
    try:
        return SubjectCatalog(
            config=index_config.catalog,
            subject_policy_map=index_config.subject_policy_map,
        ).discover()
    except Exception as exc:
        raise GoldDatasetAuthoringError(
            f"subject catalog discovery failed: {type(exc).__name__}"
        ) from exc


def _catalog_sources(
    snapshot: SubjectCatalogSnapshot,
) -> dict[str, SourceCatalogEntry]:
    sources: dict[str, SourceCatalogEntry] = {}
    for source in snapshot.source_entries():
        if source.source_relpath in sources:
            raise GoldDatasetAuthoringError("catalog returned duplicate source_relpath")
        sources[source.source_relpath] = source
    return sources


def _load_source_document(
    *,
    index_config: RagIndexConfig,
    snapshot: SubjectCatalogSnapshot,
    source: SourceCatalogEntry,
) -> CleanedSourceDocument:
    resolved_policy = resolve_subject_chunk_policy(index_config, source.subject_id)
    try:
        return load_cleaned_source(
            SourceEntry(
                schema_version="source_entry_v1",
                source_path=source.source_path,
                data_root=snapshot.data_root,
                subject=source.subject_id,
                doc_type=_doc_type_for_source(source),
            ),
            resolved_policy.loader_config,
        )
    except Exception as exc:
        raise GoldDatasetAuthoringError(
            f"page-aware source loading failed: {type(exc).__name__}"
        ) from exc


def _section_for_span(
    source: CleanedSourceDocument,
    *,
    start_char: int,
    end_char: int,
) -> tuple[str, ...]:
    matching = tuple(
        section
        for section in detect_document_sections(source.content)
        if section.start_char <= start_char and end_char <= section.end_char
    )
    if len(matching) != 1:
        raise GoldDatasetAuthoringError(
            "gold evidence must be contained in exactly one detected section"
        )
    return matching[0].section_path


def validate_gold_dataset(
    *,
    dataset: GoldDataset,
    index_config: RagIndexConfig,
    source_groups: SourceGroupManifest,
) -> GoldDataset:
    """Prove all submitted Gold evidence against the candidate loader output."""

    snapshot = _catalog_snapshot(index_config)
    sources = _catalog_sources(snapshot)
    documents: dict[str, CleanedSourceDocument] = {}
    known_subjects = set(snapshot.subject_ids())
    for query in dataset.queries:
        if query.subject not in known_subjects:
            raise GoldDatasetAuthoringError("Gold query subject is absent from catalog")
        for span in query.gold_spans:
            source = sources.get(span.source_relpath)
            if source is None:
                raise GoldDatasetAuthoringError(
                    "Gold source_relpath is absent from catalog"
                )
            if source.subject_id != query.subject:
                raise GoldDatasetAuthoringError(
                    "Gold query subject must match the source catalog subject"
                )
            expected_group = source_groups.source_groups.get(span.source_relpath)
            if expected_group is None:
                raise GoldDatasetAuthoringError(
                    "Gold evidence source_relpath has no explicit source group"
                )
            if span.source_group_id != expected_group:
                raise GoldDatasetAuthoringError(
                    "Gold evidence source_group_id must match source_groups manifest"
                )
            document = documents.get(span.source_relpath)
            if document is None:
                document = _load_source_document(
                    index_config=index_config,
                    snapshot=snapshot,
                    source=source,
                )
                documents[span.source_relpath] = document
            _validate_gold_span(span=span, source=document)
    return dataset


def _validate_gold_span(
    *,
    span: GoldEvidenceSpan,
    source: CleanedSourceDocument,
) -> None:
    if span.doc_id != source.doc_id:
        raise GoldDatasetAuthoringError("Gold doc_id does not match cleaned source")
    if span.pagination_kind != source.pagination_kind:
        raise GoldDatasetAuthoringError(
            "Gold pagination_kind does not match cleaned source"
        )
    if span.end_char > len(source.content):
        raise GoldDatasetAuthoringError("Gold span exceeds cleaned source length")
    if not source.content[span.start_char : span.end_char].strip():
        raise GoldDatasetAuthoringError("Gold span selects only cleaned whitespace")
    expected_pages = page_range_for_span(
        source,
        start_char=span.start_char,
        end_char=span.end_char,
    )
    if (span.page_start, span.page_end) != expected_pages:
        raise GoldDatasetAuthoringError(
            "Gold page range does not match cleaned-character span"
        )
    if span.section_path is None:
        raise GoldDatasetAuthoringError("Gold section_path must be explicit")
    expected_section_path = _section_for_span(
        source,
        start_char=span.start_char,
        end_char=span.end_char,
    )
    if span.section_path != expected_section_path:
        raise GoldDatasetAuthoringError(
            "Gold section_path does not match the detected source section"
        )


def inspect_gold_source(
    *,
    index_config: RagIndexConfig,
    source_relpath: str,
) -> GoldSourceInspection:
    """Return exact page/section coordinates and cleaned text for local review."""

    try:
        parsed = PurePosixPath(source_relpath)
    except TypeError as exc:
        raise GoldDatasetAuthoringError(
            "source_relpath must be a POSIX string"
        ) from exc
    if (
        not source_relpath
        or "\\" in source_relpath
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise GoldDatasetAuthoringError("source_relpath must be a contained POSIX path")
    snapshot = _catalog_snapshot(index_config)
    source = _catalog_sources(snapshot).get(parsed.as_posix())
    if source is None:
        raise GoldDatasetAuthoringError("source_relpath is absent from catalog")
    document = _load_source_document(
        index_config=index_config,
        snapshot=snapshot,
        source=source,
    )
    pages = tuple(
        GoldInspectionPage(
            page_number=page.page_number,
            start_char=span.start_char,
            content_end_char=span.content_end_char,
            end_char=span.end_char,
            cleaned_text=page.cleaned_text,
        )
        for page, span in zip(document.source_pages, document.page_spans, strict=True)
    )
    sections = tuple(
        GoldInspectionSection(
            start_char=section.start_char,
            end_char=section.end_char,
            section_path=section.section_path,
        )
        for section in detect_document_sections(document.content)
    )
    return GoldSourceInspection(
        schema_version="gold_source_inspection_v1",
        source_relpath=document.source_relpath,
        doc_id=document.doc_id,
        subject=document.subject,
        pagination_kind=document.pagination_kind,
        pages=pages,
        sections=sections,
        cleaned_content=document.content,
    )


def gold_dataset_to_query_inventory(
    dataset: GoldDataset,
) -> tuple[QueryInventoryRecord, ...]:
    """Project the single source-of-truth dataset into readiness-only records."""

    return tuple(
        QueryInventoryRecord(
            query_id=query.query_id,
            subject=query.subject,
            query=query.query,
            dataset_kind=query.dataset_kind,
            eligible_for_rollout=query.eligible_for_rollout,
        )
        for query in sorted(dataset.queries, key=lambda item: item.query_id)
    )


def write_gold_dataset_jsonl_exports(
    *,
    project_root: Path,
    dataset: GoldDataset,
    human_output: Path,
    historical_output: Path,
    synthetic_output: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    """Write deterministic readiness inventories derived only from GoldDataset."""

    root = resolve_project_path(
        project_root=project_root,
        value=project_root,
        must_exist=True,
    )
    output_specs = (
        ("human_gold", human_output),
        ("historical_annotated", historical_output),
        ("synthetic_smoke", synthetic_output),
    )
    seen_outputs: set[Path] = set()
    outputs: list[Path] = []
    records = gold_dataset_to_query_inventory(dataset)
    for dataset_kind, requested_path in output_specs:
        output = resolve_project_path(
            project_root=root,
            value=requested_path,
            must_exist=False,
        )
        if output in seen_outputs:
            raise GoldDatasetPathError("JSONL export paths must be distinct")
        seen_outputs.add(output)
        payload = b"".join(
            model_json_bytes(record) + b"\n"
            for record in records
            if record.dataset_kind == dataset_kind
        )
        written = atomic_write_project_bytes(
            root,
            output,
            payload,
            overwrite=overwrite,
        )
        outputs.append(written)
    return tuple(outputs)  # type: ignore[return-value]


def write_gold_model(
    *,
    project_root: Path,
    output_path: Path,
    model: GoldDatasetDraft | GoldDataset | GoldSourceInspection,
    overwrite: bool,
) -> Path:
    """Persist one validated local artifact under the strict project boundary."""

    root = resolve_project_path(
        project_root=project_root,
        value=project_root,
        must_exist=True,
    )
    output = resolve_project_path(
        project_root=root,
        value=output_path,
        must_exist=False,
    )
    return atomic_write_project_bytes(
        root,
        output,
        model_json_bytes(model),
        overwrite=overwrite,
    )


__all__ = [
    "GoldDatasetAuthoringError",
    "GoldDatasetDraft",
    "GoldDatasetPathError",
    "GoldInspectionPage",
    "GoldInspectionSection",
    "GoldSourceInspection",
    "gold_dataset_to_query_inventory",
    "inspect_gold_source",
    "load_gold_dataset",
    "load_gold_dataset_draft_or_final",
    "project_relative_path",
    "resolve_project_path",
    "validate_gold_dataset",
    "write_gold_dataset_jsonl_exports",
    "write_gold_model",
]
