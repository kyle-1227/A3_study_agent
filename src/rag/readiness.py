"""Read-only corpus and evaluation-data readiness audit for production RAG.

The audit never mutates source documents or indexes.  Missing source-group
metadata, unreadable documents, and insufficient human evidence are reported
as explicit production blockers instead of being replaced by inferred values.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PurePosixPath
from typing import Literal

import fitz
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.rag.subject_catalog import SourceCatalogEntry, SubjectCatalogSnapshot


DatasetKind = Literal["synthetic_smoke", "human_gold", "historical_annotated"]


class ReadinessAuditError(RuntimeError):
    """Raised when an audit input cannot be inspected safely."""


class QueryInventoryRecord(BaseModel):
    """Strict query record used only for readiness inventory counts."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    query_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    query: str = Field(min_length=1)
    dataset_kind: DatasetKind
    eligible_for_rollout: bool

    @model_validator(mode="after")
    def _validate_eligibility(self) -> QueryInventoryRecord:
        if self.dataset_kind == "synthetic_smoke" and self.eligible_for_rollout:
            raise ValueError("synthetic_smoke records cannot be rollout eligible")
        return self


class SourceGroupManifest(BaseModel):
    """Explicit mapping from source-relative path to independent source group."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["source_groups_v1"]
    source_groups: dict[str, str]

    @model_validator(mode="after")
    def _validate_groups(self) -> SourceGroupManifest:
        for relative_path, group_id in self.source_groups.items():
            _validate_relative_path(relative_path)
            if not group_id.strip():
                raise ValueError(f"source group is blank for {relative_path}")
        return self


class InspectedSource(BaseModel):
    """Per-file, content-free inspection metrics."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_relpath: str
    suffix: str
    source_group: str | None
    bytes: int = Field(ge=0)
    pages: int = Field(ge=0)
    valid_chars: int = Field(ge=0)
    empty_pages: int = Field(ge=0)
    low_text_pages: int = Field(ge=0)
    inspection_error_code: str | None


class SubjectReadiness(BaseModel):
    """Audit result and blockers for one explicitly primary subject."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    subject: str
    file_count: int = Field(ge=0)
    independent_source_count: int | None
    file_types: tuple[str, ...]
    total_bytes: int = Field(ge=0)
    valid_chars: int = Field(ge=0)
    pages: int = Field(ge=0)
    empty_pages: int = Field(ge=0)
    low_text_pages: int = Field(ge=0)
    extractable_page_ratio: float = Field(ge=0.0, le=1.0)
    synthetic_query_count: int = Field(ge=0)
    human_gold_query_count: int = Field(ge=0)
    historical_annotated_query_count: int = Field(ge=0)
    recommended_additional_source_count: int = Field(ge=0)
    recommended_additional_gold_queries: int = Field(ge=0)
    blockers: tuple[str, ...]
    production_recommendation_blocked: bool
    inspected_sources: tuple[InspectedSource, ...]


class ReadinessReport(BaseModel):
    """Top-level readiness report; contains metrics and no source body text."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["rag_readiness_v1"]
    primary_subjects: tuple[str, ...]
    global_human_or_historical_query_count: int = Field(ge=0)
    minimum_global_gold_queries: int = Field(gt=0)
    production_recommendation_blocked: bool
    global_blockers: tuple[str, ...]
    subjects: tuple[SubjectReadiness, ...]


class ReadinessAuditArtifact(BaseModel):
    """Persisted audit envelope including explicit missing-data diagnostics."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["rag_readiness_artifact_v2"]
    index_config_path: str
    benchmark_config_path: str
    gold_dataset_path: str
    gold_dataset_sha256: str
    data_root: str
    report: ReadinessReport


def _validate_relative_path(value: str) -> None:
    if not value or value != value.strip() or "\\" in value or "\x00" in value:
        raise ValueError(f"source path must be a contained POSIX path: {value}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[0].endswith(":")
    ):
        raise ValueError(f"source path must be relative and contained: {value}")


def load_source_group_manifest(path: Path) -> SourceGroupManifest:
    """Load a strict source-group JSON manifest from an explicit path."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessAuditError(
            f"unable to load source-group manifest: {type(exc).__name__}"
        ) from exc
    return SourceGroupManifest.model_validate(payload)


def load_query_inventory(path: Path) -> tuple[QueryInventoryRecord, ...]:
    """Load strict JSONL query inventory records without repairing payloads."""

    records: list[QueryInventoryRecord] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(QueryInventoryRecord.model_validate_json(line))
                except Exception as exc:
                    raise ReadinessAuditError(
                        "invalid query inventory record "
                        f"at line {line_number}: {type(exc).__name__}"
                    ) from exc
    except OSError as exc:
        raise ReadinessAuditError(
            f"unable to load query inventory: {type(exc).__name__}"
        ) from exc
    return tuple(records)


def _inspect_pdf(path: Path, *, low_text_page_chars: int) -> tuple[int, int, int, int]:
    try:
        document = fitz.open(path)
        counts = [len("".join(page.get_text("text").split())) for page in document]
    except Exception as exc:
        raise ReadinessAuditError(f"pdf_inspection_{type(exc).__name__}") from exc
    return (
        len(counts),
        sum(counts),
        sum(value == 0 for value in counts),
        sum(value < low_text_page_chars for value in counts),
    )


def _inspect_text(path: Path, *, low_text_page_chars: int) -> tuple[int, int, int, int]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReadinessAuditError(f"text_inspection_{type(exc).__name__}") from exc
    valid_chars = len("".join(text.split()))
    return (
        1,
        valid_chars,
        int(valid_chars == 0),
        int(valid_chars < low_text_page_chars),
    )


def _inspect_source(
    source: SourceCatalogEntry,
    *,
    data_root: Path,
    source_groups: dict[str, str],
    low_text_page_chars: int,
) -> InspectedSource:
    resolved_root = data_root.resolve()
    path = source.source_path
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ReadinessAuditError("source_path_outside_data_root") from exc
    if path.is_symlink():
        raise ReadinessAuditError("source_symlink_not_allowed")

    suffix = source.extension
    error_code: str | None = None
    pages = valid_chars = empty_pages = low_text_pages = 0
    try:
        if suffix == ".pdf":
            pages, valid_chars, empty_pages, low_text_pages = _inspect_pdf(
                path, low_text_page_chars=low_text_page_chars
            )
        elif suffix in {".md", ".txt"}:
            pages, valid_chars, empty_pages, low_text_pages = _inspect_text(
                path, low_text_page_chars=low_text_page_chars
            )
        else:
            raise ReadinessAuditError("unsupported_source_type")
    except ReadinessAuditError as exc:
        error_code = str(exc)

    return InspectedSource(
        source_relpath=relative,
        suffix=suffix,
        source_group=source_groups.get(relative),
        bytes=source.file_size_bytes,
        pages=pages,
        valid_chars=valid_chars,
        empty_pages=empty_pages,
        low_text_pages=low_text_pages,
        inspection_error_code=error_code,
    )


def audit_rag_readiness(
    *,
    catalog_snapshot: SubjectCatalogSnapshot,
    source_group_manifest: SourceGroupManifest,
    query_records: tuple[QueryInventoryRecord, ...],
    low_text_page_chars: int,
    minimum_independent_sources: int,
    minimum_subject_gold_queries: int,
    minimum_global_gold_queries: int,
) -> ReadinessReport:
    """Inspect the configured corpus and return explicit production blockers."""

    if low_text_page_chars <= 0:
        raise ValueError("low_text_page_chars must be positive")
    if (
        min(
            minimum_independent_sources,
            minimum_subject_gold_queries,
            minimum_global_gold_queries,
        )
        <= 0
    ):
        raise ValueError("readiness minimums must be positive")

    resolved_root = catalog_snapshot.data_root.resolve()
    if not resolved_root.is_dir():
        raise ReadinessAuditError("data_root_not_directory")

    primary_subjects = catalog_snapshot.subject_ids()
    if not primary_subjects:
        raise ReadinessAuditError("subject_catalog_has_no_active_subjects")
    subject_entries = {
        subject.subject_id: subject for subject in catalog_snapshot.subjects
    }
    unknown_query_subjects = sorted(
        {record.subject for record in query_records} - set(primary_subjects)
    )
    if unknown_query_subjects:
        raise ReadinessAuditError("query_inventory_contains_unknown_subject")
    query_counts: Counter[tuple[str, DatasetKind]] = Counter(
        (record.subject, record.dataset_kind) for record in query_records
    )
    subjects: list[SubjectReadiness] = []

    for subject in primary_subjects:
        sources = subject_entries[subject].sources
        inspected = tuple(
            _inspect_source(
                source,
                data_root=resolved_root,
                source_groups=source_group_manifest.source_groups,
                low_text_page_chars=low_text_page_chars,
            )
            for source in sources
        )
        groups = {
            item.source_group for item in inspected if item.source_group is not None
        }
        source_group_complete = all(item.source_group is not None for item in inspected)
        independent_count = len(groups) if source_group_complete else None
        human_count = query_counts[(subject, "human_gold")]
        historical_count = query_counts[(subject, "historical_annotated")]
        eligible_gold_count = human_count + historical_count

        blockers: list[str] = []
        if not inspected:
            blockers.append("no_supported_sources")
        if any(item.inspection_error_code is not None for item in inspected):
            blockers.append("source_inspection_failed")
        if not source_group_complete:
            blockers.append("source_group_manifest_incomplete")
        if (
            independent_count is not None
            and independent_count < minimum_independent_sources
        ):
            blockers.append("insufficient_independent_sources")
        if eligible_gold_count < minimum_subject_gold_queries:
            blockers.append("insufficient_human_or_historical_gold_queries")

        pages = sum(item.pages for item in inspected)
        low_text_pages = sum(item.low_text_pages for item in inspected)
        subjects.append(
            SubjectReadiness(
                subject=subject,
                file_count=len(inspected),
                independent_source_count=independent_count,
                file_types=tuple(sorted({item.suffix for item in inspected})),
                total_bytes=sum(item.bytes for item in inspected),
                valid_chars=sum(item.valid_chars for item in inspected),
                pages=pages,
                empty_pages=sum(item.empty_pages for item in inspected),
                low_text_pages=low_text_pages,
                extractable_page_ratio=(pages - low_text_pages) / pages
                if pages
                else 0.0,
                synthetic_query_count=query_counts[(subject, "synthetic_smoke")],
                human_gold_query_count=human_count,
                historical_annotated_query_count=historical_count,
                recommended_additional_source_count=max(
                    0,
                    minimum_independent_sources - (independent_count or 0),
                ),
                recommended_additional_gold_queries=max(
                    0, minimum_subject_gold_queries - eligible_gold_count
                ),
                blockers=tuple(blockers),
                production_recommendation_blocked=bool(blockers),
                inspected_sources=inspected,
            )
        )

    global_gold = sum(
        1
        for record in query_records
        if record.dataset_kind in {"human_gold", "historical_annotated"}
        and record.subject in primary_subjects
        and record.eligible_for_rollout
    )
    global_blockers: list[str] = []
    if global_gold < minimum_global_gold_queries:
        global_blockers.append("insufficient_global_human_or_historical_gold_queries")
    if any(item.production_recommendation_blocked for item in subjects):
        global_blockers.append("one_or_more_primary_subjects_blocked")

    return ReadinessReport(
        schema_version="rag_readiness_v1",
        primary_subjects=primary_subjects,
        global_human_or_historical_query_count=global_gold,
        minimum_global_gold_queries=minimum_global_gold_queries,
        production_recommendation_blocked=bool(global_blockers),
        global_blockers=tuple(global_blockers),
        subjects=tuple(subjects),
    )
