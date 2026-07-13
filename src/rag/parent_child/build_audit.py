"""Read-only, provider-free audits for a real Parent-Child chunk dry run.

The functions below invoke the production SubjectCatalog, page-aware loader, and
parent-child splitter. They never create Chroma, BM25, Parent Store, or registry
artifacts; the calling build orchestrator owns report persistence.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable
import math
from pathlib import Path
from statistics import mean, median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.rag_index_config import (
    RagIndexConfig,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.ids import make_doc_id
from src.rag.parent_child.config_adapter import resolve_subject_chunk_policy
from src.rag.parent_child.loader import load_cleaned_source
from src.rag.parent_child.models import ChildDocument, ParentRecord, SourceEntry
from src.rag.parent_child.project_paths import (
    require_project_file,
    resolve_project_root,
)
from src.rag.parent_child.splitter import (
    ProtectedAtomicSpan,
    build_parent_child_bundle,
    detect_protected_atomic_spans,
)
from src.rag.readiness import load_source_group_manifest
from src.rag.subject_catalog import SubjectCatalog, SubjectCatalogSnapshot


_SOURCE_DOC_TYPES = {".md": "markdown", ".pdf": "pdf", ".txt": "text"}


class BuildAuditError(RuntimeError):
    """The local build preflight or chunk dry run has a typed failure."""


class NumericDistribution(BaseModel):
    """A content-free numeric distribution with an explicit empty state."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["rag_numeric_distribution_v1"]
    count: int = Field(ge=0)
    minimum: float | None
    mean: float | None
    median: float | None
    p95: float | None
    maximum: float | None

    @model_validator(mode="after")
    def _validate_distribution(self) -> NumericDistribution:
        values = (self.minimum, self.mean, self.median, self.p95, self.maximum)
        if self.count == 0:
            if any(value is not None for value in values):
                raise ValueError("empty distributions must use null summary values")
            return self
        if any(value is None for value in values):
            raise ValueError("non-empty distributions require every summary value")
        finite_values = tuple(value for value in values if value is not None)
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("distribution summary values must be finite")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("distribution minimum must not exceed maximum")
        return self


class SourceGroupCompleteness(BaseModel):
    """Coverage of every discovered source by its explicit source-group map."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["rag_source_group_completeness_v1"]
    manifest_schema_version: str = Field(min_length=1)
    discovered_source_count: int = Field(ge=0)
    mapped_source_count: int = Field(ge=0)
    missing_source_relpaths: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_completeness(self) -> SourceGroupCompleteness:
        if self.mapped_source_count > self.discovered_source_count:
            raise ValueError(
                "mapped source count cannot exceed discovered source count"
            )
        if len(self.missing_source_relpaths) != (
            self.discovered_source_count - self.mapped_source_count
        ):
            raise ValueError("missing source paths must match the coverage counts")
        if len(set(self.missing_source_relpaths)) != len(self.missing_source_relpaths):
            raise ValueError("missing source paths must be unique")
        return self


class SourceChunkStats(BaseModel):
    """Chunk counts and character distributions for one source without text."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["rag_source_chunk_stats_v1"]
    subject: str = Field(min_length=1)
    source_relpath: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    parent_count: int = Field(ge=0)
    child_count: int = Field(ge=0)
    parent_chars: NumericDistribution
    child_chars: NumericDistribution
    parent_overlap_chars: NumericDistribution
    child_overlap_chars: NumericDistribution

    @model_validator(mode="after")
    def _validate_counts(self) -> SourceChunkStats:
        if self.parent_count == 0 or self.child_count == 0:
            raise ValueError("every source must produce at least one parent and child")
        if self.parent_chars.count != self.parent_count:
            raise ValueError("parent distribution count must equal parent_count")
        if self.child_chars.count != self.child_count:
            raise ValueError("child distribution count must equal child_count")
        return self


class SubjectChunkStats(BaseModel):
    """Per-subject aggregation of the real dry-run chunk results."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["rag_subject_chunk_stats_v1"]
    subject: str = Field(min_length=1)
    source_count: int = Field(ge=0)
    parent_count: int = Field(ge=0)
    child_count: int = Field(ge=0)
    parent_chars: NumericDistribution
    child_chars: NumericDistribution
    parent_overlap_chars: NumericDistribution
    child_overlap_chars: NumericDistribution
    sources: tuple[SourceChunkStats, ...]

    @model_validator(mode="after")
    def _validate_subject_aggregation(self) -> SubjectChunkStats:
        if not self.source_count or not self.parent_count or not self.child_count:
            raise ValueError(
                "every subject must produce sources, parents, and children"
            )
        if self.source_count != len(self.sources):
            raise ValueError("source_count must equal the number of source rows")
        if self.parent_count != sum(source.parent_count for source in self.sources):
            raise ValueError("parent_count must equal the source-level total")
        if self.child_count != sum(source.child_count for source in self.sources):
            raise ValueError("child_count must equal the source-level total")
        if self.parent_chars.count != self.parent_count:
            raise ValueError("parent distribution count must equal parent_count")
        if self.child_chars.count != self.child_count:
            raise ValueError("child distribution count must equal child_count")
        return self


class ChunkStatsReport(BaseModel):
    """Strict, content-free output of a complete local Parent-Child dry run."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["rag_chunk_stats_v1"]
    generation_id: str = Field(min_length=1)
    subjects: tuple[SubjectChunkStats, ...]
    source_group_completeness: SourceGroupCompleteness
    total_sources: int = Field(ge=0)
    total_parents: int = Field(ge=0)
    total_children: int = Field(ge=0)
    parent_chars: NumericDistribution
    child_chars: NumericDistribution
    parent_overlap_chars: NumericDistribution
    child_overlap_chars: NumericDistribution
    empty_content_count: int = Field(ge=0)
    orphan_child_count: int = Field(ge=0)
    parent_over_hard_max_count: int = Field(ge=0)
    child_over_hard_max_count: int = Field(ge=0)
    protected_atomic_block_count: int = Field(ge=0)
    protected_atomic_block_violation_count: int = Field(ge=0)
    loader_failure_count: int = Field(ge=0)
    embedding_batch_size: int = Field(gt=0)
    estimated_embedding_batch_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_totals(self) -> ChunkStatsReport:
        subjects = tuple(subject.subject for subject in self.subjects)
        if not subjects or len(set(subjects)) != len(subjects):
            raise ValueError("chunk report subjects must be non-empty and unique")
        if self.total_sources != sum(subject.source_count for subject in self.subjects):
            raise ValueError("total_sources must equal the subject-level total")
        if self.total_parents != sum(subject.parent_count for subject in self.subjects):
            raise ValueError("total_parents must equal the subject-level total")
        if self.total_children != sum(subject.child_count for subject in self.subjects):
            raise ValueError("total_children must equal the subject-level total")
        if self.parent_chars.count != self.total_parents:
            raise ValueError("parent distribution count must equal total_parents")
        if self.child_chars.count != self.total_children:
            raise ValueError("child distribution count must equal total_children")
        expected_batches = math.ceil(self.total_children / self.embedding_batch_size)
        if self.estimated_embedding_batch_count != expected_batches:
            raise ValueError(
                "estimated embedding batches must use configured batch size"
            )
        integrity_counts = (
            self.empty_content_count,
            self.orphan_child_count,
            self.parent_over_hard_max_count,
            self.child_over_hard_max_count,
            self.protected_atomic_block_violation_count,
            self.loader_failure_count,
        )
        if any(integrity_counts):
            raise ValueError(
                "successful chunk stats cannot contain integrity violations"
            )
        return self


def _distribution(values: Iterable[int]) -> NumericDistribution:
    """Return an interpolation-based P95 so one-value series stay well-defined."""

    ordered = sorted(values)
    if not ordered:
        return NumericDistribution(
            schema_version="rag_numeric_distribution_v1",
            count=0,
            minimum=None,
            mean=None,
            median=None,
            p95=None,
            maximum=None,
        )
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return NumericDistribution(
        schema_version="rag_numeric_distribution_v1",
        count=len(ordered),
        minimum=float(ordered[0]),
        mean=float(mean(ordered)),
        median=float(median(ordered)),
        p95=float(p95),
        maximum=float(ordered[-1]),
    )


def _overlaps(spans: Iterable[tuple[int, int]]) -> tuple[int, ...]:
    ordered = sorted(spans)
    return tuple(
        max(0, previous_end - current_start)
        for (_, previous_end), (current_start, _) in zip(
            ordered,
            ordered[1:],
        )
    )


def _doc_type(extension: str) -> str:
    try:
        return _SOURCE_DOC_TYPES[extension]
    except KeyError as exc:
        raise BuildAuditError(
            f"catalog returned unsupported source extension: {extension}"
        ) from exc


def validate_experimental_catalog(config: RagIndexConfig) -> SubjectCatalogSnapshot:
    """Enforce the configured local corpus and exclusion contract."""

    if not config.catalog.data_root.is_absolute():
        raise BuildAuditError("catalog.data_root must be resolved before discovery")
    if not config.storage.index_root.is_absolute():
        raise BuildAuditError("storage.index_root must be resolved before discovery")
    catalog = config.catalog
    missing_exclusions: list[str] = []
    if "evaluation" not in catalog.excluded_exact_names:
        missing_exclusions.append("evaluation")
    if not catalog.exclude_hidden:
        missing_exclusions.append("hidden directories")
    if not catalog.exclude_cache_directories:
        missing_exclusions.append("cache directories")
    if not catalog.cache_directory_names:
        missing_exclusions.append("configured cache directory names")
    if not catalog.exclude_unclassified:
        missing_exclusions.append("unclassified")
    if not catalog.exclude_needs_ocr:
        missing_exclusions.append("_needs_ocr")
    if missing_exclusions:
        raise BuildAuditError(
            "catalog exclusion contract is incomplete: " + ", ".join(missing_exclusions)
        )
    if catalog.symlink_policy != "reject":
        raise BuildAuditError("catalog symlink_policy must be 'reject'")

    snapshot = SubjectCatalog(
        config=catalog,
        subject_policy_map=config.subject_policy_map,
    ).discover()
    if not snapshot.subject_ids():
        raise BuildAuditError("catalog must discover at least one subject")
    return snapshot


def validate_source_group_completeness(
    *,
    project_root: Path,
    source_groups_path: Path,
    catalog_snapshot: SubjectCatalogSnapshot,
) -> SourceGroupCompleteness:
    """Require an explicit group for every discovered catalog source."""

    root = resolve_project_root(project_root)
    manifest = load_source_group_manifest(
        require_project_file(root, source_groups_path)
    )
    discovered = tuple(
        source.source_relpath for source in catalog_snapshot.source_entries()
    )
    missing = tuple(
        source_relpath
        for source_relpath in discovered
        if source_relpath not in manifest.source_groups
    )
    result = SourceGroupCompleteness(
        schema_version="rag_source_group_completeness_v1",
        manifest_schema_version=manifest.schema_version,
        discovered_source_count=len(discovered),
        mapped_source_count=len(discovered) - len(missing),
        missing_source_relpaths=missing,
    )
    if missing:
        raise BuildAuditError(
            "source-group manifest is incomplete for discovered sources: "
            + ", ".join(missing)
        )
    return result


def _source_stats(
    *,
    subject: str,
    source_relpath: str,
    doc_id: str,
    parents: list[ParentRecord],
    children: list[ChildDocument],
) -> SourceChunkStats:
    children_by_parent: dict[str, list[ChildDocument]] = defaultdict(list)
    for child in children:
        children_by_parent[child.metadata.parent_id].append(child)
    child_overlaps = tuple(
        overlap
        for entries in children_by_parent.values()
        for overlap in _overlaps(
            (
                (
                    child.metadata.child_start_in_parent,
                    child.metadata.child_end_in_parent,
                )
                for child in entries
            )
        )
    )
    return SourceChunkStats(
        schema_version="rag_source_chunk_stats_v1",
        subject=subject,
        source_relpath=source_relpath,
        doc_id=doc_id,
        parent_count=len(parents),
        child_count=len(children),
        parent_chars=_distribution(parent.parent_chars for parent in parents),
        child_chars=_distribution(child.metadata.child_chars for child in children),
        parent_overlap_chars=_distribution(
            _overlaps((parent.start_char, parent.end_char) for parent in parents)
        ),
        child_overlap_chars=_distribution(child_overlaps),
    )


def _subject_parent_overlaps(parents: Iterable[ParentRecord]) -> tuple[int, ...]:
    by_doc: dict[str, list[ParentRecord]] = defaultdict(list)
    for parent in parents:
        by_doc[parent.doc_id].append(parent)
    return tuple(
        overlap
        for records in by_doc.values()
        for overlap in _overlaps(
            (parent.start_char, parent.end_char) for parent in records
        )
    )


def _subject_child_overlaps(children: Iterable[ChildDocument]) -> tuple[int, ...]:
    by_parent: dict[str, list[ChildDocument]] = defaultdict(list)
    for child in children:
        by_parent[child.metadata.parent_id].append(child)
    return tuple(
        overlap
        for records in by_parent.values()
        for overlap in _overlaps(
            (
                (
                    child.metadata.child_start_in_parent,
                    child.metadata.child_end_in_parent,
                )
                for child in records
            )
        )
    )


def _atomic_boundary_violation_count(
    *,
    parents: Iterable[ParentRecord],
    children: Iterable[ChildDocument],
    atomic_spans: Iterable[ProtectedAtomicSpan],
) -> int:
    """Count actual parent or child boundaries that bisect protected spans."""

    boundaries = {
        boundary
        for parent in parents
        for boundary in (parent.start_char, parent.end_char)
    }
    boundaries.update(
        boundary
        for child in children
        for boundary in (child.metadata.start_char, child.metadata.end_char)
    )
    ordered_boundaries = sorted(boundaries)
    return sum(
        bisect_left(ordered_boundaries, span.end_char)
        - bisect_right(ordered_boundaries, span.start_char)
        for span in atomic_spans
    )


def collect_chunk_stats(
    *,
    project_root: Path,
    config_path: Path,
    generation_id: str,
    source_groups_path: Path,
    progress: Callable[[str], None] | None = None,
) -> ChunkStatsReport:
    """Execute a complete local loader/splitter dry run and return strict stats."""

    if not generation_id or generation_id != generation_id.strip():
        raise BuildAuditError("generation_id must be a non-empty stripped string")
    root = resolve_project_root(project_root)
    loaded_config = load_rag_index_config(require_project_file(root, config_path))
    config = resolve_rag_index_config_paths(loaded_config, project_root=root)
    snapshot = validate_experimental_catalog(config)
    source_group_completeness = validate_source_group_completeness(
        project_root=root,
        source_groups_path=source_groups_path,
        catalog_snapshot=snapshot,
    )

    all_parents: list[ParentRecord] = []
    all_children: list[ChildDocument] = []
    source_rows: dict[str, list[SourceChunkStats]] = defaultdict(list)
    parent_hard_max_count = 0
    child_hard_max_count = 0
    protected_atomic_block_count = 0
    protected_atomic_block_violation_count = 0
    seen_parent_ids: set[str] = set()
    seen_child_ids: set[str] = set()
    seen_doc_ids: set[str] = set()

    for subject in snapshot.subjects:
        resolved_policy = resolve_subject_chunk_policy(config, subject.subject_id)
        policy = resolved_policy.parent_child_policy
        for source in subject.sources:
            if progress is not None:
                progress(
                    "chunk dry run source="
                    + source.source_relpath
                    + " subject="
                    + subject.subject_id
                )
            try:
                cleaned = load_cleaned_source(
                    SourceEntry(
                        schema_version="source_entry_v1",
                        source_path=source.source_path,
                        data_root=snapshot.data_root,
                        subject=subject.subject_id,
                        doc_type=_doc_type(source.extension),
                    ),
                    resolved_policy.loader_config,
                )
                bundle = build_parent_child_bundle(
                    cleaned,
                    policy,
                    generation_id,
                )
            except Exception as exc:
                raise BuildAuditError(
                    "chunk dry run failed for source "
                    f"'{source.source_relpath}' ({type(exc).__name__})"
                ) from exc

            expected_doc_id = make_doc_id(
                subject=subject.subject_id,
                source_relpath=source.source_relpath,
                file_sha1=cleaned.source_file_sha1,
            )
            if cleaned.doc_id != expected_doc_id:
                raise BuildAuditError("loader returned an unstable doc_id")
            if cleaned.doc_id in seen_doc_ids:
                raise BuildAuditError(
                    "catalog sources produced duplicate doc_id values"
                )
            seen_doc_ids.add(cleaned.doc_id)
            if bundle.generation_id != generation_id:
                raise BuildAuditError("chunk bundle generation_id mismatch")

            parents = list(bundle.parents)
            children = list(bundle.children)
            if not parents or not children:
                raise BuildAuditError(
                    "chunk dry run produced no parent or child for source "
                    f"'{source.source_relpath}'"
                )
            parent_hard_max_count += sum(
                parent.parent_chars > policy.parent_hard_max for parent in parents
            )
            child_hard_max_count += sum(
                child.metadata.child_chars > policy.child_hard_max for child in children
            )
            protected_atomic_spans = detect_protected_atomic_spans(cleaned, policy)
            protected_atomic_block_count += len(protected_atomic_spans)
            atomic_violations = _atomic_boundary_violation_count(
                parents=parents,
                children=children,
                atomic_spans=protected_atomic_spans,
            )
            protected_atomic_block_violation_count += atomic_violations
            if atomic_violations:
                raise BuildAuditError(
                    "protected atomic blocks were split for source "
                    f"'{source.source_relpath}' ({atomic_violations} boundaries)"
                )
            for parent in parents:
                if parent.parent_id in seen_parent_ids:
                    raise BuildAuditError("chunk dry run produced duplicate parent_id")
                if parent.generation_id != generation_id:
                    raise BuildAuditError("parent generation_id mismatch")
                seen_parent_ids.add(parent.parent_id)
            for child in children:
                if child.metadata.child_id in seen_child_ids:
                    raise BuildAuditError("chunk dry run produced duplicate child_id")
                if child.metadata.generation_id != generation_id:
                    raise BuildAuditError("child generation_id mismatch")
                seen_child_ids.add(child.metadata.child_id)
            source_rows[subject.subject_id].append(
                _source_stats(
                    subject=subject.subject_id,
                    source_relpath=source.source_relpath,
                    doc_id=cleaned.doc_id,
                    parents=parents,
                    children=children,
                )
            )
            all_parents.extend(parents)
            all_children.extend(children)

    parent_by_id = {parent.parent_id: parent for parent in all_parents}
    orphan_child_count = sum(
        child.metadata.parent_id not in parent_by_id for child in all_children
    )
    if orphan_child_count:
        raise BuildAuditError("chunk dry run produced orphan children")

    parents_by_subject: dict[str, list[ParentRecord]] = defaultdict(list)
    children_by_subject: dict[str, list[ChildDocument]] = defaultdict(list)
    for parent in all_parents:
        parents_by_subject[parent.subject].append(parent)
    for child in all_children:
        children_by_subject[child.metadata.subject].append(child)
    zero_output_subjects = tuple(
        subject_id
        for subject_id in snapshot.subject_ids()
        if not parents_by_subject[subject_id] or not children_by_subject[subject_id]
    )
    if zero_output_subjects:
        raise BuildAuditError(
            "one or more subjects produced no parents or children: "
            + ", ".join(zero_output_subjects)
        )
    empty_content_count = sum(
        not parent.content.strip() for parent in all_parents
    ) + sum(not child.content.strip() for child in all_children)
    if empty_content_count:
        raise BuildAuditError("chunk dry run produced empty parent or child content")
    if parent_hard_max_count or child_hard_max_count:
        raise BuildAuditError(
            "chunk dry run produced chunks above a configured hard max"
        )
    subjects = tuple(
        SubjectChunkStats(
            schema_version="rag_subject_chunk_stats_v1",
            subject=subject_id,
            source_count=len(source_rows[subject_id]),
            parent_count=len(parents_by_subject[subject_id]),
            child_count=len(children_by_subject[subject_id]),
            parent_chars=_distribution(
                parent.parent_chars for parent in parents_by_subject[subject_id]
            ),
            child_chars=_distribution(
                child.metadata.child_chars for child in children_by_subject[subject_id]
            ),
            parent_overlap_chars=_distribution(
                _subject_parent_overlaps(parents_by_subject[subject_id])
            ),
            child_overlap_chars=_distribution(
                _subject_child_overlaps(children_by_subject[subject_id])
            ),
            sources=tuple(
                sorted(source_rows[subject_id], key=lambda row: row.source_relpath)
            ),
        )
        for subject_id in snapshot.subject_ids()
    )
    return ChunkStatsReport(
        schema_version="rag_chunk_stats_v1",
        generation_id=generation_id,
        subjects=subjects,
        source_group_completeness=source_group_completeness,
        total_sources=len(seen_doc_ids),
        total_parents=len(all_parents),
        total_children=len(all_children),
        parent_chars=_distribution(parent.parent_chars for parent in all_parents),
        child_chars=_distribution(child.metadata.child_chars for child in all_children),
        parent_overlap_chars=_distribution(_subject_parent_overlaps(all_parents)),
        child_overlap_chars=_distribution(_subject_child_overlaps(all_children)),
        empty_content_count=empty_content_count,
        orphan_child_count=orphan_child_count,
        parent_over_hard_max_count=parent_hard_max_count,
        child_over_hard_max_count=child_hard_max_count,
        protected_atomic_block_count=protected_atomic_block_count,
        protected_atomic_block_violation_count=protected_atomic_block_violation_count,
        loader_failure_count=0,
        embedding_batch_size=config.embedding.batch_size,
        estimated_embedding_batch_count=math.ceil(
            len(all_children) / config.embedding.batch_size
        ),
    )


def _format_distribution(distribution: NumericDistribution) -> str:
    if distribution.count == 0:
        return "n=0"
    return (
        f"n={distribution.count}; min={distribution.minimum:.0f}; "
        f"mean={distribution.mean:.2f}; median={distribution.median:.2f}; "
        f"p95={distribution.p95:.2f}; max={distribution.maximum:.0f}"
    )


def render_chunk_stats_markdown(stats: ChunkStatsReport) -> str:
    """Render a bounded report without source or model-response text."""

    lines = [
        "# RAG chunk statistics",
        "",
        f"- generation_id: `{stats.generation_id}`",
        f"- discovered sources: {stats.total_sources}",
        f"- parents: {stats.total_parents}",
        f"- children awaiting embeddings: {stats.total_children}",
        f"- configured embedding batch size: {stats.embedding_batch_size}",
        f"- estimated embedding batches: {stats.estimated_embedding_batch_count}",
        (
            "- source-group coverage: "
            f"{stats.source_group_completeness.mapped_source_count}/"
            f"{stats.source_group_completeness.discovered_source_count}"
        ),
        "",
        "## Per-subject counts",
        "",
        "| Subject | Sources | Parents | Children |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {subject.subject} | {subject.source_count} | "
        f"{subject.parent_count} | {subject.child_count} |"
        for subject in stats.subjects
    )
    lines.extend(
        [
            "",
            "## Character and overlap distributions",
            "",
            f"- parent chars: {_format_distribution(stats.parent_chars)}",
            f"- child chars: {_format_distribution(stats.child_chars)}",
            (
                "- parent overlap chars: "
                f"{_format_distribution(stats.parent_overlap_chars)}"
            ),
            (
                "- child overlap chars (within parent): "
                f"{_format_distribution(stats.child_overlap_chars)}"
            ),
            "",
            "## Integrity checks",
            "",
            f"- empty content: {stats.empty_content_count}",
            f"- orphan children: {stats.orphan_child_count}",
            f"- parents over hard max: {stats.parent_over_hard_max_count}",
            f"- children over hard max: {stats.child_over_hard_max_count}",
            f"- protected atomic blocks checked: {stats.protected_atomic_block_count}",
            (
                "- protected atomic block violations: "
                f"{stats.protected_atomic_block_violation_count}"
            ),
            f"- loader failures: {stats.loader_failure_count}",
            "",
            "## Per-source chunk counts",
            "",
            "| Subject | Source | Parents | Children |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {subject.subject} | `{source.source_relpath}` | "
        f"{source.parent_count} | {source.child_count} |"
        for subject in stats.subjects
        for source in subject.sources
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "BuildAuditError",
    "ChunkStatsReport",
    "NumericDistribution",
    "SourceChunkStats",
    "SourceGroupCompleteness",
    "SubjectChunkStats",
    "collect_chunk_stats",
    "render_chunk_stats_markdown",
    "validate_experimental_catalog",
    "validate_source_group_completeness",
]
