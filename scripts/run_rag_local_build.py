"""Run a strictly isolated local experimental RAG build.

This command is deliberately a build-time tool rather than a runtime switch.  It
never activates, rolls back, shadows, or loads an active generation.  A normal
``--execute`` run only proceeds forward through the declared build stages; an
error records a redacted failure report and stops later stages.  ``--offline-
dry-run`` exercises the real catalog, loader, and splitter without contacting a
provider or touching Chroma, the generation registry, or an existing generation.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from time import perf_counter_ns
from typing import TYPE_CHECKING, Literal

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.audit_rag_readiness import run_audit
from scripts.build_flat_baseline import build_flat_baseline
from scripts.build_parent_child_generation import run_build
from src.config.rag_index_config import (
    RagIndexConfig,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.config.rag_benchmark_config import load_rag_benchmark_config
from src.rag.parent_child._storage_io import validate_generation_id
from src.rag.parent_child.project_paths import (
    atomic_write_project_bytes,
    require_project_file,
    resolve_project_path,
    resolve_project_root,
)
from src.rag.gold_dataset import load_gold_dataset
from src.rag.subject_catalog import SubjectCatalog, SubjectCatalogSnapshot


if TYPE_CHECKING:
    from src.rag.parent_child.provider_probe import LlmProbeConfig
    from src.rag.parent_child.runtime_loader import LoadedGenerationRuntime


_BUILD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
_REQUIRED_BUILD_MODULES: tuple[str, ...] = (
    "chromadb",
    "fitz",
    "httpx",
    "jieba",
    "rank_bm25",
    "yaml",
)


class LocalBuildError(RuntimeError):
    """The requested local build cannot safely continue."""


class ProviderProbeFailed(LocalBuildError):
    """A strict provider probe completed but did not pass."""


class _StrictModel(BaseModel):
    """Strict, frozen report models; reports never rely on loose dictionaries."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


BuildMode = Literal["plan", "execute", "offline_dry_run"]
RunStatus = Literal["planned", "success", "failed", "offline_complete"]
StageStatus = Literal["completed", "failed", "not_run"]
BuildStage = Literal[
    "preflight",
    "catalog",
    "source_groups_and_readiness",
    "provider_probe",
    "chunk_dry_run",
    "flat_baseline",
    "parent_child_generation",
    "artifact_validation",
    "smoke_retrieval",
    "grounded_llm_smoke",
]


class SecretPresenceEntry(_StrictModel):
    """One environment-variable presence result, never its value."""

    name: str = Field(min_length=1)
    present: bool


class SecretPresenceReport(_StrictModel):
    schema_version: Literal["rag_build_secret_presence_v1"]
    entries: tuple[SecretPresenceEntry, ...]


class DependencyReport(_StrictModel):
    """A content-free record of local dependencies required by this build."""

    schema_version: Literal["rag_build_dependency_report_v1"]
    required_modules: tuple[str, ...]
    missing_modules: tuple[str, ...]


class CatalogSubjectSummary(_StrictModel):
    subject_id: str = Field(min_length=1)
    source_file_count: int = Field(gt=0)


class CatalogSummary(_StrictModel):
    schema_version: Literal["rag_build_catalog_summary_v1"]
    subjects: tuple[CatalogSubjectSummary, ...]


class ReadinessSummary(_StrictModel):
    schema_version: Literal["rag_build_readiness_summary_v1"]
    audit_completed: bool
    evaluation_eligible: bool
    source_group_complete: bool
    global_blockers: tuple[str, ...]
    audit_failure_type: str | None


class PreflightReport(_StrictModel):
    schema_version: Literal["rag_build_preflight_v1"]
    run_id: str = Field(min_length=1)
    mode: BuildMode
    project_root: str
    index_config_path: str
    gold_dataset_path: str
    benchmark_config_path: str
    report_directory: str
    flat_build_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    requested_code_revision: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    head_code_revision: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    revision_matches_head: bool
    catalog_data_root: str
    storage_index_root: str
    registry_path: str
    dependencies: DependencyReport
    secrets: SecretPresenceReport
    experimental_only: Literal[True]
    activation_prohibited: Literal[True]


class StageRecord(_StrictModel):
    stage: BuildStage
    status: StageStatus
    duration_ms: float = Field(ge=0.0)
    failure_type: str | None


class FailureSummary(_StrictModel):
    stage: BuildStage
    error_type: str = Field(min_length=1)


class FlatBaselineSummary(_StrictModel):
    build_id: str = Field(min_length=1)
    chroma_path: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    collection_name: str = Field(min_length=3)
    chunk_count: int = Field(gt=0)
    sampled_chunk_count: int = Field(ge=1)
    similarity_query_succeeded: Literal[True]


class GenerationSummary(_StrictModel):
    generation_id: str = Field(min_length=1)
    generation_path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_state: Literal["READY"]
    active: Literal[False]
    orphan_child_count: Literal[0]
    hydration_failure_count: Literal[0]
    child_count: int = Field(gt=0)
    parent_count: int = Field(gt=0)


class SmokeHit(_StrictModel):
    rank: int = Field(gt=0)
    source_relpath: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: tuple[str, ...]
    score: float | None
    reranker_score: float | None
    parent_hydrated: bool


class SmokeChannelResult(_StrictModel):
    status: Literal["ok", "empty", "not_run"]
    hits: tuple[SmokeHit, ...]
    latency_ms: float = Field(ge=0.0)
    context_token_estimate: int = Field(ge=0)
    parent_hydration_success_count: int = Field(ge=0)


class SmokeRetrievalRecord(_StrictModel):
    query_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    flat_baseline: SmokeChannelResult
    parent_child_candidate: SmokeChannelResult


class SmokeRetrievalArtifact(_StrictModel):
    schema_version: Literal["rag_smoke_retrieval_v1"]
    status: Literal["completed", "not_run"]
    reason: str | None
    generation_id: str | None
    records: tuple[SmokeRetrievalRecord, ...]


class GroundedCitation(_StrictModel):
    source_relpath: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class GroundedSmokeRecord(_StrictModel):
    query_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sources: tuple[GroundedCitation, ...]
    answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_chars: int = Field(ge=1)
    citations: tuple[GroundedCitation, ...]
    context_token_estimate: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    evidence_insufficient: bool
    answer_supported: bool
    citation_supported: bool
    hallucination_suspected: bool


class GroundedSmokeArtifact(_StrictModel):
    schema_version: Literal["rag_llm_grounded_smoke_v1"]
    status: Literal["completed", "not_run"]
    reason: str | None
    records: tuple[GroundedSmokeRecord, ...]
    private_output_written: bool


class _GroundedAnswerPayload(_StrictModel):
    """Exact external chat contract; no JSON repair or key aliasing is allowed."""

    answer: str = Field(min_length=1)
    citations: list[GroundedCitation]
    evidence_insufficient: bool


class _GroundedReviewPayload(_StrictModel):
    """Exact reviewer contract for smoke-only evidence assessment."""

    answer_supported: bool
    citation_supported: bool
    hallucination_suspected: bool


class LocalBuildReport(_StrictModel):
    """The safe final run envelope for success, failure, and offline runs."""

    schema_version: Literal["rag_local_build_report_v1"]
    run_id: str = Field(min_length=1)
    mode: BuildMode
    status: RunStatus
    requested_code_revision: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    head_code_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{7,64}$")
    revision_matches_head: bool | None
    runtime_config_path: str = Field(min_length=1)
    gold_dataset_path: str = Field(min_length=1)
    catalog: CatalogSummary | None
    readiness: ReadinessSummary | None
    secrets: SecretPresenceReport | None
    flat_baseline: FlatBaselineSummary | None
    generation: GenerationSummary | None
    smoke_retrieval_path: str = Field(min_length=1)
    grounded_smoke_path: str = Field(min_length=1)
    stages: tuple[StageRecord, ...]
    failure: FailureSummary | None
    experimental_only: Literal[True]
    activation_prohibited: Literal[True]

    @model_validator(mode="after")
    def _validate_status_contract(self) -> "LocalBuildReport":
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed local builds require a typed failure summary")
        if self.status != "failed" and self.failure is not None:
            raise ValueError("non-failed local builds cannot contain a failure summary")
        if self.status == "success" and self.generation is None:
            raise ValueError("successful local builds require a READY generation")
        if self.status == "success" and self.flat_baseline is None:
            raise ValueError("successful local builds require a flat baseline")
        return self


@dataclass(frozen=True, slots=True)
class BuildInputs:
    project_root: Path
    index_config: Path
    benchmark_config: Path
    gold_dataset: Path
    build_id: str
    generation_id: str
    code_revision: str
    mode: BuildMode
    run_id: str
    private_smoke_output: Path | None
    llm_provider: str | None
    llm_protocol: str | None
    llm_model: str | None
    llm_base_url: str | None
    llm_endpoint_path: str | None
    llm_api_key_env: str | None
    llm_timeout_seconds: float | None


@dataclass(frozen=True, slots=True)
class BuildContext:
    inputs: BuildInputs
    root: Path
    index_config_path: Path
    gold_dataset_path: Path
    benchmark_config_path: Path
    source_groups_path: Path
    report_directory: Path
    config: RagIndexConfig
    head_revision: str


@dataclass(frozen=True, slots=True)
class SmokeQuery:
    """A GoldDataset-derived smoke query that is never serialized verbatim."""

    query_id: str
    subject: str
    query: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--gold-dataset", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--private-smoke-output", type=Path)
    parser.add_argument("--llm-provider")
    parser.add_argument("--llm-protocol")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-endpoint-path")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-timeout-seconds", type=float)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--offline-dry-run", action="store_true")
    return parser


def _utc_run_id() -> str:
    return "rag_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _BUILD_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}")
    return value


def _build_inputs(args: argparse.Namespace) -> BuildInputs:
    mode: BuildMode
    if args.execute:
        mode = "execute"
    elif args.offline_dry_run:
        mode = "offline_dry_run"
    else:
        mode = "plan"
    build_id = _validate_identifier(args.build_id, field_name="build_id")
    generation_id = validate_generation_id(args.generation_id)
    run_id = validate_generation_id(args.run_id or _utc_run_id())
    if (
        not isinstance(args.code_revision, str)
        or _REVISION_PATTERN.fullmatch(args.code_revision) is None
    ):
        raise ValueError("code_revision must be a lowercase Git SHA prefix or SHA")
    return BuildInputs(
        project_root=args.project_root,
        index_config=args.index_config,
        benchmark_config=args.benchmark_config,
        gold_dataset=args.gold_dataset,
        build_id=build_id,
        generation_id=generation_id,
        code_revision=args.code_revision,
        mode=mode,
        run_id=run_id,
        private_smoke_output=args.private_smoke_output,
        llm_provider=args.llm_provider,
        llm_protocol=args.llm_protocol,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_endpoint_path=args.llm_endpoint_path,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout_seconds=args.llm_timeout_seconds,
    )


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _current_head(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalBuildError("GitRevisionUnavailable") from exc
    head = completed.stdout.strip()
    if _REVISION_PATTERN.fullmatch(head) is None:
        raise LocalBuildError("GitRevisionInvalid")
    return head


def _secret_presence(
    config: RagIndexConfig, *, llm_api_key_env: str | None = None
) -> SecretPresenceReport:
    required = (
        "RAG_EMBEDDING_API_KEY",
        "RAG_RERANKER_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        config.embedding.api_key_env,
        config.reranker.api_key_env,
        *(() if llm_api_key_env is None else (llm_api_key_env,)),
    )
    names = tuple(dict.fromkeys(required))
    return SecretPresenceReport(
        schema_version="rag_build_secret_presence_v1",
        entries=tuple(
            SecretPresenceEntry(
                name=name,
                present=bool((os.environ.get(name) or "").strip()),
            )
            for name in names
        ),
    )


def _dependency_report() -> DependencyReport:
    """Report required local imports before a provider or index side effect."""

    missing = tuple(
        module_name
        for module_name in _REQUIRED_BUILD_MODULES
        if find_spec(module_name) is None
    )
    return DependencyReport(
        schema_version="rag_build_dependency_report_v1",
        required_modules=_REQUIRED_BUILD_MODULES,
        missing_modules=missing,
    )


def _require_build_dependencies(report: DependencyReport) -> None:
    """Fail preflight explicitly instead of choosing a degraded local path."""

    if report.missing_modules:
        raise LocalBuildError("RequiredBuildDependencyMissing")


def _collection_name(build_id: str) -> str:
    # ``build_id`` is already strictly validated. Dots are invalid for this
    # collection contract, so this is a deterministic identifier derivation,
    # not a provider/model default or a fallback.
    value = "flat_" + build_id.replace(".", "_")
    if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise LocalBuildError("DerivedCollectionNameInvalid")
    return value


def _prepare_context(inputs: BuildInputs) -> BuildContext:
    root = resolve_project_root(inputs.project_root)
    index_config_path = require_project_file(root, inputs.index_config)
    gold_dataset_path = require_project_file(root, inputs.gold_dataset)
    benchmark_config_path = require_project_file(root, inputs.benchmark_config)
    benchmark_config = load_rag_benchmark_config(benchmark_config_path)
    source_groups_path = require_project_file(
        root, benchmark_config.source_group_manifest_path
    )
    config = resolve_rag_index_config_paths(
        load_rag_index_config(index_config_path), project_root=root
    )
    report_directory = resolve_project_path(
        root,
        Path("reports") / "rag_build" / inputs.run_id,
        must_exist=False,
    )
    return BuildContext(
        inputs=inputs,
        root=root,
        index_config_path=index_config_path,
        gold_dataset_path=gold_dataset_path,
        benchmark_config_path=benchmark_config_path,
        source_groups_path=source_groups_path,
        report_directory=report_directory,
        config=config,
        head_revision=_current_head(root),
    )


def _json_bytes(model: BaseModel) -> bytes:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_model(root: Path, output: Path, model: BaseModel) -> Path:
    return atomic_write_project_bytes(
        root,
        Path(_relative(root, output)),
        _json_bytes(model),
        overwrite=True,
    )


def _safe_failure_type(exc: BaseException) -> str:
    """Return only an exception class name; exception text can contain secrets."""

    return type(exc).__name__[:128] or "UnknownFailure"


def _new_stage(
    *,
    stage: BuildStage,
    status: StageStatus,
    started_ns: int,
    exc: BaseException | None,
) -> StageRecord:
    return StageRecord(
        stage=stage,
        status=status,
        duration_ms=(perf_counter_ns() - started_ns) / 1_000_000.0,
        failure_type=_safe_failure_type(exc) if exc is not None else None,
    )


def _preflight_report(context: BuildContext) -> PreflightReport:
    config = context.config
    return PreflightReport(
        schema_version="rag_build_preflight_v1",
        run_id=context.inputs.run_id,
        mode=context.inputs.mode,
        project_root=".",
        index_config_path=_relative(context.root, context.index_config_path),
        gold_dataset_path=_relative(context.root, context.gold_dataset_path),
        benchmark_config_path=_relative(context.root, context.benchmark_config_path),
        report_directory=_relative(context.root, context.report_directory),
        flat_build_id=context.inputs.build_id,
        generation_id=context.inputs.generation_id,
        requested_code_revision=context.inputs.code_revision,
        head_code_revision=context.head_revision,
        revision_matches_head=(context.inputs.code_revision == context.head_revision),
        catalog_data_root=_relative(context.root, config.catalog.data_root),
        storage_index_root=_relative(context.root, config.storage.index_root),
        registry_path=_relative(context.root, config.storage.resolved_registry_path()),
        dependencies=_dependency_report(),
        secrets=_secret_presence(
            config, llm_api_key_env=context.inputs.llm_api_key_env
        ),
        experimental_only=True,
        activation_prohibited=True,
    )


def _catalog_summary(
    context: BuildContext,
) -> tuple[CatalogSummary, SubjectCatalogSnapshot]:
    snapshot = SubjectCatalog(
        config=context.config.catalog,
        subject_policy_map=context.config.subject_policy_map,
    ).discover()
    return (
        CatalogSummary(
            schema_version="rag_build_catalog_summary_v1",
            subjects=tuple(
                CatalogSubjectSummary(
                    subject_id=subject.subject_id,
                    source_file_count=len(subject.sources),
                )
                for subject in snapshot.subjects
            ),
        ),
        snapshot,
    )


def _readiness_summary(context: BuildContext) -> ReadinessSummary:
    """Run readiness as a non-blocking activation diagnostic.

    A readiness blocker makes the result experimental-only but must not turn a
    technically valid local index build into a fabricated failure.
    """

    output = context.report_directory / "readiness.json"
    artifact = run_audit(
        project_root=context.root,
        index_config_path=context.index_config_path,
        benchmark_config_path=context.benchmark_config_path,
        gold_dataset_path=context.gold_dataset_path,
        output_path=output,
        overwrite=True,
    )
    report = artifact.report
    source_group_complete = all(
        "source_group_manifest_incomplete" not in subject.blockers
        for subject in report.subjects
    )
    if not source_group_complete:
        raise LocalBuildError("SourceGroupManifestIncomplete")
    return ReadinessSummary(
        schema_version="rag_build_readiness_summary_v1",
        audit_completed=True,
        evaluation_eligible=not report.production_recommendation_blocked,
        source_group_complete=source_group_complete,
        global_blockers=report.global_blockers,
        audit_failure_type=None,
    )


def _select_gold_smoke_queries(
    *, context: BuildContext, catalog: SubjectCatalogSnapshot
) -> tuple[SmokeQuery, ...]:
    """Select exactly three existing GoldDataset queries per discovered subject.

    This deliberately derives all subjects and query text from caller-supplied
    data.  It avoids an in-code course list, and reports only stable query IDs.
    """

    dataset = load_gold_dataset(context.gold_dataset_path)
    known_subjects = set(catalog.subject_ids())
    grouped: dict[str, list[SmokeQuery]] = {
        subject.subject_id: [] for subject in catalog.subjects
    }
    for item in dataset.queries:
        if item.subject not in known_subjects:
            raise LocalBuildError("GoldDatasetSubjectNotInCatalog")
        grouped[item.subject].append(
            SmokeQuery(
                query_id=item.query_id,
                subject=item.subject,
                query=item.query,
            )
        )
    selected: list[SmokeQuery] = []
    for subject in catalog.subjects:
        candidates = grouped[subject.subject_id]
        if len(candidates) < 3:
            raise LocalBuildError("InsufficientGoldQueriesForSmoke")
        selected.extend(candidates[:3])
    return tuple(selected)


def _placeholder_smoke_artifacts(
    context: BuildContext, *, reason: str
) -> tuple[Path, Path]:
    retrieval_path = context.report_directory / "smoke_retrieval.json"
    grounded_path = context.report_directory / "llm_grounded_smoke.json"

    def write_if_not_completed(
        path: Path, model: BaseModel, model_type: type[BaseModel]
    ) -> None:
        if path.is_file():
            try:
                existing = model_type.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise LocalBuildError("ExistingSmokeArtifactContractInvalid") from exc
            if getattr(existing, "status", None) == "completed":
                return
        _write_model(context.root, path, model)

    write_if_not_completed(
        retrieval_path,
        SmokeRetrievalArtifact(
            schema_version="rag_smoke_retrieval_v1",
            status="not_run",
            reason=reason,
            generation_id=None,
            records=(),
        ),
        SmokeRetrievalArtifact,
    )
    write_if_not_completed(
        grounded_path,
        GroundedSmokeArtifact(
            schema_version="rag_llm_grounded_smoke_v1",
            status="not_run",
            reason=reason,
            records=(),
            private_output_written=False,
        ),
        GroundedSmokeArtifact,
    )
    return retrieval_path, grounded_path


def _validate_new_targets(context: BuildContext) -> None:
    """Refuse reuse before any external request or registry mutation."""

    flat_root = resolve_project_path(
        context.root,
        Path("artifacts") / "rag" / context.inputs.build_id,
        must_exist=False,
    )
    if flat_root.exists():
        raise FileExistsError("FlatBaselineTargetAlreadyExists")
    generation_final = context.config.storage.index_root / context.inputs.generation_id
    generation_staging = (
        context.config.storage.index_root / ".staging" / context.inputs.generation_id
    )
    if generation_final.exists() or generation_staging.exists():
        raise FileExistsError("GenerationTargetAlreadyExists")


def _read_optional_report_artifact(
    context: BuildContext,
    *,
    filename: str,
    model_type: type[BaseModel],
) -> BaseModel | None:
    """Load one local, strict, content-free report when the stage wrote it.

    A missing artifact is meaningful for a stage that was never reached.  An
    existing artifact must validate exactly; the human report must not conceal a
    corrupt or schema-drifted machine report behind a prose approximation.
    """

    candidate = resolve_project_path(
        context.root,
        context.report_directory / filename,
        must_exist=False,
    )
    if not candidate.exists():
        return None
    try:
        path = require_project_file(context.root, candidate)
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LocalBuildError("BuildReportArtifactContractInvalid") from exc


def _report_artifact_exists(context: BuildContext, *, filename: str) -> bool:
    """Check a report path with the same project-containment rules as reads."""

    candidate = resolve_project_path(
        context.root,
        context.report_directory / filename,
        must_exist=False,
    )
    return candidate.is_file()


def _stage_status(
    report: LocalBuildReport, stage_name: BuildStage
) -> StageStatus | None:
    for stage in report.stages:
        if stage.stage == stage_name:
            return stage.status
    return None


def _format_probe_value(value: object) -> str:
    """Format only typed scalar probe facts for a safe Markdown report."""

    if value is None:
        return "`not observed`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, float):
        return f"`{value:.6f}`"
    return f"`{value}`"


def _render_build_markdown(
    context: BuildContext,
    report: LocalBuildReport,
) -> str:
    """Render a content-free, secret-free human report from the strict model."""

    lines = [
        "# Local experimental RAG build report",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Status: `{report.status}`",
        f"- Requested code revision: `{report.requested_code_revision}`",
        f"- Actual HEAD: `{report.head_code_revision or 'not checked in plan mode'}`",
        "- Revision matches HEAD: "
        f"`{str(report.revision_matches_head).lower() if report.revision_matches_head is not None else 'not checked in plan mode'}`",
        f"- Runtime config: `{report.runtime_config_path}`",
        f"- Gold dataset: `{report.gold_dataset_path}`",
        "- Experimental only: `true`",
        "- Activation prohibited: `true`",
        "",
        "## Stages",
        "",
    ]
    for stage in report.stages:
        suffix = f" ({stage.failure_type})" if stage.failure_type else ""
        lines.append(
            f"- `{stage.stage}`: `{stage.status}` ({stage.duration_ms:.2f} ms){suffix}"
        )

    preflight = _read_optional_report_artifact(
        context,
        filename="preflight.json",
        model_type=PreflightReport,
    )
    if preflight is not None:
        if not isinstance(preflight, PreflightReport):
            raise LocalBuildError("PreflightReportTypeInvalid")
        lines.extend(
            [
                "",
                "## Preflight",
                "",
                f"- Data root: `{preflight.catalog_data_root}`",
                f"- Parent-child index root: `{preflight.storage_index_root}`",
                f"- Registry path: `{preflight.registry_path}`",
                "- Missing required local modules: "
                + (
                    "`none`"
                    if not preflight.dependencies.missing_modules
                    else ", ".join(
                        f"`{item}`" for item in preflight.dependencies.missing_modules
                    )
                ),
            ]
        )
    if report.catalog is not None:
        lines.extend(["", "## Catalog", ""])
        for item in report.catalog.subjects:
            lines.append(f"- `{item.subject_id}`: {item.source_file_count} sources")
    if report.readiness is not None:
        lines.extend(
            [
                "",
                "## Readiness",
                "",
                f"- Evaluation eligible: `{str(report.readiness.evaluation_eligible).lower()}`",
                f"- Source groups complete: `{str(report.readiness.source_group_complete).lower()}`",
            ]
        )
        if report.readiness.audit_failure_type is not None:
            lines.append(
                f"- Audit failure type: `{report.readiness.audit_failure_type}`"
            )
    if report.secrets is not None:
        lines.extend(["", "## Secret presence", ""])
        for entry in report.secrets.entries:
            lines.append(f"- `{entry.name}`: present={str(entry.present).lower()}")

    lines.extend(["", "## Provider probes", ""])
    if not _report_artifact_exists(context, filename="provider_probe.json"):
        lines.append(
            "- Provider probe: `not run` "
            f"(stage={_stage_status(report, 'provider_probe') or 'not recorded'})"
        )
    else:
        from src.rag.parent_child.provider_probe import ProviderProbeReport

        provider_probe = _read_optional_report_artifact(
            context,
            filename="provider_probe.json",
            model_type=ProviderProbeReport,
        )
        if not isinstance(provider_probe, ProviderProbeReport):
            raise LocalBuildError("ProviderProbeReportTypeInvalid")
        embedding = provider_probe.embedding
        reranker = provider_probe.reranker
        llm = provider_probe.llm
        lines.extend(
            [
                "- Embedding: "
                f"status=`{embedding.status}`, provider=`{embedding.provider}`, "
                f"model=`{embedding.model}`, http={_format_probe_value(embedding.http_status)}, "
                f"dimension={_format_probe_value(embedding.actual_dimension)}, "
                f"batch_supported={_format_probe_value(embedding.batch_supported)}, "
                f"input_type_supported={_format_probe_value(embedding.input_type_supported)}, "
                f"failure={_format_probe_value(embedding.failure_type)}",
                "- Reranker: "
                f"status=`{reranker.status}`, provider=`{reranker.provider}`, "
                f"model=`{reranker.model}`, http={_format_probe_value(reranker.http_status)}, "
                "complete_unique_indices="
                f"{_format_probe_value(reranker.returned_indices_complete_unique)}, "
                f"score_min={_format_probe_value(reranker.score_min)}, "
                f"score_max={_format_probe_value(reranker.score_max)}, "
                "relevant_above_irrelevant="
                f"{_format_probe_value(reranker.relevant_documents_above_irrelevant)}, "
                f"failure={_format_probe_value(reranker.failure_type)}",
                "- Chat LLM: "
                f"status=`{llm.status}`, provider={_format_probe_value(llm.provider)}, "
                f"model={_format_probe_value(llm.model)}, http={_format_probe_value(llm.http_status)}, "
                f"real_text_returned={_format_probe_value(llm.real_text_returned)}, "
                f"failure={_format_probe_value(llm.failure_type)}",
            ]
        )

    lines.extend(["", "## Real chunk dry run", ""])
    if not _report_artifact_exists(context, filename="chunk_stats.json"):
        lines.append(
            "- Chunk dry run: `not run` "
            f"(stage={_stage_status(report, 'chunk_dry_run') or 'not recorded'})"
        )
    else:
        from src.rag.parent_child.build_audit import ChunkStatsReport

        chunk_stats = _read_optional_report_artifact(
            context,
            filename="chunk_stats.json",
            model_type=ChunkStatsReport,
        )
        if not isinstance(chunk_stats, ChunkStatsReport):
            raise LocalBuildError("ChunkStatsReportTypeInvalid")
        for subject in chunk_stats.subjects:
            lines.append(
                f"- `{subject.subject}`: {subject.source_count} sources, "
                f"{subject.parent_count} parents, {subject.child_count} children"
            )
        lines.extend(
            [
                f"- Planned child embedding count: `{chunk_stats.total_children}`",
                f"- Estimated embedding batches: `{chunk_stats.estimated_embedding_batch_count}` "
                f"(configured batch size `{chunk_stats.embedding_batch_size}`)",
                f"- Empty content: `{chunk_stats.empty_content_count}`",
                f"- Orphan children: `{chunk_stats.orphan_child_count}`",
                "- Over-hard-max parents/children: "
                f"`{chunk_stats.parent_over_hard_max_count}`/"
                f"`{chunk_stats.child_over_hard_max_count}`",
                "- Protected atomic blocks/violations: "
                f"`{chunk_stats.protected_atomic_block_count}`/"
                f"`{chunk_stats.protected_atomic_block_violation_count}`",
                f"- Loader failures: `{chunk_stats.loader_failure_count}`",
            ]
        )
    if report.flat_baseline is not None:
        flat = report.flat_baseline
        lines.extend(
            [
                "",
                "## Flat baseline",
                "",
                f"- Chroma: `{flat.chroma_path}`",
                f"- Manifest: `{flat.manifest_path}`",
                f"- Collection: `{flat.collection_name}`",
                f"- Count: `{flat.chunk_count}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Flat baseline",
                "",
                "- Not built; no local Chroma count, manifest, or vector validation exists for this run.",
            ]
        )
    if report.generation is not None:
        generation = report.generation
        lines.extend(
            [
                "",
                "## Parent-child generation",
                "",
                f"- ID: `{generation.generation_id}`",
                f"- Registry state: `{generation.registry_state}`",
                "- Active: `false`",
                f"- Orphan children: `{generation.orphan_child_count}`",
                f"- Hydration failures: `{generation.hydration_failure_count}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Parent-child generation",
                "",
                "- Not built; no registry row, READY state, Parent Store, BM25 artifact, or generation Chroma validation exists for this run.",
                "- Candidate active: `false` (no candidate was created or routed).",
            ]
        )

    smoke = _read_optional_report_artifact(
        context,
        filename="smoke_retrieval.json",
        model_type=SmokeRetrievalArtifact,
    )
    grounded = _read_optional_report_artifact(
        context,
        filename="llm_grounded_smoke.json",
        model_type=GroundedSmokeArtifact,
    )
    lines.extend(
        [
            "",
            "## Smoke artifacts",
            "",
            f"- Retrieval: `{report.smoke_retrieval_path}`",
            f"- Grounded LLM: `{report.grounded_smoke_path}`",
        ]
    )
    if smoke is not None:
        if not isinstance(smoke, SmokeRetrievalArtifact):
            raise LocalBuildError("SmokeRetrievalArtifactTypeInvalid")
        flat_hits = sum(len(item.flat_baseline.hits) for item in smoke.records)
        candidate_hits = sum(
            len(item.parent_child_candidate.hits) for item in smoke.records
        )
        hydrated = sum(
            item.parent_child_candidate.parent_hydration_success_count
            for item in smoke.records
        )
        lines.append(
            "- Retrieval outcome: "
            f"status=`{smoke.status}`, records=`{len(smoke.records)}`, "
            f"flat_hits=`{flat_hits}`, candidate_hits=`{candidate_hits}`, "
            f"candidate_hydrations=`{hydrated}`, reason={_format_probe_value(smoke.reason)}"
        )
    if grounded is not None:
        if not isinstance(grounded, GroundedSmokeArtifact):
            raise LocalBuildError("GroundedSmokeArtifactTypeInvalid")
        answer_supported = sum(item.answer_supported for item in grounded.records)
        citation_supported = sum(item.citation_supported for item in grounded.records)
        hallucination_suspected = sum(
            item.hallucination_suspected for item in grounded.records
        )
        lines.append(
            "- Grounded LLM outcome: "
            f"status=`{grounded.status}`, records=`{len(grounded.records)}`, "
            f"answer_supported=`{answer_supported}`, "
            f"citation_supported=`{citation_supported}`, "
            f"hallucination_suspected=`{hallucination_suspected}`, "
            f"reason={_format_probe_value(grounded.reason)}"
        )
    if report.failure is not None:
        lines.extend(
            [
                "",
                "## Failure",
                "",
                f"- Stage: `{report.failure.stage}`",
                f"- Type: `{report.failure.error_type}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Activation decision",
            "",
            "- Evaluation eligible: "
            f"`{str(report.readiness.evaluation_eligible).lower() if report.readiness is not None else 'not evaluated'}`",
            "- Activation allowed: `false`",
            "- This tool never activates, shadows, rolls back, or changes an active generation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_final_report(context: BuildContext, report: LocalBuildReport) -> None:
    _write_model(context.root, context.report_directory / "build_report.json", report)
    markdown_path = context.report_directory / "build_report.md"
    atomic_write_project_bytes(
        context.root,
        Path(_relative(context.root, markdown_path)),
        _render_build_markdown(context, report).encode("utf-8"),
        overwrite=True,
    )


def _plan_report(inputs: BuildInputs) -> LocalBuildReport:
    """Return a no-write plan; it intentionally does not load or build resources."""

    root = inputs.project_root.resolve(strict=False)
    report_directory = root / "reports" / "rag_build" / inputs.run_id
    return LocalBuildReport(
        schema_version="rag_local_build_report_v1",
        run_id=inputs.run_id,
        mode="plan",
        status="planned",
        requested_code_revision=inputs.code_revision,
        # A plan intentionally performs no preflight side effect or Git call.
        head_code_revision=None,
        revision_matches_head=None,
        runtime_config_path=str(inputs.index_config),
        gold_dataset_path=str(inputs.gold_dataset),
        catalog=None,
        readiness=None,
        secrets=None,
        flat_baseline=None,
        generation=None,
        smoke_retrieval_path=str(report_directory / "smoke_retrieval.json"),
        grounded_smoke_path=str(report_directory / "llm_grounded_smoke.json"),
        stages=(),
        failure=None,
        experimental_only=True,
        activation_prohibited=True,
    )


def _failure_report(
    *,
    context: BuildContext,
    stages: Sequence[StageRecord],
    failed_stage: BuildStage,
    exc: BaseException,
    catalog: CatalogSummary | None,
    readiness: ReadinessSummary | None,
    flat: FlatBaselineSummary | None,
    generation: GenerationSummary | None,
) -> LocalBuildReport:
    retrieval_path, grounded_path = _placeholder_smoke_artifacts(
        context, reason=f"{failed_stage}_failed"
    )
    return LocalBuildReport(
        schema_version="rag_local_build_report_v1",
        run_id=context.inputs.run_id,
        mode=context.inputs.mode,
        status="failed",
        requested_code_revision=context.inputs.code_revision,
        head_code_revision=context.head_revision,
        revision_matches_head=(context.inputs.code_revision == context.head_revision),
        runtime_config_path=_relative(context.root, context.index_config_path),
        gold_dataset_path=_relative(context.root, context.gold_dataset_path),
        catalog=catalog,
        readiness=readiness,
        secrets=_secret_presence(
            context.config, llm_api_key_env=context.inputs.llm_api_key_env
        ),
        flat_baseline=flat,
        generation=generation,
        smoke_retrieval_path=_relative(context.root, retrieval_path),
        grounded_smoke_path=_relative(context.root, grounded_path),
        stages=tuple(stages),
        failure=FailureSummary(stage=failed_stage, error_type=_safe_failure_type(exc)),
        experimental_only=True,
        activation_prohibited=True,
    )


def _append_not_run_stages(stages: list[StageRecord], *, after: BuildStage) -> None:
    ordered: tuple[BuildStage, ...] = (
        "preflight",
        "catalog",
        "source_groups_and_readiness",
        "provider_probe",
        "chunk_dry_run",
        "flat_baseline",
        "parent_child_generation",
        "artifact_validation",
        "smoke_retrieval",
        "grounded_llm_smoke",
    )
    for stage in ordered[ordered.index(after) + 1 :]:
        stages.append(
            StageRecord(
                stage=stage,
                status="not_run",
                duration_ms=0.0,
                failure_type=None,
            )
        )


def _run_chunk_dry_run(context: BuildContext) -> None:
    """Run the real page-aware loader and splitter through the audit helper."""

    # This module is intentionally owned by the chunk-audit implementation.
    # It is imported only for an execution mode so a no-write plan has no
    # dependency on optional build tooling.
    from src.rag.parent_child.build_audit import (
        collect_chunk_stats,
        render_chunk_stats_markdown,
    )

    stats = collect_chunk_stats(
        project_root=context.root,
        config_path=context.index_config_path,
        generation_id=context.inputs.generation_id,
        source_groups_path=context.source_groups_path,
        progress=lambda message: print(message, flush=True),
    )
    if not isinstance(stats, BaseModel):
        raise LocalBuildError("ChunkStatsContractInvalid")
    _write_model(context.root, context.report_directory / "chunk_stats.json", stats)
    markdown = render_chunk_stats_markdown(stats)
    if not isinstance(markdown, str) or not markdown.strip():
        raise LocalBuildError("ChunkStatsMarkdownContractInvalid")
    atomic_write_project_bytes(
        context.root,
        Path(_relative(context.root, context.report_directory / "chunk_stats.md")),
        markdown.encode("utf-8"),
        overwrite=True,
    )


def _llm_probe_config(context: BuildContext) -> LlmProbeConfig | None:
    """Build a chat config only from every caller-supplied explicit coordinate.

    ``None`` is intentionally passed to the probe when any required explicit
    coordinate is absent.  The probe then writes its own strict, redacted
    configuration failure artifact; this is not an environment fallback.
    """

    from src.rag.parent_child.provider_probe import LlmProbeConfig

    inputs = context.inputs
    values = (
        inputs.llm_provider,
        inputs.llm_protocol,
        inputs.llm_model,
        inputs.llm_base_url,
        inputs.llm_endpoint_path,
        inputs.llm_api_key_env,
        inputs.llm_timeout_seconds,
    )
    if any(value is None for value in values):
        return None
    assert inputs.llm_provider is not None
    assert inputs.llm_protocol is not None
    assert inputs.llm_model is not None
    assert inputs.llm_base_url is not None
    assert inputs.llm_endpoint_path is not None
    assert inputs.llm_api_key_env is not None
    assert inputs.llm_timeout_seconds is not None
    return LlmProbeConfig(
        provider=inputs.llm_provider,
        protocol=inputs.llm_protocol,
        model=inputs.llm_model,
        base_url=inputs.llm_base_url,
        endpoint_path=inputs.llm_endpoint_path,
        api_key_env=inputs.llm_api_key_env,
        timeout_seconds=inputs.llm_timeout_seconds,
    )


def _run_provider_probe(context: BuildContext) -> object:
    """Run the real provider probes and require every requested probe to pass."""

    from src.rag.parent_child.provider_probe import run_provider_probe

    report = run_provider_probe(
        project_root=context.root,
        index_config_path=context.index_config_path,
        run_id=context.inputs.run_id,
        output_directory=context.report_directory,
        probe_llm_enabled=True,
        llm_config=_llm_probe_config(context),
    )
    if not bool(report.success):
        raise ProviderProbeFailed("ProviderProbeFailed")
    return report


def _validate_finite_vector(vector: object, *, expected_dimension: int) -> None:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise LocalBuildError("PersistedVectorShapeInvalid")
    if len(vector) != expected_dimension:
        raise LocalBuildError("PersistedVectorDimensionInvalid")
    for coordinate in vector:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise LocalBuildError("PersistedVectorCoordinateInvalid")
        if not math.isfinite(float(coordinate)):
            raise LocalBuildError("PersistedVectorCoordinateNotFinite")


def _sample_ids(ids: Sequence[str], *, limit: int = 10) -> tuple[str, ...]:
    if not ids:
        raise LocalBuildError("ChromaCollectionEmpty")
    count = min(limit, len(ids))
    return tuple(random.SystemRandom().sample(list(ids), count))


def _validate_flat_baseline(
    *, context: BuildContext, manifest: object, persist_directory: Path
) -> FlatBaselineSummary:
    """Open the persisted flat collection and prove bounded sample integrity."""

    import chromadb
    from chromadb.config import Settings

    from src.rag.parent_child.flat_baseline import (
        FlatBaselineChunkMetadata,
        FlatBaselineDocument,
        FlatBaselineManifest,
        read_flat_collection_ids,
    )
    from src.rag.parent_child.provider_clients import StrictEmbeddingClient

    if not isinstance(manifest, FlatBaselineManifest):
        raise LocalBuildError("FlatManifestContractInvalid")
    if persist_directory.is_symlink() or not persist_directory.is_dir():
        raise LocalBuildError("FlatChromaDirectoryInvalid")
    expected_metadata = {
        "schema_version": "flat_baseline_chroma_v2",
        "expected_dimension": manifest.embedding.dimension,
        "hnsw:space": manifest.embedding.distance_metric,
    }
    with chromadb.PersistentClient(
        path=str(persist_directory), settings=Settings(anonymized_telemetry=False)
    ) as client:
        collection = client.get_collection(
            manifest.collection_name, embedding_function=None
        )
        if collection.count() != manifest.chunk_count:
            raise LocalBuildError("FlatChromaCountMismatch")
        if collection.metadata != expected_metadata:
            raise LocalBuildError("FlatChromaMetadataMismatch")
        ids = read_flat_collection_ids(
            collection=collection,
            expected_count=manifest.chunk_count,
            page_size=context.config.embedding.batch_size,
        )
        sample_ids = _sample_ids(ids)
        sample = collection.get(
            ids=list(sample_ids), include=["documents", "metadatas", "embeddings"]
        )
        documents = sample.get("documents")
        metadatas = sample.get("metadatas")
        embeddings = sample.get("embeddings")
        sampled_ids = sample.get("ids")
        if not (
            isinstance(sampled_ids, list)
            and isinstance(documents, list)
            and isinstance(metadatas, list)
            and isinstance(embeddings, list)
            and len(sampled_ids)
            == len(documents)
            == len(metadatas)
            == len(embeddings)
            == len(sample_ids)
        ):
            raise LocalBuildError("FlatChromaSampleShapeInvalid")
        for identifier, document, metadata, vector in zip(
            sampled_ids, documents, metadatas, embeddings, strict=True
        ):
            if not isinstance(identifier, str) or not isinstance(document, str):
                raise LocalBuildError("FlatChromaSampleTypeInvalid")
            decoded = FlatBaselineDocument(
                schema_version="flat_baseline_document_v1",
                content=document,
                metadata=FlatBaselineChunkMetadata.from_chroma_metadata(metadata),
            )
            if decoded.metadata.chunk_id != identifier:
                raise LocalBuildError("FlatChromaSampleIdMismatch")
            _validate_finite_vector(
                vector, expected_dimension=manifest.embedding.dimension
            )
        embedding = StrictEmbeddingClient.production(config=context.config.embedding)
        try:
            query_vector = embedding.embed_query("RAG local persistence smoke query")
        finally:
            embedding.close()
        query = collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            include=["metadatas", "distances"],
        )
        query_ids = query.get("ids")
        if not (
            isinstance(query_ids, list)
            and len(query_ids) == 1
            and isinstance(query_ids[0], list)
            and query_ids[0]
        ):
            raise LocalBuildError("FlatChromaSimilarityQueryEmpty")
    return FlatBaselineSummary(
        build_id=context.inputs.build_id,
        chroma_path=_relative(context.root, persist_directory),
        manifest_path=_relative(
            context.root,
            Path("artifacts") / "rag" / context.inputs.build_id / "manifest.json",
        ),
        collection_name=manifest.collection_name,
        chunk_count=manifest.chunk_count,
        sampled_chunk_count=len(sample_ids),
        similarity_query_succeeded=True,
    )


def _build_flat_baseline(context: BuildContext) -> FlatBaselineSummary:
    artifact_root = Path("artifacts") / "rag" / context.inputs.build_id
    persist_directory = context.root / artifact_root / "chroma"
    manifest_output = context.root / artifact_root / "manifest.json"
    manifest = build_flat_baseline(
        project_root=context.root,
        index_config_path=context.index_config_path,
        persist_directory=persist_directory,
        manifest_output=manifest_output,
        collection_name=_collection_name(context.inputs.build_id),
        flat_build_id=context.inputs.build_id,
    )
    return _validate_flat_baseline(
        context=context,
        manifest=manifest,
        persist_directory=persist_directory,
    )


def _build_parent_child_generation(context: BuildContext) -> str:
    registry_path = context.config.storage.resolved_registry_path()
    registry_mode = "existing" if registry_path.is_file() else "create"
    return run_build(
        project_root=context.root,
        index_config_path=context.index_config_path,
        generation_id=context.inputs.generation_id,
        code_revision=context.inputs.code_revision,
        registry_mode=registry_mode,
    )


def _open_registry_record(context: BuildContext) -> object:
    """Read only this build's registry row; never resolve an active artifact."""

    from src.rag.parent_child.registry import GenerationRegistry

    with GenerationRegistry.open(
        context.config.storage.resolved_registry_path(),
        index_root=context.config.storage.index_root,
        expected_schema_version=context.config.storage.registry_schema_version,
        marker_schema_version=context.config.storage.owner_marker_schema_version,
        busy_timeout_seconds=context.config.storage.registry_busy_timeout_seconds,
    ) as registry:
        return registry.get_generation(context.inputs.generation_id)


def _validate_generation(
    *, context: BuildContext, manifest_sha256: str
) -> GenerationSummary:
    """Validate only the newly built READY generation and its sealed artifacts."""

    import chromadb
    from chromadb.config import Settings

    from src.rag.parent_child.bm25_artifact import (
        compute_tokenizer_fingerprint,
        digest_identifier_set,
        read_subject_bm25_artifact,
    )
    from src.rag.parent_child.builder import compute_embedding_fingerprint
    from src.rag.parent_child.generation import validate_sealed_generation
    from src.rag.parent_child.models import ChildMetadata
    from src.rag.parent_child.parent_store import ParentStore
    from src.rag.parent_child.registry import GenerationRegistry

    registry_path = context.config.storage.resolved_registry_path()
    with GenerationRegistry.open(
        registry_path,
        index_root=context.config.storage.index_root,
        expected_schema_version=context.config.storage.registry_schema_version,
        marker_schema_version=context.config.storage.owner_marker_schema_version,
        busy_timeout_seconds=context.config.storage.registry_busy_timeout_seconds,
    ) as registry:
        record = registry.get_generation(context.inputs.generation_id)
        deployment = registry.deployment()
    if record.state != "READY" or record.manifest_sha256 != manifest_sha256:
        raise LocalBuildError("GenerationRegistryReadyContractInvalid")
    # Reading the deployment pointer is not a retrieval of an active generation;
    # it proves this tool did not install its candidate into any pointer slot.
    if context.inputs.generation_id in {
        deployment.primary_generation_id,
        deployment.previous_generation_id,
        deployment.shadow_generation_id,
    }:
        raise LocalBuildError("CandidateUnexpectedlyReferencedByDeployment")
    manifest = validate_sealed_generation(
        context.config.storage.index_root,
        context.inputs.generation_id,
        expected_manifest_sha256=manifest_sha256,
        expected_marker_schema_version=context.config.storage.owner_marker_schema_version,
    )
    if manifest.code_revision != context.inputs.code_revision:
        raise LocalBuildError("GenerationCodeRevisionMismatch")
    if manifest.embedding.fingerprint != compute_embedding_fingerprint(context.config):
        raise LocalBuildError("GenerationEmbeddingFingerprintMismatch")
    if manifest.embedding.dimension != context.config.embedding.expected_dimension:
        raise LocalBuildError("GenerationEmbeddingDimensionMismatch")
    generation_root = (
        context.config.storage.index_root / context.inputs.generation_id
    ).resolve(strict=True)
    chroma_path = generation_root / "chroma_children"
    if chroma_path.is_symlink() or not chroma_path.is_dir():
        raise LocalBuildError("GenerationChromaDirectoryInvalid")
    expected_metadata = {
        "schema_version": "chroma_children_v1",
        "generation_id": context.inputs.generation_id,
        "expected_dimension": context.config.embedding.expected_dimension,
        "hnsw:space": context.config.embedding.distance_metric,
    }
    children_by_subject: dict[str, set[str]] = {}
    parent_ids: set[str] = set()
    with chromadb.PersistentClient(
        path=str(chroma_path), settings=Settings(anonymized_telemetry=False)
    ) as client:
        collection = client.get_collection(
            manifest.collection_name, embedding_function=None
        )
        if collection.count() != manifest.counts.child_count:
            raise LocalBuildError("GenerationChromaCountMismatch")
        if collection.metadata != expected_metadata:
            raise LocalBuildError("GenerationChromaMetadataMismatch")
        payload = collection.get(include=["metadatas"])
        identifiers = payload.get("ids")
        metadatas = payload.get("metadatas")
        if not (
            isinstance(identifiers, list)
            and isinstance(metadatas, list)
            and len(identifiers) == len(metadatas) == manifest.counts.child_count
        ):
            raise LocalBuildError("GenerationChromaMetadataShapeInvalid")
        ids = tuple(str(item) for item in identifiers)
        if digest_identifier_set(ids) != manifest.child_id_set_sha256:
            raise LocalBuildError("GenerationChromaIdDigestMismatch")
        for identifier, raw_metadata in zip(ids, metadatas, strict=True):
            metadata = ChildMetadata.from_chroma_metadata(raw_metadata)
            if metadata.child_id != identifier:
                raise LocalBuildError("GenerationChildIdMetadataMismatch")
            if metadata.generation_id != context.inputs.generation_id:
                raise LocalBuildError("GenerationChildIdMismatch")
            expected_policy = context.config.subject_policy_map.get(metadata.subject)
            if expected_policy is None or metadata.policy_id != expected_policy:
                raise LocalBuildError("GenerationChildPolicyMismatch")
            children_by_subject.setdefault(metadata.subject, set()).add(identifier)
            parent_ids.add(metadata.parent_id)
        sample_ids = _sample_ids(ids)
        sample = collection.get(ids=list(sample_ids), include=["embeddings"])
        sample_vectors = sample.get("embeddings")
        if not isinstance(sample_vectors, list) or len(sample_vectors) != len(
            sample_ids
        ):
            raise LocalBuildError("GenerationChromaVectorSampleInvalid")
        for vector in sample_vectors:
            _validate_finite_vector(
                vector, expected_dimension=context.config.embedding.expected_dimension
            )
    with ParentStore.open_readonly(
        generation_root,
        "parents.sqlite",
        expected_schema_version=context.config.storage.parent_store_schema_version,
        expected_generation_id=context.inputs.generation_id,
        busy_timeout_seconds=context.config.storage.parent_store_busy_timeout_seconds,
    ) as parent_store:
        parent_store.verify_integrity()
        parents = parent_store.get_many(tuple(sorted(parent_ids)))
    if len(parents) != len(parent_ids):
        raise LocalBuildError("GenerationParentHydrationMismatch")
    tokenizer_fingerprint = compute_tokenizer_fingerprint(
        tokenizer_name=context.config.bm25.tokenizer,
        tokenizer_version=context.config.bm25.tokenizer_version,
        dictionary_sha256=context.config.bm25.dictionary_hash,
    )
    for subject_id, policy_id in context.config.subject_policy_map.items():
        _ = policy_id
        _bm25_manifest, rows = read_subject_bm25_artifact(
            generation_root,
            f"bm25/{subject_id}.manifest.json",
            expected_manifest_schema_version="bm25_manifest_v1",
            expected_generation_id=context.inputs.generation_id,
            expected_subject=subject_id,
            expected_tokenizer_fingerprint=tokenizer_fingerprint,
        )
        bm25_ids = {row.child_id for row in rows}
        if bm25_ids != children_by_subject.get(subject_id, set()):
            raise LocalBuildError("GenerationBm25ChildSetMismatch")
    if set(children_by_subject) != set(context.config.subject_policy_map):
        raise LocalBuildError("GenerationSubjectInventoryMismatch")
    return GenerationSummary(
        generation_id=context.inputs.generation_id,
        generation_path=_relative(context.root, generation_root),
        manifest_sha256=manifest_sha256,
        registry_state="READY",
        active=False,
        orphan_child_count=0,
        hydration_failure_count=0,
        child_count=manifest.counts.child_count,
        parent_count=manifest.counts.parent_count,
    )


def _flat_smoke_channel(result: object) -> SmokeChannelResult:
    from src.rag.parent_child.flat_baseline import FlatBaselineRetrievalResult

    if not isinstance(result, FlatBaselineRetrievalResult):
        raise LocalBuildError("FlatSmokeResultContractInvalid")
    hits = tuple(
        SmokeHit(
            rank=hit.rank,
            source_relpath=hit.document.metadata.source_relpath,
            page_start=hit.document.metadata.page_start,
            page_end=hit.document.metadata.page_end,
            section_path=hit.document.metadata.section_path,
            score=None,
            reranker_score=hit.rerank_score,
            parent_hydrated=False,
        )
        for hit in result.hits
    )
    return SmokeChannelResult(
        status="ok" if hits else "empty",
        hits=hits,
        latency_ms=result.total_ms,
        context_token_estimate=sum(len(hit.document.content) for hit in result.hits)
        // 4,
        parent_hydration_success_count=0,
    )


def _candidate_smoke_channel(result: object) -> SmokeChannelResult:
    from src.rag.parent_child.retrieval import HybridRetrievalResult

    if not isinstance(result, HybridRetrievalResult):
        raise LocalBuildError("CandidateSmokeResultContractInvalid")
    hydrated_parent_ids = {item.parent.parent_id for item in result.hydrated_parents}
    hits = tuple(
        SmokeHit(
            rank=hit.final_rank,
            source_relpath=hit.document.metadata.source_relpath,
            page_start=hit.document.metadata.page_start,
            page_end=hit.document.metadata.page_end,
            section_path=hit.document.metadata.section_path,
            score=hit.rrf_score,
            reranker_score=hit.rerank_score,
            parent_hydrated=(hit.document.metadata.parent_id in hydrated_parent_ids),
        )
        for hit in result.ranked_children
    )
    context_chars = sum(
        len(window.content)
        for parent in result.hydrated_parents
        for window in parent.windows
    )
    return SmokeChannelResult(
        status=result.status,
        hits=hits,
        latency_ms=result.timings.total_ms,
        context_token_estimate=context_chars // 4,
        parent_hydration_success_count=len(result.hydrated_parents),
    )


def _load_flat_manifest(context: BuildContext) -> object:
    from src.rag.parent_child.flat_baseline import FlatBaselineManifest

    path = (
        context.root / "artifacts" / "rag" / context.inputs.build_id / "manifest.json"
    )
    try:
        return FlatBaselineManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise LocalBuildError("FlatManifestReloadFailed") from exc


def _run_smoke_retrieval(
    *,
    context: BuildContext,
    catalog_snapshot: SubjectCatalogSnapshot,
) -> tuple[SmokeRetrievalArtifact, dict[str, object], tuple[SmokeQuery, ...]]:
    """Run real Flat and candidate retrieval for GoldDataset-derived queries."""

    from src.rag.parent_child.flat_baseline import FlatBaselineRuntime
    from src.rag.parent_child.provider_clients import (
        StrictEmbeddingClient,
        StrictRerankerClient,
    )
    from src.rag.parent_child.retrieval import HybridRetrievalRequest
    from src.rag.parent_child.runtime_loader import load_generation_runtime
    from src.rag.parent_child.tokenizer import ConfiguredJiebaTokenizer

    manifest = _load_flat_manifest(context)
    record = _open_registry_record(context)
    selected = _select_gold_smoke_queries(context=context, catalog=catalog_snapshot)
    embedding = StrictEmbeddingClient.production(config=context.config.embedding)
    reranker = StrictRerankerClient.production(config=context.config.reranker)
    flat_runtime: FlatBaselineRuntime | None = None
    candidate_runtime: LoadedGenerationRuntime | None = None
    results: dict[str, object] = {}
    try:
        tokenizer = ConfiguredJiebaTokenizer(config=context.config.bm25)
        flat_runtime = FlatBaselineRuntime(
            persist_directory=(
                context.root / "artifacts" / "rag" / context.inputs.build_id / "chroma"
            ),
            manifest=manifest,
            query_embedding_provider=embedding,
            reranker=reranker,
            tokenizer=tokenizer,
            read_page_size=context.config.embedding.batch_size,
        )
        candidate_runtime = load_generation_runtime(
            config=context.config,
            registry_record=record,
            query_embedding_provider=embedding,
            reranker=reranker,
            bm25_tokenizer=tokenizer,
        )
        retriever = candidate_runtime.retriever()
        records: list[SmokeRetrievalRecord] = []
        for query in selected:
            flat_result = flat_runtime.retrieve(
                query=query.query,
                subject=query.subject,
                vector_top_k=context.config.retrieval.vector_top_k,
                bm25_top_k=context.config.retrieval.bm25_top_k,
                reranker_top_n=context.config.retrieval.reranker_top_n,
            )
            candidate_result = retriever.retrieve(
                HybridRetrievalRequest(
                    schema_version="hybrid_retrieval_request_v1",
                    request_id=query.query_id,
                    query=query.query,
                    subject=query.subject,
                    generation_id=context.inputs.generation_id,
                )
            )
            results[query.query_id] = candidate_result
            records.append(
                SmokeRetrievalRecord(
                    query_id=query.query_id,
                    subject=query.subject,
                    flat_baseline=_flat_smoke_channel(flat_result),
                    parent_child_candidate=_candidate_smoke_channel(candidate_result),
                )
            )
    finally:
        if flat_runtime is not None:
            flat_runtime.close()
        if candidate_runtime is not None:
            candidate_runtime.close()
        reranker.close()
        embedding.close()
    artifact = SmokeRetrievalArtifact(
        schema_version="rag_smoke_retrieval_v1",
        status="completed",
        reason=None,
        generation_id=context.inputs.generation_id,
        records=tuple(records),
    )
    _write_model(
        context.root, context.report_directory / "smoke_retrieval.json", artifact
    )
    return artifact, results, selected


def _context_for_grounded_smoke(
    result: object,
) -> tuple[tuple[GroundedCitation, ...], str]:
    from src.rag.parent_child.retrieval import HybridRetrievalResult

    if not isinstance(result, HybridRetrievalResult):
        raise LocalBuildError("GroundedSmokeCandidateContractInvalid")
    citations: list[GroundedCitation] = []
    blocks: list[str] = []
    for item in result.hydrated_parents:
        citation = GroundedCitation(
            source_relpath=item.parent.source_relpath,
            page_start=item.parent.page_start,
            page_end=item.parent.page_end,
        )
        if citation not in citations:
            citations.append(citation)
        blocks.append(
            "[source="
            + citation.source_relpath
            + "; pages="
            + str(citation.page_start)
            + "-"
            + str(citation.page_end)
            + "]\n"
            + "\n".join(window.content for window in item.windows)
        )
    return tuple(citations), "\n\n".join(blocks)


def _validate_grounded_citations(
    *, payload: _GroundedAnswerPayload, available: tuple[GroundedCitation, ...]
) -> tuple[GroundedCitation, ...]:
    citations = tuple(payload.citations)
    if len(citations) != len(set(citations)):
        raise LocalBuildError("GroundedAnswerDuplicateCitations")
    available_set = set(available)
    if any(citation not in available_set for citation in citations):
        raise LocalBuildError("GroundedAnswerCitationOutsideContext")
    if payload.evidence_insufficient and citations:
        raise LocalBuildError("GroundedAnswerInsufficientEvidenceHasCitations")
    if not payload.evidence_insufficient and not citations:
        raise LocalBuildError("GroundedAnswerMissingCitations")
    return citations


def _private_smoke_path(context: BuildContext) -> Path | None:
    requested = context.inputs.private_smoke_output
    if requested is None:
        return None
    output = resolve_project_path(context.root, requested, must_exist=False)
    reports_root = resolve_project_path(context.root, Path("reports"), must_exist=False)
    if not output.is_relative_to(reports_root):
        raise LocalBuildError("PrivateSmokeOutputMustBeUnderReports")
    relative = _relative(context.root, output)
    try:
        ignored = (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(context.root),
                    "check-ignore",
                    "--no-index",
                    "-q",
                    relative,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalBuildError("PrivateSmokeIgnoreCheckFailed") from exc
    if not ignored:
        raise LocalBuildError("PrivateSmokeOutputNotIgnored")
    return output


def _run_grounded_smoke(
    *,
    context: BuildContext,
    candidate_results: dict[str, object],
    selected_queries: tuple[SmokeQuery, ...],
) -> GroundedSmokeArtifact:
    """Use the real, explicitly configured chat client over candidate context."""

    from src.rag.parent_child.provider_probe import (
        ChatRequestMessage,
        StrictChatCompletionClient,
    )

    llm_config = _llm_probe_config(context)
    if llm_config is None:
        raise LocalBuildError("ExplicitLlmConfigurationMissing")
    private_path = _private_smoke_path(context)
    by_subject: dict[str, SmokeQuery] = {}
    for query in selected_queries:
        by_subject.setdefault(query.subject, query)
    model_fingerprint = hashlib.sha256(
        (
            llm_config.provider
            + "\x00"
            + llm_config.model
            + "\x00"
            + llm_config.base_url
            + llm_config.endpoint_path
        ).encode("utf-8")
    ).hexdigest()
    client = StrictChatCompletionClient.production(config=llm_config)
    records: list[GroundedSmokeRecord] = []
    private_records: list[dict[str, object]] = []
    try:
        for subject in sorted(by_subject):
            query = by_subject[subject]
            candidate = candidate_results.get(query.query_id)
            if candidate is None:
                raise LocalBuildError("GroundedSmokeRetrievalResultMissing")
            coordinates, context_text = _context_for_grounded_smoke(candidate)
            prompt = (
                "Return only strict JSON with exactly these keys: answer, citations, "
                "evidence_insufficient. citations must be a JSON array of objects with "
                "source_relpath, page_start, page_end. Answer only from Context. If the "
                "Context is insufficient, set evidence_insufficient=true and do not use "
                "memory.\n\nQuestion:\n"
                + query.query
                + "\n\nContext:\n"
                + (context_text if context_text else "[no retrieved evidence]")
            )
            started = perf_counter_ns()
            completion = client.complete(
                messages=(
                    ChatRequestMessage(
                        role="system",
                        content="You are a grounded RAG answerer. Never use unstated knowledge.",
                    ),
                    ChatRequestMessage(role="user", content=prompt),
                )
            )
            answer_payload = _GroundedAnswerPayload.model_validate_json(
                completion.content
            )
            citations = _validate_grounded_citations(
                payload=answer_payload, available=coordinates
            )
            review_prompt = (
                "Return only strict JSON with exactly boolean keys answer_supported, "
                "citation_supported, hallucination_suspected. Judge whether this answer "
                "is supported only by the listed context coordinates.\n\nCoordinates:\n"
                + json.dumps(
                    [item.model_dump(mode="json") for item in coordinates],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n\nContext:\n"
                + (context_text if context_text else "[no retrieved evidence]")
                + "\n\nAnswer JSON:\n"
                + completion.content
            )
            review_completion = client.complete(
                messages=(
                    ChatRequestMessage(
                        role="system",
                        content="You are a strict evidence reviewer.",
                    ),
                    ChatRequestMessage(role="user", content=review_prompt),
                )
            )
            review_payload = _GroundedReviewPayload.model_validate_json(
                review_completion.content
            )
            context_tokens = len(context_text) // 4
            records.append(
                GroundedSmokeRecord(
                    query_id=query.query_id,
                    subject=query.subject,
                    model_fingerprint=model_fingerprint,
                    context_sources=coordinates,
                    answer_sha256=hashlib.sha256(
                        answer_payload.answer.encode("utf-8")
                    ).hexdigest(),
                    answer_chars=len(answer_payload.answer),
                    citations=citations,
                    context_token_estimate=context_tokens,
                    latency_ms=(perf_counter_ns() - started) / 1_000_000.0,
                    evidence_insufficient=answer_payload.evidence_insufficient,
                    answer_supported=review_payload.answer_supported,
                    citation_supported=review_payload.citation_supported,
                    hallucination_suspected=review_payload.hallucination_suspected,
                )
            )
            private_records.append(
                {
                    "query_id": query.query_id,
                    "answer": answer_payload.answer,
                    "citations": [item.model_dump(mode="json") for item in citations],
                    "review": review_payload.model_dump(mode="json"),
                }
            )
    finally:
        client.close()
    if private_path is not None:
        atomic_write_project_bytes(
            context.root,
            Path(_relative(context.root, private_path)),
            json.dumps(
                {
                    "schema_version": "rag_private_grounded_smoke_v1",
                    "records": private_records,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
            overwrite=False,
        )
    artifact = GroundedSmokeArtifact(
        schema_version="rag_llm_grounded_smoke_v1",
        status="completed",
        reason=None,
        records=tuple(records),
        private_output_written=private_path is not None,
    )
    _write_model(
        context.root, context.report_directory / "llm_grounded_smoke.json", artifact
    )
    return artifact


def _run_offline_dry_run(context: BuildContext) -> tuple[LocalBuildReport, int]:
    stages: list[StageRecord] = []
    catalog: CatalogSummary | None = None
    readiness: ReadinessSummary | None = None
    started = perf_counter_ns()
    try:
        context.report_directory.mkdir(parents=True, exist_ok=False)
        preflight = _preflight_report(context)
        _write_model(
            context.root,
            context.report_directory / "preflight.json",
            preflight,
        )
        _require_build_dependencies(preflight.dependencies)
        stages.append(
            _new_stage(
                stage="preflight", status="completed", started_ns=started, exc=None
            )
        )

        started = perf_counter_ns()
        catalog, _snapshot = _catalog_summary(context)
        stages.append(
            _new_stage(
                stage="catalog", status="completed", started_ns=started, exc=None
            )
        )

        started = perf_counter_ns()
        readiness = _readiness_summary(context)
        stages.append(
            _new_stage(
                stage="source_groups_and_readiness",
                status="completed",
                started_ns=started,
                exc=None,
            )
        )

        started = perf_counter_ns()
        _run_chunk_dry_run(context)
        stages.append(
            _new_stage(
                stage="chunk_dry_run", status="completed", started_ns=started, exc=None
            )
        )
        retrieval_path, grounded_path = _placeholder_smoke_artifacts(
            context, reason="offline_dry_run"
        )
        report = LocalBuildReport(
            schema_version="rag_local_build_report_v1",
            run_id=context.inputs.run_id,
            mode="offline_dry_run",
            status="offline_complete",
            requested_code_revision=context.inputs.code_revision,
            head_code_revision=context.head_revision,
            revision_matches_head=(
                context.inputs.code_revision == context.head_revision
            ),
            runtime_config_path=_relative(context.root, context.index_config_path),
            gold_dataset_path=_relative(context.root, context.gold_dataset_path),
            catalog=catalog,
            readiness=readiness,
            secrets=_secret_presence(context.config),
            flat_baseline=None,
            generation=None,
            smoke_retrieval_path=_relative(context.root, retrieval_path),
            grounded_smoke_path=_relative(context.root, grounded_path),
            stages=tuple(stages),
            failure=None,
            experimental_only=True,
            activation_prohibited=True,
        )
        _write_final_report(context, report)
        return report, 0
    except Exception as exc:
        failed_stage: BuildStage = "chunk_dry_run"
        if not stages:
            failed_stage = "preflight"
        elif stages[-1].stage == "preflight":
            failed_stage = "catalog"
        elif stages[-1].stage == "catalog":
            failed_stage = "source_groups_and_readiness"
        stages.append(
            _new_stage(stage=failed_stage, status="failed", started_ns=started, exc=exc)
        )
        report = _failure_report(
            context=context,
            stages=stages,
            failed_stage=failed_stage,
            exc=exc,
            catalog=catalog,
            readiness=readiness,
            flat=None,
            generation=None,
        )
        _write_final_report(context, report)
        return report, 1


def run_local_build(inputs: BuildInputs) -> tuple[LocalBuildReport, int]:
    """Execute the selected safe local-build mode.

    The full provider-backed path is completed below as dedicated helpers so
    every external side effect remains stage-bound and is never replaced by a
    legacy or active-generation fallback.
    """

    if inputs.mode == "plan":
        return _plan_report(inputs), 0
    context = _prepare_context(inputs)
    if inputs.mode == "offline_dry_run":
        return _run_offline_dry_run(context)
    return _run_execute(context)


def _run_execute(context: BuildContext) -> tuple[LocalBuildReport, int]:
    """Run the provider-backed path without activation or fallback behavior."""

    stages: list[StageRecord] = []
    catalog: CatalogSummary | None = None
    catalog_snapshot: SubjectCatalogSnapshot | None = None
    readiness: ReadinessSummary | None = None
    flat: FlatBaselineSummary | None = None
    generation: GenerationSummary | None = None
    current_stage: BuildStage = "preflight"
    started = perf_counter_ns()
    try:
        context.report_directory.mkdir(parents=True, exist_ok=False)
        preflight = _preflight_report(context)
        _write_model(
            context.root,
            context.report_directory / "preflight.json",
            preflight,
        )
        _require_build_dependencies(preflight.dependencies)
        _validate_new_targets(context)
        stages.append(
            _new_stage(
                stage="preflight", status="completed", started_ns=started, exc=None
            )
        )
        _placeholder_smoke_artifacts(context, reason="not_started")

        current_stage = "catalog"
        started = perf_counter_ns()
        catalog, catalog_snapshot = _catalog_summary(context)
        stages.append(
            _new_stage(
                stage="catalog", status="completed", started_ns=started, exc=None
            )
        )

        current_stage = "source_groups_and_readiness"
        started = perf_counter_ns()
        readiness = _readiness_summary(context)
        stages.append(
            _new_stage(
                stage="source_groups_and_readiness",
                status="completed",
                started_ns=started,
                exc=None,
            )
        )

        current_stage = "provider_probe"
        started = perf_counter_ns()
        _run_provider_probe(context)
        stages.append(
            _new_stage(
                stage="provider_probe",
                status="completed",
                started_ns=started,
                exc=None,
            )
        )

        current_stage = "chunk_dry_run"
        started = perf_counter_ns()
        _run_chunk_dry_run(context)
        stages.append(
            _new_stage(
                stage="chunk_dry_run",
                status="completed",
                started_ns=started,
                exc=None,
            )
        )

        current_stage = "flat_baseline"
        started = perf_counter_ns()
        flat = _build_flat_baseline(context)
        stages.append(
            _new_stage(
                stage="flat_baseline",
                status="completed",
                started_ns=started,
                exc=None,
            )
        )

        current_stage = "parent_child_generation"
        started = perf_counter_ns()
        manifest_sha256 = _build_parent_child_generation(context)
        stages.append(
            _new_stage(
                stage="parent_child_generation",
                status="completed",
                started_ns=started,
                exc=None,
            )
        )

        current_stage = "artifact_validation"
        started = perf_counter_ns()
        generation = _validate_generation(
            context=context, manifest_sha256=manifest_sha256
        )
        stages.append(
            _new_stage(
                stage="artifact_validation",
                status="completed",
                started_ns=started,
                exc=None,
            )
        )

        if catalog_snapshot is None:
            raise LocalBuildError("CatalogSnapshotMissing")
        current_stage = "smoke_retrieval"
        started = perf_counter_ns()
        _smoke, candidate_results, selected_queries = _run_smoke_retrieval(
            context=context, catalog_snapshot=catalog_snapshot
        )
        stages.append(
            _new_stage(
                stage="smoke_retrieval",
                status="completed",
                started_ns=started,
                exc=None,
            )
        )

        current_stage = "grounded_llm_smoke"
        started = perf_counter_ns()
        _run_grounded_smoke(
            context=context,
            candidate_results=candidate_results,
            selected_queries=selected_queries,
        )
        stages.append(
            _new_stage(
                stage="grounded_llm_smoke",
                status="completed",
                started_ns=started,
                exc=None,
            )
        )
        report = LocalBuildReport(
            schema_version="rag_local_build_report_v1",
            run_id=context.inputs.run_id,
            mode="execute",
            status="success",
            requested_code_revision=context.inputs.code_revision,
            head_code_revision=context.head_revision,
            revision_matches_head=(
                context.inputs.code_revision == context.head_revision
            ),
            runtime_config_path=_relative(context.root, context.index_config_path),
            gold_dataset_path=_relative(context.root, context.gold_dataset_path),
            catalog=catalog,
            readiness=readiness,
            secrets=_secret_presence(
                context.config, llm_api_key_env=context.inputs.llm_api_key_env
            ),
            flat_baseline=flat,
            generation=generation,
            smoke_retrieval_path=_relative(
                context.root, context.report_directory / "smoke_retrieval.json"
            ),
            grounded_smoke_path=_relative(
                context.root, context.report_directory / "llm_grounded_smoke.json"
            ),
            stages=tuple(stages),
            failure=None,
            experimental_only=True,
            activation_prohibited=True,
        )
        _write_final_report(context, report)
        return report, 0
    except Exception as exc:
        stages.append(
            _new_stage(
                stage=current_stage,
                status="failed",
                started_ns=started,
                exc=exc,
            )
        )
        _append_not_run_stages(stages, after=current_stage)
        report = _failure_report(
            context=context,
            stages=stages,
            failed_stage=current_stage,
            exc=exc,
            catalog=catalog,
            readiness=readiness,
            flat=flat,
            generation=generation,
        )
        _write_final_report(context, report)
        return report, 1


def main(argv: list[str] | None = None) -> int:
    try:
        inputs = _build_inputs(_parser().parse_args(argv))
        report, exit_code = run_local_build(inputs)
    except Exception as exc:
        # There is no guaranteed contained report location before argument and
        # project-root validation.  Do not print exception text: it can include
        # a provider URL or secret-bearing environment value.
        print(f"local RAG build failed: {_safe_failure_type(exc)}", file=sys.stderr)
        return 2
    print(report.model_dump_json(indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
