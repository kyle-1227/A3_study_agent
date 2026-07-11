"""Validate the strict parent-child RAG environment without network access."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path
import sys
from typing import Literal

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from pydantic import Field  # noqa: E402

from src.config._rag_config import (  # noqa: E402
    StrictRagConfigModel,
    resolve_required_secret,
)
from src.config.rag_benchmark_config import (  # noqa: E402
    load_rag_benchmark_config,
)
from src.config.rag_index_config import (  # noqa: E402
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.config.rag_rollout_config import (  # noqa: E402
    load_rag_rollout_config,
)
from src.rag.parent_child._storage_io import (  # noqa: E402
    atomic_write_bytes,
    model_json_bytes,
)
from src.rag.subject_catalog import SubjectCatalog  # noqa: E402


_REQUIRED_MODULES: tuple[str, ...] = (
    "chromadb",
    "fitz",
    "httpx",
    "jieba",
    "rank_bm25",
    "yaml",
)


class DoctorSubjectEntry(StrictRagConfigModel):
    """Content-free inventory for one configured subject."""

    subject_id: str
    source_file_count: int = Field(gt=0)
    policy_id: str


class RagDoctorReport(StrictRagConfigModel):
    """Successful strict-environment validation result."""

    schema_version: Literal["rag_doctor_report_v1"]
    pipeline: Literal["parent-child"]
    required_modules: tuple[str, ...]
    subjects: tuple[DoctorSubjectEntry, ...]
    embedding_secret_present: Literal[True]
    reranker_secret_present: Literal[True]
    registry_exists: bool
    activation_enabled: bool
    shadow_enabled: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline", choices=("parent-child",), required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--rollout-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _contained_path(
    project_root: Path,
    value: Path,
    *,
    must_exist: bool,
) -> Path:
    candidate = value if value.is_absolute() else project_root / value
    if candidate.is_symlink():
        raise ValueError("doctor paths must not be symlinks")
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(project_root):
        raise ValueError("doctor paths must remain inside project_root")
    if must_exist and not resolved.is_file():
        raise ValueError("doctor input must be a regular non-symlink file")
    return resolved


def validate_subject_alignment(
    *,
    index_subjects: tuple[str, ...],
    benchmark_subjects: tuple[str, ...],
    rollout_subjects: tuple[str, ...],
) -> None:
    """Require all production control planes to name the same subject set."""

    if len(index_subjects) != len(set(index_subjects)):
        raise ValueError("index subjects must be unique")
    expected = set(index_subjects)
    if set(benchmark_subjects) != expected:
        raise ValueError("benchmark primary subjects differ from index subjects")
    if set(rollout_subjects) != expected:
        raise ValueError("rollout primary subjects differ from index subjects")


def run_doctor(
    *,
    project_root: Path,
    index_config_path: Path,
    benchmark_config_path: Path,
    rollout_config_path: Path,
    output_path: Path,
) -> RagDoctorReport:
    """Validate configuration, dependencies, secrets, and source discovery."""

    root = project_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project-root must be a directory")
    index_path = _contained_path(root, index_config_path, must_exist=True)
    benchmark_path = _contained_path(root, benchmark_config_path, must_exist=True)
    rollout_path = _contained_path(root, rollout_config_path, must_exist=True)
    output = _contained_path(root, output_path, must_exist=False)

    missing_modules = tuple(
        module_name
        for module_name in _REQUIRED_MODULES
        if find_spec(module_name) is None
    )
    if missing_modules:
        raise RuntimeError(
            "required RAG modules are missing: " + ", ".join(missing_modules)
        )

    index_config = resolve_rag_index_config_paths(
        load_rag_index_config(index_path),
        project_root=root,
    )
    benchmark_config = load_rag_benchmark_config(benchmark_path)
    rollout_config = load_rag_rollout_config(rollout_path)
    catalog = SubjectCatalog(
        config=index_config.catalog,
        subject_policy_map=index_config.subject_policy_map,
    ).discover()
    index_subjects = catalog.subject_ids()
    validate_subject_alignment(
        index_subjects=index_subjects,
        benchmark_subjects=benchmark_config.primary_subjects,
        rollout_subjects=rollout_config.primary_subjects,
    )

    # Resolve only to prove presence. Secret values never enter the report.
    resolve_required_secret(index_config.embedding.api_key_env)
    resolve_required_secret(index_config.reranker.api_key_env)
    report = RagDoctorReport(
        schema_version="rag_doctor_report_v1",
        pipeline="parent-child",
        required_modules=_REQUIRED_MODULES,
        subjects=tuple(
            DoctorSubjectEntry(
                subject_id=subject.subject_id,
                source_file_count=len(subject.sources),
                policy_id=subject.policy_id,
            )
            for subject in catalog.subjects
        ),
        embedding_secret_present=True,
        reranker_secret_present=True,
        registry_exists=index_config.storage.resolved_registry_path().is_file(),
        activation_enabled=rollout_config.activation_enabled,
        shadow_enabled=rollout_config.shadow_enabled,
    )
    atomic_write_bytes(
        root,
        output.relative_to(root).as_posix(),
        model_json_bytes(report),
        overwrite=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_doctor(
        project_root=args.project_root,
        index_config_path=args.index_config,
        benchmark_config_path=args.benchmark_config,
        rollout_config_path=args.rollout_config,
        output_path=args.output,
    )
    print(
        "RAG environment validated without network access: "
        f"subjects={len(report.subjects)}, registry_exists={report.registry_exists}, "
        f"activation_enabled={report.activation_enabled}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
