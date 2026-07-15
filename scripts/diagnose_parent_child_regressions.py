"""Run a safe 10-20 query regression diagnosis on one exact READY generation.

The command never reads deployment pointers.  It uses the existing disposable
Chroma runtime snapshot and writes only identities, coordinates, ranks, scores,
stage outcomes, and latency.  Query text, chunk/parent content, provider bodies,
and secrets are intentionally absent from the output contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.rag_index_config import (  # noqa: E402
    RagIndexConfig,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child._storage_io import (  # noqa: E402
    canonical_json_bytes,
    model_json_bytes,
    sha256_bytes,
)
from src.rag.parent_child.evaluation import GoldDataset, GoldQuery  # noqa: E402
from src.rag.parent_child.project_paths import (  # noqa: E402
    ProjectPathError,
    atomic_write_project_bytes,
    require_project_directory,
    require_project_file,
    resolve_project_path,
    resolve_project_root,
)
from src.rag.parent_child.provider_clients import (  # noqa: E402
    StrictEmbeddingClient,
    StrictRerankerClient,
)
from src.rag.parent_child.registry import (  # noqa: E402
    GenerationRegistry,
    GenerationRegistryRecord,
)
from src.rag.parent_child.regression_diagnostics import (  # noqa: E402
    ParentChildRegressionReport,
    QueryRegressionDiagnostic,
    RegressionQuerySubset,
    build_regression_report,
    diagnose_gold_query,
)
from src.rag.parent_child.retrieval import (  # noqa: E402
    HybridRetrievalPolicy,
    HybridRetrievalRequest,
    ParentChildHybridRetriever,
)
from src.rag.parent_child.runtime_loader import (  # noqa: E402
    LoadedGenerationRuntime,
    load_generation_runtime,
)
from src.rag.parent_child.tokenizer import ConfiguredJiebaTokenizer  # noqa: E402


class RegressionDiagnosisCliError(RuntimeError):
    """A required exact-generation diagnostic input is invalid."""


class RegressionDiagnosisFailure(BaseModel):
    """Safe failure marker containing no query, content, provider body, or secret."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["regression_diagnosis_failure_v1"]
    failure_code: Literal["regression_diagnosis_failed"]
    failure_type: str = Field(min_length=1, max_length=128)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--gold-dataset", type=Path, required=True)
    parser.add_argument("--candidate-generation-id", required=True)
    parser.add_argument("--reranker-top-n", type=int, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--query-id", action="append")
    selection.add_argument("--query-subset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_config(*, root: Path, path: Path) -> RagIndexConfig:
    config_path = require_project_file(root, path)
    config = resolve_rag_index_config_paths(
        load_rag_index_config(config_path),
        project_root=root,
    )
    if config.catalog.symlink_policy != "reject":
        raise RegressionDiagnosisCliError(
            "regression diagnosis requires catalog symlink_policy=reject"
        )
    require_project_directory(root, config.catalog.data_root)
    index_root = require_project_directory(root, config.storage.index_root)
    registry_path = require_project_file(
        root,
        config.storage.resolved_registry_path(),
    )
    if not registry_path.is_relative_to(index_root):
        raise RegressionDiagnosisCliError(
            "registry path must remain within configured index_root"
        )
    return config


def _load_gold(path: Path) -> tuple[GoldDataset, str]:
    payload = path.read_bytes()
    try:
        dataset = GoldDataset.model_validate_json(payload)
    except Exception as exc:
        raise RegressionDiagnosisCliError("GoldDataset validation failed") from exc
    canonical = canonical_json_bytes(dataset.model_dump(mode="json"))
    if payload != canonical:
        raise RegressionDiagnosisCliError("GoldDataset must use canonical JSON")
    return dataset, sha256_bytes(payload)


def _load_subset(path: Path, *, gold_sha256: str) -> tuple[str, ...]:
    payload = path.read_bytes()
    try:
        subset = RegressionQuerySubset.model_validate_json(payload)
    except Exception as exc:
        raise RegressionDiagnosisCliError(
            "regression query subset validation failed"
        ) from exc
    if payload != model_json_bytes(subset):
        raise RegressionDiagnosisCliError(
            "regression query subset must use canonical JSON"
        )
    if subset.gold_dataset_sha256 != gold_sha256:
        raise RegressionDiagnosisCliError(
            "regression query subset GoldDataset digest mismatch"
        )
    return subset.query_ids


def _select_queries(
    *,
    root: Path,
    dataset: GoldDataset,
    gold_sha256: str,
    query_ids: list[str] | None,
    subset_path: Path | None,
) -> tuple[GoldQuery, ...]:
    if (query_ids is None) == (subset_path is None):
        raise RegressionDiagnosisCliError(
            "exactly one query selection mechanism is required"
        )
    if query_ids is not None:
        selected_ids = tuple(query_ids)
    else:
        if subset_path is None:
            raise RegressionDiagnosisCliError(
                "query subset path is required for subset selection"
            )
        selected_ids = _load_subset(
            require_project_file(root, subset_path),
            gold_sha256=gold_sha256,
        )
    if not 10 <= len(selected_ids) <= 20:
        raise RegressionDiagnosisCliError(
            "regression diagnosis requires between 10 and 20 query IDs"
        )
    if len(selected_ids) != len(set(selected_ids)):
        raise RegressionDiagnosisCliError("diagnostic query IDs must be unique")
    by_id = {query.query_id: query for query in dataset.queries}
    unknown = set(selected_ids) - set(by_id)
    if unknown:
        raise RegressionDiagnosisCliError(
            "diagnostic query selection contains unknown query IDs"
        )
    return tuple(by_id[query_id] for query_id in selected_ids)


def _select_ready_generation(
    *,
    registry: GenerationRegistry,
    generation_id: str,
) -> GenerationRegistryRecord:
    record = registry.get_generation(generation_id)
    if record.generation_id != generation_id:
        raise RegressionDiagnosisCliError(
            "registry returned a generation other than the explicit request"
        )
    if record.state != "READY" or record.manifest_sha256 is None:
        raise RegressionDiagnosisCliError(
            "regression diagnosis requires the explicit generation to be READY"
        )
    return record


def _runtime_policy(
    *,
    sealed_policy: HybridRetrievalPolicy,
    reranker_top_n: int,
) -> HybridRetrievalPolicy:
    payload = sealed_policy.model_dump(mode="python")
    payload["reranker_top_n"] = reranker_top_n
    try:
        return HybridRetrievalPolicy.model_validate(payload)
    except Exception as exc:
        raise RegressionDiagnosisCliError(
            "runtime-only retrieval policy validation failed"
        ) from exc


def _retriever(
    *,
    runtime: LoadedGenerationRuntime,
    policy: HybridRetrievalPolicy,
) -> ParentChildHybridRetriever:
    resources = runtime.resources
    return ParentChildHybridRetriever(
        policy=policy,
        vector_search=resources.vector,
        bm25_search=resources.bm25,
        reranker=resources.reranker,
        parent_hydrator=resources.parents,
    )


def _write_failure(*, root: Path, output: Path, error: Exception) -> None:
    failure_path = output.with_name(output.name + ".failure.json")
    if failure_path.exists():
        return
    try:
        atomic_write_project_bytes(
            root,
            failure_path,
            model_json_bytes(
                RegressionDiagnosisFailure(
                    schema_version="regression_diagnosis_failure_v1",
                    failure_code="regression_diagnosis_failed",
                    failure_type=type(error).__name__,
                )
            ),
            overwrite=False,
        )
    except (FileExistsError, OSError, ProjectPathError):
        return


def run_diagnosis(
    *,
    project_root: Path,
    index_config_path: Path,
    gold_dataset_path: Path,
    candidate_generation_id: str,
    reranker_top_n: int,
    query_ids: list[str] | None,
    query_subset_path: Path | None,
    output_path: Path,
) -> ParentChildRegressionReport:
    """Run one exact-generation diagnostic batch and atomically publish success."""

    root = resolve_project_root(project_root)
    config = _load_config(root=root, path=index_config_path)
    gold_path = require_project_file(root, gold_dataset_path)
    output = resolve_project_path(root, output_path, must_exist=False)
    if output.exists():
        raise FileExistsError(output)
    dataset, gold_sha256 = _load_gold(gold_path)
    selected_queries = _select_queries(
        root=root,
        dataset=dataset,
        gold_sha256=gold_sha256,
        query_ids=query_ids,
        subset_path=query_subset_path,
    )

    embedding = StrictEmbeddingClient.production(config=config.embedding)
    reranker = StrictRerankerClient.production(config=config.reranker)
    tokenizer = ConfiguredJiebaTokenizer(config=config.bm25)
    registry: GenerationRegistry | None = None
    runtime = None
    try:
        registry = GenerationRegistry.open(
            config.storage.resolved_registry_path(),
            index_root=config.storage.index_root,
            expected_schema_version=config.storage.registry_schema_version,
            marker_schema_version=config.storage.owner_marker_schema_version,
            busy_timeout_seconds=config.storage.registry_busy_timeout_seconds,
        )
        record = _select_ready_generation(
            registry=registry,
            generation_id=candidate_generation_id,
        )
        manifest_sha256 = record.manifest_sha256
        if manifest_sha256 is None:
            raise RegressionDiagnosisCliError(
                "READY generation is missing its sealed manifest digest"
            )
        runtime = load_generation_runtime(
            config=config,
            registry_record=record,
            query_embedding_provider=embedding,
            reranker=reranker,
            bm25_tokenizer=tokenizer,
        )
        policy = _runtime_policy(
            sealed_policy=runtime.retrieval_policy,
            reranker_top_n=reranker_top_n,
        )
        retriever = _retriever(runtime=runtime, policy=policy)
        diagnostics: list[QueryRegressionDiagnostic] = []
        for query in selected_queries:
            _result, trace = retriever.retrieve_with_diagnostics(
                HybridRetrievalRequest(
                    schema_version="hybrid_retrieval_request_v1",
                    request_id=query.query_id,
                    query=query.query,
                    subject=query.subject,
                    generation_id=candidate_generation_id,
                )
            )
            diagnostics.append(diagnose_gold_query(query=query, trace=trace))
        report = build_regression_report(
            dataset_id=dataset.dataset_id,
            gold_dataset_sha256=gold_sha256,
            generation_id=candidate_generation_id,
            generation_manifest_sha256=manifest_sha256,
            retrieval_policy=policy,
            diagnostics=tuple(diagnostics),
        )
        atomic_write_project_bytes(
            root,
            output,
            model_json_bytes(report),
            overwrite=False,
        )
        return report
    finally:
        if runtime is not None:
            runtime.close()
        if registry is not None:
            registry.close()
        reranker.close()
        embedding.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_diagnosis(
            project_root=args.project_root,
            index_config_path=args.index_config,
            gold_dataset_path=args.gold_dataset,
            candidate_generation_id=args.candidate_generation_id,
            reranker_top_n=args.reranker_top_n,
            query_ids=args.query_id,
            query_subset_path=args.query_subset,
            output_path=args.output,
        )
    except Exception as exc:
        try:
            root = resolve_project_root(args.project_root)
            output = resolve_project_path(root, args.output, must_exist=False)
            _write_failure(root=root, output=output, error=exc)
        except Exception as artifact_exc:
            print(
                "Regression failure artifact unavailable: "
                + type(artifact_exc).__name__,
                file=sys.stderr,
            )
        print(
            "Parent-child regression diagnosis failed: " + type(exc).__name__,
            file=sys.stderr,
        )
        return 1
    print(
        "Parent-child regression diagnosis complete: "
        f"generation={report.generation_id}, queries={len(report.query_ids)}, "
        f"retrieval={report.retrieval_fingerprint[:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
