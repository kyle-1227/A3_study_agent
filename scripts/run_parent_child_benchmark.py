"""Run one explicit Flat Baseline versus Parent-Child candidate benchmark.

The command never consults deployment pointers.  It loads the requested READY
generation by ID, executes both strict retrieval chains on exactly one
GoldDataset, and writes only safe projections required by the formal validator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.rag_index_config import (  # noqa: E402
    RagIndexConfig,
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.context_engineering.tokenizer import estimate_text_tokens_mixed  # noqa: E402
from src.rag.parent_child._storage_io import (  # noqa: E402
    canonical_json_bytes,
    model_json_bytes,
    sha256_bytes,
)
from src.rag.parent_child.benchmarking import (  # noqa: E402
    BenchmarkRunBinding,
    PairedBenchmarkExecution,
    run_paired_benchmark,
)
from src.rag.parent_child.bm25_artifact import compute_tokenizer_fingerprint  # noqa: E402
from src.rag.parent_child.builder import compute_embedding_fingerprint  # noqa: E402
from src.rag.parent_child.evaluation import GoldDataset  # noqa: E402
from src.rag.parent_child.evaluation_gate import OperationalBenchmarkOutcome  # noqa: E402
from src.rag.parent_child.flat_baseline import (  # noqa: E402
    FlatBaselineManifest,
    FlatBaselineRuntime,
    compute_flat_retrieval_fingerprint,
    flat_manifest_bytes,
)
from src.rag.parent_child.manifests import EmbeddingManifestIdentity  # noqa: E402
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
from src.rag.parent_child.retrieval import (  # noqa: E402
    HybridRetrievalRequest,
    compute_retrieval_fingerprint,
)
from src.rag.parent_child.runtime_loader import load_generation_runtime  # noqa: E402
from src.rag.parent_child.tokenizer import ConfiguredJiebaTokenizer  # noqa: E402


class BenchmarkCliError(RuntimeError):
    """A required benchmark input or resource does not satisfy its contract."""


class BenchmarkFailureArtifact(BaseModel):
    """Safe non-success marker; it contains no query, body, secret, or URL."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["parent_child_benchmark_failure_v1"]
    failure_code: str = Field(min_length=1, max_length=128)
    failure_type: str = Field(min_length=1, max_length=128)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--gold-dataset", type=Path, required=True)
    parser.add_argument("--baseline-persist-dir", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-generation-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _embedding_identity(config: RagIndexConfig) -> EmbeddingManifestIdentity:
    embedding = config.embedding
    return EmbeddingManifestIdentity(
        provider=embedding.provider,
        model=embedding.model,
        base_url_identity=embedding.base_url.rstrip("/") + embedding.endpoint_path,
        input_types=(embedding.document_input_type, embedding.query_input_type),
        fingerprint=compute_embedding_fingerprint(config),
        dimension=embedding.expected_dimension,
        distance_metric=embedding.distance_metric,
    )


def _load_index_config(*, root: Path, config_path: Path) -> RagIndexConfig:
    config = resolve_rag_index_config_paths(
        load_rag_index_config(config_path), project_root=root
    )
    if config.catalog.symlink_policy != "reject":
        raise BenchmarkCliError(
            "benchmark tooling requires catalog symlink_policy=reject"
        )
    require_project_directory(root, config.catalog.data_root)
    index_root = resolve_project_path(root, config.storage.index_root, must_exist=True)
    registry_path = config.storage.resolved_registry_path()
    require_project_file(root, registry_path)
    if not registry_path.is_relative_to(index_root):
        raise BenchmarkCliError("registry path must remain within strict index_root")
    return config


def _load_canonical_gold(path: Path) -> tuple[GoldDataset, str]:
    payload = path.read_bytes()
    try:
        dataset = GoldDataset.model_validate_json(payload)
    except Exception as exc:
        raise BenchmarkCliError("GoldDataset contract validation failed") from exc
    canonical = canonical_json_bytes(dataset.model_dump(mode="json"))
    if payload != canonical:
        raise BenchmarkCliError("GoldDataset must use canonical JSON serialization")
    return dataset, sha256_bytes(payload)


def _load_flat_manifest(path: Path) -> tuple[FlatBaselineManifest, str]:
    payload = path.read_bytes()
    try:
        manifest = FlatBaselineManifest.model_validate_json(payload)
    except Exception as exc:
        raise BenchmarkCliError("flat baseline manifest validation failed") from exc
    if payload != flat_manifest_bytes(manifest):
        raise BenchmarkCliError("flat baseline manifest must use canonical JSON")
    return manifest, sha256_bytes(payload)


def _assert_flat_manifest_matches_config(
    *, config: RagIndexConfig, manifest: FlatBaselineManifest
) -> None:
    if manifest.embedding != _embedding_identity(config):
        raise BenchmarkCliError("flat baseline embedding identity differs from config")
    expected_tokenizer = compute_tokenizer_fingerprint(
        tokenizer_name=config.bm25.tokenizer,
        tokenizer_version=config.bm25.tokenizer_version,
        dictionary_sha256=config.bm25.dictionary_hash,
    )
    if manifest.bm25_tokenizer_fingerprint != expected_tokenizer:
        raise BenchmarkCliError("flat baseline BM25 identity differs from config")


def _select_requested_generation(
    *, registry: GenerationRegistry, candidate_generation_id: str
) -> GenerationRegistryRecord:
    """Read exactly the requested lifecycle record without consulting deployment.

    The benchmark intentionally has no active/primary-generation lookup.  The
    runtime loader performs the subsequent READY and sealed-manifest checks;
    this boundary only makes the explicit-ID selection auditable and testable.
    """

    record = registry.get_generation(candidate_generation_id)
    if record.generation_id != candidate_generation_id:
        raise BenchmarkCliError(
            "registry returned a different generation than requested"
        )
    return record


def _operational_outcome(
    *, execution: PairedBenchmarkExecution
) -> OperationalBenchmarkOutcome:
    details = execution.operational
    baseline = details.baseline
    candidate = details.candidate
    hydrated_parent_count = sum(
        diagnostic.hydrated_parent_count
        for diagnostic in execution.diagnostics
        if diagnostic.arm == "candidate"
    )
    return OperationalBenchmarkOutcome(
        schema_version="operational_benchmark_outcome_v2",
        dataset_id=details.dataset_id,
        gold_dataset_sha256=details.gold_dataset_sha256,
        baseline_run_id=details.baseline_run_id,
        candidate_run_id=details.candidate_run_id,
        candidate_generation_id=details.candidate_generation_id,
        embedding_fingerprint=details.embedding_fingerprint,
        baseline_artifact_manifest_sha256=(
            execution.baseline_input.artifact_manifest_sha256
        ),
        candidate_artifact_manifest_sha256=(
            execution.candidate_input.artifact_manifest_sha256
        ),
        query_count=baseline.query_count,
        baseline_p50_latency_ms=baseline.total.p50_ms,
        baseline_p95_latency_ms=baseline.total.p95_ms,
        candidate_p50_latency_ms=candidate.total.p50_ms,
        candidate_p95_latency_ms=candidate.total.p95_ms,
        baseline_error_count=baseline.error_count,
        candidate_error_count=candidate.error_count,
        baseline_error_rate=baseline.error_rate,
        candidate_error_rate=candidate.error_rate,
        baseline_context_tokens_total=baseline.context_tokens_total,
        candidate_context_tokens_total=candidate.context_tokens_total,
        baseline_context_tokens_mean=baseline.context_tokens_mean,
        candidate_context_tokens_mean=candidate.context_tokens_mean,
        baseline_context_tokens_p95=baseline.context_tokens_p95,
        candidate_context_tokens_p95=candidate.context_tokens_p95,
        parent_context_token_ratio=details.parent_context_token_ratio,
        parent_hydration_attempt_count=hydrated_parent_count,
        parent_hydration_success_count=hydrated_parent_count,
        orphan_child_count=details.orphan_child_count,
        parent_hydration_failure_count=details.parent_hydration_failure_count,
        generation_mismatch_count=details.generation_mismatch_count,
    )


def _write_success_artifacts(
    *,
    root: Path,
    output_dir: Path,
    execution: PairedBenchmarkExecution,
    operational: OperationalBenchmarkOutcome,
) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid4().hex}"
    stage.mkdir()
    try:
        atomic_write_project_bytes(
            root,
            stage / "baseline_retrieval_input.json",
            model_json_bytes(execution.baseline_input),
            overwrite=False,
        )
        atomic_write_project_bytes(
            root,
            stage / "candidate_retrieval_input.json",
            model_json_bytes(execution.candidate_input),
            overwrite=False,
        )
        atomic_write_project_bytes(
            root,
            stage / "operational_outcome.json",
            model_json_bytes(operational),
            overwrite=False,
        )
        atomic_write_project_bytes(
            root,
            stage / "operational_details.json",
            model_json_bytes(execution.operational),
            overwrite=False,
        )
        diagnostics = b"".join(
            json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for item in execution.diagnostics
        )
        atomic_write_project_bytes(
            root,
            stage / "benchmark_diagnostics.jsonl",
            diagnostics,
            overwrite=False,
        )
        stage.replace(output_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _write_failure_artifact(*, root: Path, output_dir: Path, error: Exception) -> None:
    failure_path = output_dir.with_name(output_dir.name + ".failure.json")
    if failure_path.exists():
        return
    try:
        atomic_write_project_bytes(
            root,
            failure_path,
            model_json_bytes(
                BenchmarkFailureArtifact(
                    schema_version="parent_child_benchmark_failure_v1",
                    failure_code="benchmark_failed",
                    failure_type=type(error).__name__,
                )
            ),
            overwrite=False,
        )
    except (FileExistsError, OSError, ProjectPathError):
        return


def run_benchmark(
    *,
    project_root: Path,
    index_config_path: Path,
    gold_dataset_path: Path,
    baseline_persist_directory: Path,
    baseline_manifest_path: Path,
    candidate_generation_id: str,
    output_directory: Path,
) -> OperationalBenchmarkOutcome:
    """Run both explicit arms and publish success artifacts only after completion."""

    root = resolve_project_root(project_root)
    config_path = require_project_file(root, index_config_path)
    gold_path = require_project_file(root, gold_dataset_path)
    baseline_dir = require_project_directory(root, baseline_persist_directory)
    manifest_path = require_project_file(root, baseline_manifest_path)
    output_dir = resolve_project_path(root, output_directory, must_exist=False)
    config = _load_index_config(root=root, config_path=config_path)
    if baseline_dir == config.storage.index_root or baseline_dir.is_relative_to(
        config.storage.index_root
    ):
        raise BenchmarkCliError(
            "baseline persist directory must be separate from index_root"
        )
    dataset, gold_sha256 = _load_canonical_gold(gold_path)
    flat_manifest, flat_manifest_sha256 = _load_flat_manifest(manifest_path)
    _assert_flat_manifest_matches_config(config=config, manifest=flat_manifest)
    tokenizer = ConfiguredJiebaTokenizer(config=config.bm25)
    embedding = StrictEmbeddingClient.production(config=config.embedding)
    reranker = StrictRerankerClient.production(config=config.reranker)
    registry: GenerationRegistry | None = None
    candidate_runtime = None
    flat_runtime: FlatBaselineRuntime | None = None
    try:
        registry = GenerationRegistry.open(
            config.storage.resolved_registry_path(),
            index_root=config.storage.index_root,
            expected_schema_version=config.storage.registry_schema_version,
            marker_schema_version=config.storage.owner_marker_schema_version,
            busy_timeout_seconds=config.storage.registry_busy_timeout_seconds,
        )
        record = _select_requested_generation(
            registry=registry,
            candidate_generation_id=candidate_generation_id,
        )
        candidate_runtime = load_generation_runtime(
            config=config,
            registry_record=record,
            query_embedding_provider=embedding,
            reranker=reranker,
            bm25_tokenizer=tokenizer,
        )
        flat_runtime = FlatBaselineRuntime.from_canonical_artifact(
            project_root=root,
            persist_directory=baseline_dir,
            manifest=flat_manifest,
            query_embedding_provider=embedding,
            reranker=reranker,
            tokenizer=tokenizer,
            read_page_size=config.embedding.batch_size,
        )
        baseline_binding = BenchmarkRunBinding(
            schema_version="benchmark_run_binding_v1",
            run_id="flat_baseline_" + flat_manifest_sha256[:16],
            dataset_id=dataset.dataset_id,
            gold_dataset_sha256=gold_sha256,
            embedding_fingerprint=flat_manifest.embedding.fingerprint,
            retrieval_fingerprint=compute_flat_retrieval_fingerprint(
                manifest=flat_manifest,
                vector_top_k=config.retrieval.vector_top_k,
                bm25_top_k=config.retrieval.bm25_top_k,
                reranker_top_n=config.retrieval.reranker_top_n,
            ),
            implementation_kind="flat_baseline",
            artifact_manifest_sha256=flat_manifest_sha256,
            generation_id=None,
        )
        candidate_binding = BenchmarkRunBinding(
            schema_version="benchmark_run_binding_v1",
            run_id="parent_child_"
            + candidate_runtime.retrieval_policy.generation_manifest_sha256[:16],
            dataset_id=dataset.dataset_id,
            gold_dataset_sha256=gold_sha256,
            embedding_fingerprint=candidate_runtime.retrieval_policy.embedding_fingerprint,
            retrieval_fingerprint=compute_retrieval_fingerprint(
                candidate_runtime.retrieval_policy
            ),
            implementation_kind="parent_child_candidate",
            artifact_manifest_sha256=candidate_runtime.retrieval_policy.generation_manifest_sha256,
            generation_id=candidate_generation_id,
        )
        retriever = candidate_runtime.retriever()
        execution = run_paired_benchmark(
            dataset=dataset,
            baseline_binding=baseline_binding,
            candidate_binding=candidate_binding,
            baseline_retrieve=lambda query: flat_runtime.retrieve(
                query=query.query,
                subject=query.subject,
                vector_top_k=config.retrieval.vector_top_k,
                bm25_top_k=config.retrieval.bm25_top_k,
                reranker_top_n=config.retrieval.reranker_top_n,
            ),
            candidate_retrieve=lambda query: retriever.retrieve(
                HybridRetrievalRequest(
                    schema_version="hybrid_retrieval_request_v1",
                    request_id="benchmark_" + query.query_id,
                    query=query.query,
                    subject=query.subject,
                    generation_id=candidate_generation_id,
                )
            ),
            token_counter=estimate_text_tokens_mixed,
        )
        operational = _operational_outcome(execution=execution)
        _write_success_artifacts(
            root=root,
            output_dir=output_dir,
            execution=execution,
            operational=operational,
        )
        return operational
    finally:
        if flat_runtime is not None:
            flat_runtime.close()
        if candidate_runtime is not None:
            candidate_runtime.close()
        if registry is not None:
            registry.close()
        reranker.close()
        embedding.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        outcome = run_benchmark(
            project_root=args.project_root,
            index_config_path=args.index_config,
            gold_dataset_path=args.gold_dataset,
            baseline_persist_directory=args.baseline_persist_dir,
            baseline_manifest_path=args.baseline_manifest,
            candidate_generation_id=args.candidate_generation_id,
            output_directory=args.output_dir,
        )
    except Exception as exc:
        try:
            root = resolve_project_root(args.project_root)
            output = resolve_project_path(root, args.output_dir, must_exist=False)
            _write_failure_artifact(root=root, output_dir=output, error=exc)
        except Exception:
            pass
        print(f"Parent-child benchmark failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        "Parent-child benchmark complete: "
        f"candidate_generation={outcome.candidate_generation_id}, "
        f"candidate_p95_ms={outcome.candidate_p95_latency_ms:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
