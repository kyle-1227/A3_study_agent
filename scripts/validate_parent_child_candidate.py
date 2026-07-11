"""Validate baseline and candidate retrieval projections on one fixed gold dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.rag_benchmark_config import load_rag_benchmark_config  # noqa: E402
from src.rag.parent_child._storage_io import (  # noqa: E402
    canonical_json_bytes,
    model_json_bytes,
    sha256_bytes,
)
from src.rag.parent_child.evaluation import (  # noqa: E402
    BootstrapConfig,
    ComparisonMetricSpec,
    EvaluationGateConfig,
    GoldDataset,
    RetrievalEvaluationInput,
    compare_paired_reports,
    evaluate_retrieval_run,
)
from src.rag.parent_child.evaluation_gate import (  # noqa: E402
    CandidateValidationArtifact,
    EndToEndQualityOutcome,
    OperationalBenchmarkOutcome,
    evaluate_activation_eligibility,
)
from src.rag.parent_child.project_paths import (  # noqa: E402
    atomic_write_project_bytes,
    require_project_file,
    resolve_project_path,
    resolve_project_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--gold-dataset", type=Path, required=True)
    parser.add_argument("--baseline-input", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--operational-outcome", type=Path, required=True)
    parser.add_argument("--end-to-end-outcome", type=Path, required=True)
    parser.add_argument(
        "--functional-tests-passed",
        choices=("true", "false"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_run_bindings(
    *,
    dataset: GoldDataset,
    gold_dataset_sha256: str,
    baseline_input: RetrievalEvaluationInput,
    candidate_input: RetrievalEvaluationInput,
    operational: OperationalBenchmarkOutcome,
    end_to_end: EndToEndQualityOutcome,
) -> None:
    """Reject cross-run artifact mixing before formal metrics are calculated."""

    if baseline_input.implementation_kind != "flat_baseline":
        raise ValueError("baseline input must declare flat_baseline")
    if candidate_input.implementation_kind != "parent_child_candidate":
        raise ValueError("candidate input must declare parent_child_candidate")
    if baseline_input.parent_aware or not candidate_input.parent_aware:
        raise ValueError("baseline/candidate parent-aware contracts are invalid")
    if candidate_input.generation_id is None:
        raise ValueError("candidate input requires an explicit generation_id")
    if (
        baseline_input.dataset_id != dataset.dataset_id
        or candidate_input.dataset_id != dataset.dataset_id
    ):
        raise ValueError("retrieval inputs must bind the supplied GoldDataset")
    if (
        baseline_input.gold_dataset_sha256 != gold_dataset_sha256
        or candidate_input.gold_dataset_sha256 != gold_dataset_sha256
    ):
        raise ValueError("retrieval inputs must bind the supplied GoldDataset digest")
    if baseline_input.embedding_fingerprint != candidate_input.embedding_fingerprint:
        raise ValueError("baseline and candidate embeddings differ")
    if (
        operational.dataset_id != dataset.dataset_id
        or operational.gold_dataset_sha256 != gold_dataset_sha256
        or operational.baseline_run_id != baseline_input.run_id
        or operational.candidate_run_id != candidate_input.run_id
        or operational.candidate_generation_id != candidate_input.generation_id
        or operational.embedding_fingerprint != baseline_input.embedding_fingerprint
        or operational.baseline_artifact_manifest_sha256
        != baseline_input.artifact_manifest_sha256
        or operational.candidate_artifact_manifest_sha256
        != candidate_input.artifact_manifest_sha256
        or operational.query_count != len(dataset.queries)
    ):
        raise ValueError(
            "operational outcome does not bind the supplied retrieval runs"
        )
    if (
        end_to_end.dataset_id != dataset.dataset_id
        or end_to_end.gold_dataset_sha256 != gold_dataset_sha256
        or end_to_end.baseline_run_id != baseline_input.run_id
        or end_to_end.candidate_run_id != candidate_input.run_id
        or end_to_end.assessment_source != "human"
    ):
        raise ValueError("end-to-end outcome does not bind the supplied retrieval runs")


def run_validation(
    *,
    project_root: Path,
    benchmark_config_path: Path,
    gold_dataset_path: Path,
    baseline_input_path: Path,
    candidate_input_path: Path,
    operational_outcome_path: Path,
    end_to_end_outcome_path: Path,
    functional_tests_passed: bool,
    output_path: Path,
    overwrite: bool,
) -> CandidateValidationArtifact:
    root = resolve_project_root(project_root)
    config_path = require_project_file(root, benchmark_config_path)
    gold_path = require_project_file(root, gold_dataset_path)
    baseline_path = require_project_file(root, baseline_input_path)
    candidate_path = require_project_file(root, candidate_input_path)
    operational_path = require_project_file(root, operational_outcome_path)
    end_to_end_path = require_project_file(root, end_to_end_outcome_path)
    output = resolve_project_path(root, output_path, must_exist=False)

    benchmark = load_rag_benchmark_config(config_path)
    gold_bytes = gold_path.read_bytes()
    dataset = GoldDataset.model_validate_json(gold_bytes)
    if gold_bytes != canonical_json_bytes(dataset.model_dump(mode="json")):
        raise ValueError("gold_dataset must use canonical JSON serialization")
    baseline_input = RetrievalEvaluationInput.model_validate_json(
        baseline_path.read_bytes()
    )
    candidate_input = RetrievalEvaluationInput.model_validate_json(
        candidate_path.read_bytes()
    )
    operational = OperationalBenchmarkOutcome.model_validate_json(
        operational_path.read_bytes()
    )
    end_to_end = EndToEndQualityOutcome.model_validate_json(
        end_to_end_path.read_bytes()
    )
    _validate_run_bindings(
        dataset=dataset,
        gold_dataset_sha256=sha256_bytes(gold_bytes),
        baseline_input=baseline_input,
        candidate_input=candidate_input,
        operational=operational,
        end_to_end=end_to_end,
    )
    gate = EvaluationGateConfig(
        schema_version="evaluation_gate_config_v1",
        primary_subjects=benchmark.primary_subjects,
        min_global_rollout_queries=benchmark.min_global_gold_queries,
        min_subject_rollout_queries=benchmark.min_subject_gold_queries,
        min_independent_sources_per_subject=benchmark.min_independent_sources,
    )
    baseline_report = evaluate_retrieval_run(
        dataset,
        baseline_input,
        top_ks=benchmark.top_ks,
        parent_top_ks=benchmark.parent_top_ks if baseline_input.parent_aware else (),
        gate_config=gate,
    )
    candidate_report = evaluate_retrieval_run(
        dataset,
        candidate_input,
        top_ks=benchmark.top_ks,
        parent_top_ks=benchmark.parent_top_ks if candidate_input.parent_aware else (),
        gate_config=gate,
    )
    comparison = compare_paired_reports(
        baseline_report,
        candidate_report,
        metric_specs=(
            ComparisonMetricSpec(
                schema_version="comparison_metric_spec_v1",
                metric_name="evidence_recall_at_k",
                k=5,
            ),
            ComparisonMetricSpec(
                schema_version="comparison_metric_spec_v1",
                metric_name="mrr",
                k=None,
            ),
            ComparisonMetricSpec(
                schema_version="comparison_metric_spec_v1",
                metric_name="noise_at_k",
                k=5,
            ),
        ),
        bootstrap=BootstrapConfig(
            schema_version="paired_bootstrap_config_v1",
            iterations=benchmark.bootstrap_samples,
            seed=benchmark.bootstrap_seed,
            confidence=benchmark.bootstrap_confidence,
        ),
        eligible_queries_only=True,
    )
    eligibility = evaluate_activation_eligibility(
        benchmark=benchmark,
        comparison=comparison,
        operational=operational,
        end_to_end=end_to_end,
        functional_tests_passed=functional_tests_passed,
    )
    artifact = CandidateValidationArtifact(
        schema_version="candidate_validation_artifact_v2",
        benchmark_config_sha256=sha256_bytes(config_path.read_bytes()),
        gold_dataset_sha256=sha256_bytes(gold_bytes),
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        comparison=comparison,
        operational=operational,
        end_to_end=end_to_end,
        eligibility=eligibility,
    )
    atomic_write_project_bytes(
        root,
        output,
        model_json_bytes(artifact),
        overwrite=overwrite,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = run_validation(
        project_root=args.project_root,
        benchmark_config_path=args.benchmark_config,
        gold_dataset_path=args.gold_dataset,
        baseline_input_path=args.baseline_input,
        candidate_input_path=args.candidate_input,
        operational_outcome_path=args.operational_outcome,
        end_to_end_outcome_path=args.end_to_end_outcome,
        functional_tests_passed=args.functional_tests_passed == "true",
        output_path=args.output,
        overwrite=args.overwrite,
    )
    print(
        "Candidate validation written: "
        f"activation_eligible={artifact.eligibility.activation_eligible}, "
        f"blocked={artifact.eligibility.production_recommendation_blocked}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
