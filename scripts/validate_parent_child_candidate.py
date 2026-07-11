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
    atomic_write_bytes,
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
    return parser


def _contained_path(
    project_root: Path,
    value: Path,
    *,
    must_exist: bool,
) -> Path:
    candidate = value if value.is_absolute() else project_root / value
    if candidate.is_symlink():
        raise ValueError("validation paths must not be symlinks")
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(project_root):
        raise ValueError("validation paths must remain inside project_root")
    if must_exist and not resolved.is_file():
        raise ValueError("validation input must be a regular non-symlink file")
    return resolved


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
) -> CandidateValidationArtifact:
    root = project_root.resolve(strict=True)
    config_path = _contained_path(root, benchmark_config_path, must_exist=True)
    gold_path = _contained_path(root, gold_dataset_path, must_exist=True)
    baseline_path = _contained_path(root, baseline_input_path, must_exist=True)
    candidate_path = _contained_path(root, candidate_input_path, must_exist=True)
    operational_path = _contained_path(root, operational_outcome_path, must_exist=True)
    end_to_end_path = _contained_path(root, end_to_end_outcome_path, must_exist=True)
    output = _contained_path(root, output_path, must_exist=False)

    benchmark = load_rag_benchmark_config(config_path)
    gold_bytes = gold_path.read_bytes()
    dataset = GoldDataset.model_validate_json(gold_bytes)
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
        schema_version="candidate_validation_artifact_v1",
        benchmark_config_sha256=sha256_bytes(config_path.read_bytes()),
        gold_dataset_sha256=sha256_bytes(gold_bytes),
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        comparison=comparison,
        operational=operational,
        end_to_end=end_to_end,
        eligibility=eligibility,
    )
    atomic_write_bytes(
        root,
        output.relative_to(root).as_posix(),
        model_json_bytes(artifact),
        overwrite=True,
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
    )
    print(
        "Candidate validation written: "
        f"activation_eligible={artifact.eligibility.activation_eligible}, "
        f"blocked={artifact.eligibility.production_recommendation_blocked}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
