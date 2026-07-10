"""Validate splitter policies with temporary vector retrieval indexes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.rag.eval.vector_policy_validator import (  # noqa: E402
    DEFAULT_INDEX_ROOT,
    DEFAULT_MAX_POLICIES,
    DEFAULT_POLICY_REPORT,
    DEFAULT_TOP_K,
    RetrievalPolicyValidationConfig,
    validate_retrieval_policies,
)


def load_project_env() -> None:
    """Load project .env for parity with build_index.py without printing values."""

    load_dotenv(project_root / ".env")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate splitter policies with policy-independent evidence hits."
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output-dir", type=Path, default=project_root / "reports")
    parser.add_argument(
        "--index-root", type=Path, default=project_root / DEFAULT_INDEX_ROOT
    )
    parser.add_argument(
        "--policy-report", type=Path, default=project_root / DEFAULT_POLICY_REPORT
    )
    parser.add_argument(
        "--max-policies", type=_positive_int, default=DEFAULT_MAX_POLICIES
    )
    parser.add_argument("--max-queries", type=_positive_int)
    parser.add_argument(
        "--top-k", nargs="+", type=_positive_int, default=list(DEFAULT_TOP_K)
    )
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--force-rebuild-index", action="store_true")
    args = parser.parse_args(argv)

    if args.trace_output is not None and not args.trace:
        parser.error("--trace-output can only be used together with --trace")
    if args.reuse_index and args.force_rebuild_index:
        parser.error("--reuse-index and --force-rebuild-index cannot both be used")
    return args


def main(argv: list[str] | None = None) -> None:
    load_project_env()
    args = parse_args(argv)
    result = validate_retrieval_policies(
        RetrievalPolicyValidationConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            index_root=args.index_root,
            policy_report=args.policy_report,
            max_policies=args.max_policies,
            max_queries=args.max_queries,
            top_k=tuple(sorted(set(args.top_k))),
            subjects=tuple(args.subject),
            trace_enabled=args.trace,
            trace_output=args.trace_output,
            reuse_index=args.reuse_index,
            force_rebuild_index=args.force_rebuild_index,
            project_root=project_root,
        )
    )
    recommendation = result["recommendation_report"]
    print(f"Dataset report saved             : {result['dataset_report_path']}")
    print(f"Candidates report saved          : {result['candidates_report_path']}")
    print(f"Subject report saved             : {result['subject_report_path']}")
    print(f"Recommendation report saved      : {result['recommendation_report_path']}")
    print(f"Global action                    : {recommendation['global_action']}")
    print(f"Global best policy               : {recommendation['global_best_policy']}")
    if result.get("trace_path"):
        print(f"Retrieval trace saved            : {result['trace_path']}")


if __name__ == "__main__":
    main()
