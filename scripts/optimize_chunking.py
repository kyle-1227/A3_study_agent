"""Run read-only chunk policy optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.rag.chunking.splitter_factory import VALID_SPLITTER_MODES  # noqa: E402
from src.rag.eval.chunk_optimizer import (  # noqa: E402
    ChunkOptimizerConfig,
    optimize_chunking,
)


def _mode(value: str) -> str:
    if value not in VALID_SPLITTER_MODES:
        expected = ", ".join(VALID_SPLITTER_MODES)
        raise argparse.ArgumentTypeError(
            f"Invalid mode {value!r}. Expected one of: {expected}."
        )
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate chunking candidates and write read-only recommendations."
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output-dir", type=Path, default=project_root / "reports")
    parser.add_argument("--modes", nargs="+", type=_mode)
    parser.add_argument("--chunk-sizes", nargs="+", type=_positive_int)
    parser.add_argument("--overlaps", nargs="+", type=_non_negative_int)
    parser.add_argument("--too-short-chars", type=_positive_int, default=80)
    parser.add_argument("--sample-limit", type=_non_negative_int)
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--max-candidates", type=_positive_int)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args(argv)

    if args.trace_output is not None and not args.trace:
        parser.error("--trace-output can only be used together with --trace")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = optimize_chunking(
        ChunkOptimizerConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            modes=tuple(args.modes) if args.modes is not None else VALID_SPLITTER_MODES,
            chunk_sizes=tuple(args.chunk_sizes)
            if args.chunk_sizes is not None
            else None,
            overlaps=tuple(args.overlaps) if args.overlaps is not None else None,
            too_short_chars=args.too_short_chars,
            sample_limit=args.sample_limit,
            subjects=tuple(args.subject),
            max_candidates=args.max_candidates,
            trace_enabled=args.trace,
            trace_output=args.trace_output,
            project_root=project_root,
        )
    )
    recommendation = result["recommendation_report"]["recommendation"]
    print(f"Candidates report saved: {result['candidates_report_path']}")
    print(f"Optimizer report saved : {result['recommendation_report_path']}")
    print(f"Recommendation action  : {recommendation['action']}")
    if result.get("trace_path"):
        print(f"Optimizer trace saved  : {result['trace_path']}")


if __name__ == "__main__":
    main()
