"""Evaluate RAG chunking modes without indexing side effects."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.rag.chunking.splitter_factory import VALID_SPLITTER_MODES  # noqa: E402
from src.rag.eval.chunk_evaluator import (  # noqa: E402
    ChunkEvaluationConfig,
    compare_modes,
    evaluate_mode,
)


def _mode(value: str) -> str:
    if value not in VALID_SPLITTER_MODES:
        expected = ", ".join(VALID_SPLITTER_MODES)
        raise argparse.ArgumentTypeError(
            f"Invalid mode {value!r}. Expected one of: {expected}."
        )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recursive vs structure RAG chunking modes."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--mode", type=_mode)
    mode_group.add_argument("--compare", nargs=2, type=_mode, metavar=("BASE", "CAND"))
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output-dir", type=Path, default=project_root / "reports")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--too-short-chars", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args(argv)

    if args.output is not None and args.mode is None:
        parser.error("--output can only be used with --mode")
    if args.sample_limit is not None and args.sample_limit < 0:
        parser.error("--sample-limit must be >= 0")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be > 0")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    subjects = tuple(args.subject)
    trace_enabled = not args.no_trace

    if args.mode:
        report = evaluate_mode(
            ChunkEvaluationConfig(
                mode=args.mode,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                output_path=args.output,
                subjects=subjects,
                too_short_chars=args.too_short_chars,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                sample_limit=args.sample_limit,
                trace_enabled=trace_enabled,
                trace_output=args.trace_output,
                project_root=project_root,
            )
        )
        output_path = args.output or args.output_dir / f"chunk_eval_{args.mode}.json"
        print(f"Report saved: {output_path}")
        if report.get("trace_enabled"):
            print(f"Trace saved : {report.get('trace_path')}")
        return

    baseline_mode, candidate_mode = args.compare
    report = compare_modes(
        baseline_mode=baseline_mode,
        candidate_mode=candidate_mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        subjects=subjects,
        too_short_chars=args.too_short_chars,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sample_limit=args.sample_limit,
        trace_enabled=trace_enabled,
        trace_output=args.trace_output,
        project_root=project_root,
    )
    print(f"Compare report saved: {args.output_dir / 'chunk_eval_compare.json'}")
    if report.get("trace_enabled"):
        print(f"Trace saved         : {report.get('trace_path')}")


if __name__ == "__main__":
    main()
