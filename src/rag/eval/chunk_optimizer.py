"""Read-only chunk policy optimizer built on Phase 4D evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.rag.chunking.splitter_factory import VALID_SPLITTER_MODES
from src.rag.eval.chunk_evaluator import ChunkEvaluationConfig, evaluate_mode
from src.rag.eval.chunk_metrics import sanitized_preview

DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_TOO_SHORT_CHARS = 80
BASELINE_CANDIDATE_NAME = "recursive_size1000_overlap200"
DEFAULT_POLICY_PAIRS = ((700, 100), (900, 150), (1000, 200), (1200, 200))
DEFAULT_CHUNK_SIZES = (700, 900, 1000, 1200)
DEFAULT_OVERLAPS = (100, 150, 200)
TOP_CANDIDATE_LIMIT = 5
ERROR_MESSAGE_CHARS = 160

DEFAULT_OPTIMIZER_WEIGHTS = {
    "metadata": 0.25,
    "size": 0.25,
    "section": 0.20,
    "short_chunk_penalty": 0.15,
    "duplicate_penalty": 0.05,
    "chunk_count_penalty": 0.10,
}


@dataclass(frozen=True)
class OptimizerThresholds:
    """Review thresholds for candidate policy scoring."""

    too_short_ratio_review_threshold: float = 0.10
    duplicate_ratio_review_threshold: float = 0.02
    chunk_count_review_ratio: float = 1.30
    high_cost_ratio: float = 1.20
    close_score_delta: float = 0.03
    minimum_consider_improvement: float = 0.05


@dataclass(frozen=True)
class ChunkPolicyCandidate:
    """Stable chunk policy candidate evaluated by the optimizer."""

    name: str
    splitter_mode: str
    chunk_size: int
    chunk_overlap: int
    too_short_chars: int = DEFAULT_TOO_SHORT_CHARS


@dataclass(frozen=True)
class ChunkOptimizerConfig:
    """Configuration for a read-only chunk optimizer run."""

    data_dir: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    modes: tuple[str, ...] = VALID_SPLITTER_MODES
    chunk_sizes: tuple[int, ...] | None = None
    overlaps: tuple[int, ...] | None = None
    too_short_chars: int = DEFAULT_TOO_SHORT_CHARS
    sample_limit: int | None = None
    subjects: tuple[str, ...] = ()
    max_candidates: int | None = None
    trace_enabled: bool = False
    trace_output: Path | None = None
    run_id: str | None = None
    project_root: Path | None = None
    thresholds: OptimizerThresholds = OptimizerThresholds()


class OptimizerTraceWriter:
    """JSONL writer for optimizer-level events only."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Path,
        trace_output: Path | None,
        run_id: str,
        project_root: Path,
    ) -> None:
        self.enabled = enabled
        self.project_root = project_root
        path = (
            trace_output
            if trace_output is not None
            else output_dir / f"chunk_optimizer_trace_{run_id}.jsonl"
        )
        self.path = path if path.is_absolute() else project_root / path

    @property
    def path_label(self) -> str | None:
        if not self.enabled:
            return None
        return _path_label(self.path, self.project_root)

    def clear(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "timestamp_utc": _utc_now(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _project_root(path: Path | None = None) -> Path:
    return path.resolve() if path is not None else Path(__file__).resolve().parents[3]


def _path_label(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _candidate_name(mode: str, chunk_size: int, chunk_overlap: int) -> str:
    return f"{mode}_size{chunk_size}_overlap{chunk_overlap}"


def _baseline_candidate(too_short_chars: int) -> ChunkPolicyCandidate:
    return ChunkPolicyCandidate(
        name=BASELINE_CANDIDATE_NAME,
        splitter_mode="recursive",
        chunk_size=1000,
        chunk_overlap=200,
        too_short_chars=too_short_chars,
    )


def _validate_modes(modes: tuple[str, ...]) -> None:
    invalid = [mode for mode in modes if mode not in VALID_SPLITTER_MODES]
    if invalid:
        expected = ", ".join(VALID_SPLITTER_MODES)
        raise ValueError(f"Invalid modes {invalid!r}. Expected one of: {expected}.")


def _validate_positive_values(
    *,
    too_short_chars: int,
    sample_limit: int | None,
    max_candidates: int | None,
    chunk_sizes: tuple[int, ...] | None,
    overlaps: tuple[int, ...] | None,
) -> None:
    if too_short_chars <= 0:
        raise ValueError("too_short_chars must be > 0")
    if sample_limit is not None and sample_limit < 0:
        raise ValueError("sample_limit must be >= 0")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if chunk_sizes is not None and any(value <= 0 for value in chunk_sizes):
        raise ValueError("chunk_sizes must all be > 0")
    if overlaps is not None and any(value < 0 for value in overlaps):
        raise ValueError("overlaps must all be >= 0")


def _stable_order(candidate: ChunkPolicyCandidate) -> tuple[int, int, int, str]:
    return (
        VALID_SPLITTER_MODES.index(candidate.splitter_mode),
        candidate.chunk_size,
        candidate.chunk_overlap,
        candidate.name,
    )


def generate_candidates(config: ChunkOptimizerConfig) -> list[ChunkPolicyCandidate]:
    """Generate stable candidates, always preserving the recursive baseline."""

    _validate_modes(config.modes)
    _validate_positive_values(
        too_short_chars=config.too_short_chars,
        sample_limit=config.sample_limit,
        max_candidates=config.max_candidates,
        chunk_sizes=config.chunk_sizes,
        overlaps=config.overlaps,
    )

    candidates: dict[str, ChunkPolicyCandidate] = {}
    use_default_pairs = config.chunk_sizes is None and config.overlaps is None
    if use_default_pairs:
        for mode in config.modes:
            for chunk_size, chunk_overlap in DEFAULT_POLICY_PAIRS:
                candidate = ChunkPolicyCandidate(
                    name=_candidate_name(mode, chunk_size, chunk_overlap),
                    splitter_mode=mode,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    too_short_chars=config.too_short_chars,
                )
                candidates[candidate.name] = candidate
    else:
        chunk_sizes = config.chunk_sizes or DEFAULT_CHUNK_SIZES
        overlaps = config.overlaps or DEFAULT_OVERLAPS
        for mode in config.modes:
            for chunk_size in chunk_sizes:
                for chunk_overlap in overlaps:
                    if chunk_overlap >= chunk_size:
                        continue
                    candidate = ChunkPolicyCandidate(
                        name=_candidate_name(mode, chunk_size, chunk_overlap),
                        splitter_mode=mode,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        too_short_chars=config.too_short_chars,
                    )
                    candidates[candidate.name] = candidate

    baseline = _baseline_candidate(config.too_short_chars)
    candidates[baseline.name] = baseline
    baseline_candidate = candidates[baseline.name]
    non_baseline = sorted(
        (
            candidate
            for candidate in candidates.values()
            if candidate.name != BASELINE_CANDIDATE_NAME
        ),
        key=_stable_order,
    )

    if config.max_candidates is not None:
        return [baseline_candidate, *non_baseline[: config.max_candidates - 1]]
    return [baseline_candidate, *non_baseline]


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    metadata = (
        report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    )
    total_chunks = _integer(summary.get("total_chunks"))
    too_long_count = _integer(summary.get("too_long_count"))
    too_long_ratio = round(too_long_count / total_chunks, 4) if total_chunks else 0.0
    return {
        "total_chunks": total_chunks,
        "source_count": _integer(summary.get("source_count")),
        "too_short_ratio": _number(summary.get("too_short_ratio")),
        "duplicate_ratio": _number(summary.get("duplicate_ratio")),
        "required_metadata_coverage": _number(
            metadata.get("required_metadata_coverage")
        ),
        "section_metadata_coverage": _number(metadata.get("section_metadata_coverage")),
        "empty_chunk_count": _integer(summary.get("empty_chunk_count")),
        "too_long_ratio": too_long_ratio,
    }


def _candidate_report_path(
    candidate: ChunkPolicyCandidate, output_dir: Path, project_root: Path
) -> Path:
    path = output_dir / f"chunk_eval_{candidate.name}.json"
    return path if path.is_absolute() else project_root / path


def _candidate_metrics_for_trace(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_chunks": metrics["total_chunks"],
        "source_count": metrics["source_count"],
        "too_short_ratio": metrics["too_short_ratio"],
        "duplicate_ratio": metrics["duplicate_ratio"],
        "required_metadata_coverage": metrics["required_metadata_coverage"],
        "section_metadata_coverage": metrics["section_metadata_coverage"],
    }


def _score_components(
    *,
    candidate: ChunkPolicyCandidate,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    thresholds: OptimizerThresholds,
) -> dict[str, float | None]:
    baseline_total = float(baseline_metrics["total_chunks"])
    chunk_count_ratio = _ratio(float(metrics["total_chunks"]), baseline_total)
    too_long_ratio = float(metrics["too_long_ratio"])
    too_short_ratio = float(metrics["too_short_ratio"])
    duplicate_ratio = float(metrics["duplicate_ratio"])
    metadata_score = _clamp(float(metrics["required_metadata_coverage"]))
    size_score = _clamp(
        1.0
        - (0.7 * _clamp(too_short_ratio / thresholds.too_short_ratio_review_threshold))
        - (0.3 * _clamp(too_long_ratio / thresholds.too_short_ratio_review_threshold))
    )
    section_score = (
        _clamp(float(metrics["section_metadata_coverage"]))
        if candidate.splitter_mode == "structure"
        else 0.0
    )
    short_chunk_penalty = _clamp(
        too_short_ratio / thresholds.too_short_ratio_review_threshold
    )
    duplicate_penalty = _clamp(
        duplicate_ratio / thresholds.duplicate_ratio_review_threshold
    )
    chunk_count_penalty = 0.0
    if chunk_count_ratio is not None and chunk_count_ratio > 1.0:
        chunk_count_penalty = _clamp(
            (chunk_count_ratio - 1.0) / (thresholds.chunk_count_review_ratio - 1.0)
        )
    return {
        "metadata_score": metadata_score,
        "size_score": size_score,
        "section_score": section_score,
        "short_chunk_penalty": short_chunk_penalty,
        "duplicate_penalty": duplicate_penalty,
        "chunk_count_penalty": chunk_count_penalty,
        "chunk_count_ratio": chunk_count_ratio,
    }


def _score_from_components(components: dict[str, float | None]) -> float:
    return round(
        _clamp(
            DEFAULT_OPTIMIZER_WEIGHTS["metadata"] * float(components["metadata_score"])
            + DEFAULT_OPTIMIZER_WEIGHTS["size"] * float(components["size_score"])
            + DEFAULT_OPTIMIZER_WEIGHTS["section"] * float(components["section_score"])
            + DEFAULT_OPTIMIZER_WEIGHTS["short_chunk_penalty"]
            * (1.0 - float(components["short_chunk_penalty"]))
            + DEFAULT_OPTIMIZER_WEIGHTS["duplicate_penalty"]
            * (1.0 - float(components["duplicate_penalty"]))
            + DEFAULT_OPTIMIZER_WEIGHTS["chunk_count_penalty"]
            * (1.0 - float(components["chunk_count_penalty"]))
        ),
        4,
    )


def _initial_status(
    *,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    components: dict[str, float | None],
    thresholds: OptimizerThresholds,
) -> tuple[str, list[str]]:
    fail_reasons: list[str] = []
    if int(metrics["total_chunks"]) == 0:
        fail_reasons.append("total_chunks is 0")
    if int(metrics["empty_chunk_count"]) > 0:
        fail_reasons.append("empty chunks detected")
    if float(metrics["required_metadata_coverage"]) < 1.0:
        fail_reasons.append("required metadata coverage below 1.0")
    if int(metrics["source_count"]) < int(baseline_metrics["source_count"]):
        fail_reasons.append("source count below baseline")
    if fail_reasons:
        return "fail", fail_reasons

    review_reasons: list[str] = []
    if float(metrics["too_short_ratio"]) > thresholds.too_short_ratio_review_threshold:
        review_reasons.append("too_short_ratio above review threshold")
    if float(metrics["duplicate_ratio"]) > thresholds.duplicate_ratio_review_threshold:
        review_reasons.append("duplicate_ratio above review threshold")
    chunk_count_ratio = components["chunk_count_ratio"]
    if (
        chunk_count_ratio is not None
        and chunk_count_ratio > thresholds.chunk_count_review_ratio
    ):
        review_reasons.append("chunk count ratio above review threshold")
    if review_reasons:
        return "needs_review", review_reasons

    return "pass", []


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": candidate["name"],
        "splitter_mode": candidate["splitter_mode"],
        "chunk_size": candidate["chunk_size"],
        "chunk_overlap": candidate["chunk_overlap"],
        "score": candidate["score"],
        "status": candidate["status"],
    }


def _rank_key(candidate: dict[str, Any]) -> tuple[int, float, int, str]:
    status_rank = {"pass": 0, "needs_review": 1, "fail": 2}
    total_chunks = int(candidate["metrics"].get("total_chunks", 0))
    return (
        status_rank.get(str(candidate["status"]), 3),
        -float(candidate["score"]),
        total_chunks,
        str(candidate["name"]),
    )


def _select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    non_fail = [candidate for candidate in candidates if candidate["status"] != "fail"]
    if non_fail:
        return sorted(non_fail, key=_rank_key)[0]
    baseline = next(
        candidate
        for candidate in candidates
        if candidate["name"] == BASELINE_CANDIDATE_NAME
    )
    return baseline


def _apply_close_score_review(
    candidates: list[dict[str, Any]], *, thresholds: OptimizerThresholds
) -> None:
    non_fail_scores = [
        float(candidate["score"])
        for candidate in candidates
        if candidate["status"] != "fail"
    ]
    if not non_fail_scores:
        return
    best_score = max(non_fail_scores)
    for candidate in candidates:
        if candidate["status"] == "fail":
            continue
        chunk_count_ratio = candidate["metrics"].get("chunk_count_ratio")
        if (
            chunk_count_ratio is not None
            and float(chunk_count_ratio) > thresholds.high_cost_ratio
            and best_score - float(candidate["score"]) <= thresholds.close_score_delta
        ):
            candidate["status"] = "needs_review"
            candidate["reasons"].append(
                "score close to best candidate but chunk cost is higher"
            )


def _build_recommendation(
    *,
    baseline: dict[str, Any],
    best: dict[str, Any],
    sampled: bool,
    thresholds: OptimizerThresholds,
) -> dict[str, Any]:
    if sampled:
        action = (
            "keep_current_default"
            if best["name"] == baseline["name"]
            else "needs_manual_review"
        )
        reason = "sampled optimizer run cannot recommend automatic candidate adoption"
    elif baseline["status"] == "fail":
        action = "needs_manual_review"
        reason = "baseline candidate failed hard checks"
    elif best["name"] == baseline["name"]:
        action = "keep_current_default"
        reason = "baseline remains the most stable candidate"
    elif best["status"] != "pass":
        action = "needs_manual_review"
        reason = "best candidate has review-level risks"
    elif float(best["score"]) - float(
        baseline["score"]
    ) >= thresholds.minimum_consider_improvement and (
        best["metrics"].get("chunk_count_ratio") is None
        or float(best["metrics"]["chunk_count_ratio"]) <= thresholds.high_cost_ratio
    ):
        action = "consider_candidate"
        reason = "best candidate improves intrinsic score without high chunk cost"
    elif float(best["score"]) > float(baseline["score"]):
        action = "needs_manual_review"
        reason = "candidate improves score but not enough for a clear recommendation"
    else:
        action = "keep_current_default"
        reason = "candidate improvement is insufficient"
    return {
        "action": action,
        "reason": reason,
        "do_not_auto_apply": True,
    }


def _build_reports(
    *,
    candidates: list[dict[str, Any]],
    config: ChunkOptimizerConfig,
    trace: OptimizerTraceWriter,
    project_root: Path,
    generated_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = next(
        candidate
        for candidate in candidates
        if candidate["name"] == BASELINE_CANDIDATE_NAME
    )
    best = _select_best_candidate(candidates)
    sampled = config.sample_limit is not None
    recommendation = _build_recommendation(
        baseline=baseline,
        best=best,
        sampled=sampled,
        thresholds=config.thresholds,
    )
    ranked = sorted(candidates, key=_rank_key)
    warnings: list[str] = []
    failed_count = sum(1 for candidate in candidates if candidate["status"] == "fail")
    if failed_count:
        warnings.append(f"{failed_count} candidate(s) failed")
    if sampled:
        warnings.append("sample_limit enabled; recommendation requires manual review")
    if baseline["status"] == "fail":
        warnings.append("baseline candidate failed")

    candidates_report = {
        "generated_at_utc": generated_at_utc,
        "data_dir": _path_label(
            config.data_dir
            if config.data_dir.is_absolute()
            else project_root / config.data_dir,
            project_root,
        ),
        "sampled": sampled,
        "sample_limit": config.sample_limit,
        "baseline_candidate": BASELINE_CANDIDATE_NAME,
        "trace_enabled": trace.enabled,
        "trace_path": trace.path_label,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    recommendation_report = {
        "generated_at_utc": generated_at_utc,
        "baseline": _candidate_summary(baseline),
        "best_candidate": _candidate_summary(best),
        "top_candidates": [
            _candidate_summary(candidate) for candidate in ranked[:TOP_CANDIDATE_LIMIT]
        ],
        "recommendation": recommendation,
        "warnings": warnings,
    }
    return candidates_report, recommendation_report


def _candidate_result_from_report(
    *,
    candidate: ChunkPolicyCandidate,
    report: dict[str, Any],
    baseline_metrics: dict[str, Any],
    report_path: Path,
    project_root: Path,
    thresholds: OptimizerThresholds,
) -> dict[str, Any]:
    metrics = _metrics_from_report(report)
    components = _score_components(
        candidate=candidate,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        thresholds=thresholds,
    )
    status, reasons = _initial_status(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        components=components,
        thresholds=thresholds,
    )
    score = 0.0 if status == "fail" else _score_from_components(components)
    metrics_payload = {
        **_candidate_metrics_for_trace(metrics),
        "chunk_count_ratio": components["chunk_count_ratio"],
    }
    return {
        "name": candidate.name,
        "splitter_mode": candidate.splitter_mode,
        "chunk_size": candidate.chunk_size,
        "chunk_overlap": candidate.chunk_overlap,
        "too_short_chars": candidate.too_short_chars,
        "score": score,
        "status": status,
        "metrics": metrics_payload,
        "report_path": _path_label(report_path, project_root),
        "trace_path": None,
        "reasons": reasons,
    }


def _candidate_failure_result(
    *,
    candidate: ChunkPolicyCandidate,
    exc: Exception,
    report_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "splitter_mode": candidate.splitter_mode,
        "chunk_size": candidate.chunk_size,
        "chunk_overlap": candidate.chunk_overlap,
        "too_short_chars": candidate.too_short_chars,
        "score": 0.0,
        "status": "fail",
        "metrics": {
            "total_chunks": 0,
            "source_count": 0,
            "too_short_ratio": 0.0,
            "duplicate_ratio": 0.0,
            "required_metadata_coverage": 0.0,
            "section_metadata_coverage": 0.0,
            "chunk_count_ratio": None,
        },
        "report_path": _path_label(report_path, project_root),
        "trace_path": None,
        "reasons": ["candidate evaluation failed"],
        "error_type": type(exc).__name__,
        "error_message": sanitized_preview(str(exc), max_chars=ERROR_MESSAGE_CHARS),
    }


def _evaluate_candidate(
    *,
    candidate: ChunkPolicyCandidate,
    config: ChunkOptimizerConfig,
    baseline_metrics: dict[str, Any],
    trace: OptimizerTraceWriter,
    project_root: Path,
    run_id: str,
) -> dict[str, Any]:
    report_path = _candidate_report_path(candidate, config.output_dir, project_root)
    trace.write(
        "candidate_started",
        {
            "run_id": run_id,
            "candidate_name": candidate.name,
            "splitter_mode": candidate.splitter_mode,
            "chunk_size": candidate.chunk_size,
            "chunk_overlap": candidate.chunk_overlap,
        },
    )
    try:
        report = evaluate_mode(
            ChunkEvaluationConfig(
                mode=candidate.splitter_mode,
                data_dir=config.data_dir,
                output_dir=config.output_dir,
                output_path=report_path,
                subjects=config.subjects,
                too_short_chars=candidate.too_short_chars,
                chunk_size=candidate.chunk_size,
                chunk_overlap=candidate.chunk_overlap,
                sample_limit=config.sample_limit,
                trace_enabled=False,
                trace_output=None,
                run_id=run_id,
                project_root=project_root,
            )
        )
        result = _candidate_result_from_report(
            candidate=candidate,
            report=report,
            baseline_metrics=baseline_metrics,
            report_path=report_path,
            project_root=project_root,
            thresholds=config.thresholds,
        )
        trace.write(
            "candidate_finished",
            {
                "run_id": run_id,
                "candidate_name": candidate.name,
                "score": result["score"],
                "status": result["status"],
                **result["metrics"],
                "report_path": result["report_path"],
            },
        )
        return result
    except Exception as exc:
        result = _candidate_failure_result(
            candidate=candidate,
            exc=exc,
            report_path=report_path,
            project_root=project_root,
        )
        trace.write(
            "candidate_failed",
            {
                "run_id": run_id,
                "candidate_name": candidate.name,
                "error_type": result["error_type"],
                "error_message": result["error_message"],
            },
        )
        return result


def optimize_chunking(config: ChunkOptimizerConfig) -> dict[str, Any]:
    """Evaluate chunking policy candidates and write optimizer reports."""

    project_root = _project_root(config.project_root)
    run_id = config.run_id or uuid4().hex[:12]
    trace = OptimizerTraceWriter(
        enabled=config.trace_enabled,
        output_dir=config.output_dir,
        trace_output=config.trace_output,
        run_id=run_id,
        project_root=project_root,
    )
    if trace.enabled:
        trace.clear()

    try:
        candidates = generate_candidates(config)
        generated_at_utc = _utc_now()
        trace.write(
            "optimizer_started",
            {
                "run_id": run_id,
                "candidate_count": len(candidates),
                "baseline_candidate": BASELINE_CANDIDATE_NAME,
                "sampled": config.sample_limit is not None,
                "sample_limit": config.sample_limit,
            },
        )

        baseline = candidates[0]
        baseline_report_path = _candidate_report_path(
            baseline, config.output_dir, project_root
        )
        trace.write(
            "candidate_started",
            {
                "run_id": run_id,
                "candidate_name": baseline.name,
                "splitter_mode": baseline.splitter_mode,
                "chunk_size": baseline.chunk_size,
                "chunk_overlap": baseline.chunk_overlap,
            },
        )
        try:
            baseline_report = evaluate_mode(
                ChunkEvaluationConfig(
                    mode=baseline.splitter_mode,
                    data_dir=config.data_dir,
                    output_dir=config.output_dir,
                    output_path=baseline_report_path,
                    subjects=config.subjects,
                    too_short_chars=baseline.too_short_chars,
                    chunk_size=baseline.chunk_size,
                    chunk_overlap=baseline.chunk_overlap,
                    sample_limit=config.sample_limit,
                    trace_enabled=False,
                    trace_output=None,
                    run_id=run_id,
                    project_root=project_root,
                )
            )
            baseline_metrics = _metrics_from_report(baseline_report)
            baseline_result = _candidate_result_from_report(
                candidate=baseline,
                report=baseline_report,
                baseline_metrics=baseline_metrics,
                report_path=baseline_report_path,
                project_root=project_root,
                thresholds=config.thresholds,
            )
            trace.write(
                "candidate_finished",
                {
                    "run_id": run_id,
                    "candidate_name": baseline.name,
                    "score": baseline_result["score"],
                    "status": baseline_result["status"],
                    **baseline_result["metrics"],
                    "report_path": baseline_result["report_path"],
                },
            )
        except Exception as exc:
            baseline_metrics = {
                "total_chunks": 0,
                "source_count": 0,
                "too_short_ratio": 0.0,
                "duplicate_ratio": 0.0,
                "required_metadata_coverage": 0.0,
                "section_metadata_coverage": 0.0,
                "empty_chunk_count": 0,
                "too_long_ratio": 0.0,
            }
            baseline_result = _candidate_failure_result(
                candidate=baseline,
                exc=exc,
                report_path=baseline_report_path,
                project_root=project_root,
            )
            trace.write(
                "candidate_failed",
                {
                    "run_id": run_id,
                    "candidate_name": baseline.name,
                    "error_type": baseline_result["error_type"],
                    "error_message": baseline_result["error_message"],
                },
            )

        results = [baseline_result]
        for candidate in candidates[1:]:
            results.append(
                _evaluate_candidate(
                    candidate=candidate,
                    config=config,
                    baseline_metrics=baseline_metrics,
                    trace=trace,
                    project_root=project_root,
                    run_id=run_id,
                )
            )

        _apply_close_score_review(results, thresholds=config.thresholds)
        candidates_report, recommendation_report = _build_reports(
            candidates=results,
            config=config,
            trace=trace,
            project_root=project_root,
            generated_at_utc=generated_at_utc,
        )
        candidates_path = (
            project_root / config.output_dir / "chunk_optimizer_candidates.json"
        )
        if config.output_dir.is_absolute():
            candidates_path = config.output_dir / "chunk_optimizer_candidates.json"
        report_path = project_root / config.output_dir / "chunk_optimizer_report.json"
        if config.output_dir.is_absolute():
            report_path = config.output_dir / "chunk_optimizer_report.json"
        _write_json(candidates_report, candidates_path)
        _write_json(recommendation_report, report_path)

        trace.write(
            "recommendation_written",
            {
                "run_id": run_id,
                "best_candidate": recommendation_report["best_candidate"]["name"],
                "recommendation_action": recommendation_report["recommendation"][
                    "action"
                ],
                "report_path": _path_label(report_path, project_root),
            },
        )
        trace.write(
            "optimizer_finished",
            {
                "run_id": run_id,
                "candidate_count": len(results),
                "successful_candidates": sum(
                    1 for result in results if "error_type" not in result
                ),
                "failed_candidates": sum(
                    1 for result in results if result["status"] == "fail"
                ),
                "best_candidate": recommendation_report["best_candidate"]["name"],
                "recommendation_action": recommendation_report["recommendation"][
                    "action"
                ],
            },
        )
        return {
            "candidates_report": candidates_report,
            "recommendation_report": recommendation_report,
            "candidates_report_path": _path_label(candidates_path, project_root),
            "recommendation_report_path": _path_label(report_path, project_root),
            "trace_path": trace.path_label,
        }
    except Exception as exc:
        trace.write(
            "optimizer_failed",
            {
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error_message": sanitized_preview(
                    str(exc), max_chars=ERROR_MESSAGE_CHARS
                ),
            },
        )
        raise
