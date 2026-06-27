"""Subject-aware read-only splitter policy optimizer."""

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
from src.rag.eval.chunk_optimizer import (
    BASELINE_CANDIDATE_NAME,
    DEFAULT_TOO_SHORT_CHARS,
    ChunkOptimizerConfig,
    ChunkPolicyCandidate,
    generate_candidates,
)

DEFAULT_OUTPUT_DIR = Path("reports")
ERROR_MESSAGE_CHARS = 160
TOP_POLICY_LIMIT = 5
ADVISORY_WARNING = (
    "Subject-level policy recommendations are advisory only; do not apply mixed "
    "policy until retrieval-level validation is implemented."
)

DEFAULT_SUBJECT_POLICY_WEIGHTS = {
    "metadata": 0.20,
    "size": 0.20,
    "section": 0.20,
    "short_chunk_penalty": 0.15,
    "duplicate_penalty": 0.05,
    "chunk_count_penalty": 0.10,
    "source_safety": 0.10,
}

SCORE_COMPONENT_FIELDS = (
    "metadata_score",
    "size_score",
    "section_score",
    "short_chunk_penalty",
    "duplicate_penalty",
    "chunk_count_penalty",
    "source_safety_score",
)


@dataclass(frozen=True)
class SubjectPolicyThresholds:
    """Subject-level status, scoring, and confidence thresholds."""

    too_short_ratio_review_threshold: float = 0.10
    duplicate_ratio_review_threshold: float = 0.02
    chunk_count_review_ratio: float = 1.30
    high_cost_ratio: float = 1.20
    close_score_delta: float = 0.03
    minimum_consider_improvement: float = 0.05
    low_source_count_threshold: int = 2


@dataclass(frozen=True)
class SplitterPolicyOptimizerConfig:
    """Configuration for subject-aware splitter policy optimization."""

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
    thresholds: SubjectPolicyThresholds = SubjectPolicyThresholds()


class PolicyTraceWriter:
    """JSONL writer for policy-level events only."""

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
            else output_dir / f"splitter_policy_trace_{run_id}.jsonl"
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


def _candidate_report_path(
    candidate: ChunkPolicyCandidate,
    output_dir: Path,
    project_root: Path,
    sample_limit: int | None,
) -> Path:
    if sample_limit is None:
        filename = f"policy_eval_{candidate.name}.json"
    else:
        filename = f"policy_eval_sample_limit{sample_limit}_{candidate.name}.json"
    path = output_dir / filename
    return path if path.is_absolute() else project_root / path


def _report_path(output_dir: Path, filename: str, project_root: Path) -> Path:
    path = output_dir / filename
    return path if path.is_absolute() else project_root / path


def _policy_report_paths(
    output_dir: Path, project_root: Path, sample_limit: int | None
) -> dict[str, Path]:
    if sample_limit is None:
        return {
            "candidates": _report_path(
                output_dir, "splitter_policy_candidates.json", project_root
            ),
            "subject": _report_path(
                output_dir, "splitter_policy_subject_report.json", project_root
            ),
            "recommendation": _report_path(
                output_dir, "splitter_policy_recommendation.json", project_root
            ),
            "candidates_full": _report_path(
                output_dir, "splitter_policy_candidates_full.json", project_root
            ),
            "subject_full": _report_path(
                output_dir, "splitter_policy_subject_report_full.json", project_root
            ),
            "recommendation_full": _report_path(
                output_dir, "splitter_policy_recommendation_full.json", project_root
            ),
        }
    suffix = f"sample_limit{sample_limit}"
    return {
        "candidates": _report_path(
            output_dir, f"splitter_policy_candidates_{suffix}.json", project_root
        ),
        "subject": _report_path(
            output_dir, f"splitter_policy_subject_report_{suffix}.json", project_root
        ),
        "recommendation": _report_path(
            output_dir,
            f"splitter_policy_recommendation_{suffix}.json",
            project_root,
        ),
    }


def _candidate_config(config: SplitterPolicyOptimizerConfig) -> ChunkOptimizerConfig:
    return ChunkOptimizerConfig(
        data_dir=config.data_dir,
        output_dir=config.output_dir,
        modes=config.modes,
        chunk_sizes=config.chunk_sizes,
        overlaps=config.overlaps,
        too_short_chars=config.too_short_chars,
        sample_limit=config.sample_limit,
        subjects=config.subjects,
        max_candidates=config.max_candidates,
        trace_enabled=False,
        trace_output=None,
        run_id=config.run_id,
        project_root=config.project_root,
    )


def _subject_name(payload: dict[str, Any]) -> str:
    value = payload.get("subject")
    return value if isinstance(value, str) and value else "unknown"


def _global_metrics(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    metadata = (
        report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    )
    structure = (
        report.get("structure") if isinstance(report.get("structure"), dict) else {}
    )
    total_chunks = _integer(summary.get("total_chunks"))
    too_short_count = _integer(summary.get("too_short_count"))
    duplicate_count = _integer(summary.get("duplicate_chunk_count"))
    return {
        "chunk_count": total_chunks,
        "source_count": _integer(summary.get("source_count")),
        "too_short_count": too_short_count,
        "too_short_ratio": _number(summary.get("too_short_ratio")),
        "empty_chunk_count": _integer(summary.get("empty_chunk_count")),
        "duplicate_chunk_count": duplicate_count,
        "duplicate_ratio": _number(summary.get("duplicate_ratio")),
        "section_metadata_coverage": _number(metadata.get("section_metadata_coverage")),
        "unique_section_count": _integer(structure.get("unique_section_count")),
        "required_metadata_coverage": _number(
            metadata.get("required_metadata_coverage")
        ),
    }


def _subject_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    chunk_count = _integer(payload.get("chunk_count"))
    too_short_count = _integer(payload.get("too_short_count"))
    duplicate_count = _integer(payload.get("duplicate_chunk_count"))
    return {
        "chunk_count": chunk_count,
        "source_count": _integer(payload.get("source_count")),
        "too_short_count": too_short_count,
        "too_short_ratio": round(too_short_count / chunk_count, 4)
        if chunk_count
        else 0.0,
        "empty_chunk_count": _integer(payload.get("empty_chunk_count")),
        "duplicate_chunk_count": duplicate_count,
        "duplicate_ratio": round(duplicate_count / chunk_count, 4)
        if chunk_count
        else 0.0,
        "section_metadata_coverage": _number(payload.get("section_metadata_coverage")),
        "unique_section_count": _integer(payload.get("unique_section_count")),
        "required_metadata_coverage": payload.get("required_metadata_coverage"),
    }


def _subjects_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_subject = report.get("per_subject")
    if not isinstance(per_subject, list):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for item in per_subject:
        if isinstance(item, dict):
            output[_subject_name(item)] = _subject_metrics(item)
    return output


def _sanitize_error(exc: Exception) -> str:
    return sanitized_preview(str(exc), max_chars=ERROR_MESSAGE_CHARS)


def _components(
    *,
    candidate: ChunkPolicyCandidate,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    thresholds: SubjectPolicyThresholds,
) -> dict[str, float | None]:
    baseline_chunks = (
        float(baseline_metrics["chunk_count"]) if baseline_metrics else 0.0
    )
    chunk_count_ratio = _ratio(float(metrics["chunk_count"]), baseline_chunks)
    metadata_value = metrics.get("required_metadata_coverage")
    metadata_score = (
        _clamp(float(metadata_value))
        if isinstance(metadata_value, int | float)
        and not isinstance(metadata_value, bool)
        else 1.0
    )
    too_short_ratio = float(metrics["too_short_ratio"])
    duplicate_ratio = float(metrics["duplicate_ratio"])
    size_score = _clamp(
        1.0 - _clamp(too_short_ratio / thresholds.too_short_ratio_review_threshold)
    )
    section_score = (
        _clamp(float(metrics["section_metadata_coverage"]))
        if candidate.splitter_mode == "structure"
        else 0.0
    )
    short_penalty = _clamp(
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
    source_safety_score = 1.0
    if baseline_metrics and int(metrics["source_count"]) < int(
        baseline_metrics["source_count"]
    ):
        source_safety_score = 0.0
    return {
        "metadata_score": metadata_score,
        "size_score": size_score,
        "section_score": section_score,
        "short_chunk_penalty": short_penalty,
        "duplicate_penalty": duplicate_penalty,
        "chunk_count_penalty": chunk_count_penalty,
        "source_safety_score": source_safety_score,
        "chunk_count_ratio": chunk_count_ratio,
    }


def _score_components_payload(
    components: dict[str, float | None] | None,
) -> dict[str, float]:
    if components is None:
        return {field: 0.0 for field in SCORE_COMPONENT_FIELDS}
    return {
        field: round(_number(components.get(field)), 4)
        for field in SCORE_COMPONENT_FIELDS
    }


def _score(components: dict[str, float | None]) -> float:
    return round(
        _clamp(
            DEFAULT_SUBJECT_POLICY_WEIGHTS["metadata"]
            * float(components["metadata_score"])
            + DEFAULT_SUBJECT_POLICY_WEIGHTS["size"] * float(components["size_score"])
            + DEFAULT_SUBJECT_POLICY_WEIGHTS["section"]
            * float(components["section_score"])
            + DEFAULT_SUBJECT_POLICY_WEIGHTS["short_chunk_penalty"]
            * (1.0 - float(components["short_chunk_penalty"]))
            + DEFAULT_SUBJECT_POLICY_WEIGHTS["duplicate_penalty"]
            * (1.0 - float(components["duplicate_penalty"]))
            + DEFAULT_SUBJECT_POLICY_WEIGHTS["chunk_count_penalty"]
            * (1.0 - float(components["chunk_count_penalty"]))
            + DEFAULT_SUBJECT_POLICY_WEIGHTS["source_safety"]
            * float(components["source_safety_score"])
        ),
        4,
    )


def _status(
    *,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    components: dict[str, float | None],
    thresholds: SubjectPolicyThresholds,
) -> tuple[str, list[str]]:
    if baseline_metrics is None:
        return "needs_review", ["subject missing from baseline report"]

    fail_reasons: list[str] = []
    if int(metrics["chunk_count"]) == 0:
        fail_reasons.append("chunk_count is 0")
    if int(metrics["source_count"]) < int(baseline_metrics["source_count"]):
        fail_reasons.append("source count below subject baseline")
    if int(metrics["empty_chunk_count"]) > 0:
        fail_reasons.append("empty chunks detected")
    metadata_coverage = metrics.get("required_metadata_coverage")
    if (
        isinstance(metadata_coverage, int | float)
        and not isinstance(metadata_coverage, bool)
        and float(metadata_coverage) < 1.0
    ):
        fail_reasons.append("required metadata coverage below 1.0")
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


def _confidence(
    *,
    status: str,
    score: float,
    baseline_score: float | None,
    metrics: dict[str, Any],
    chunk_count_ratio: float | None,
    thresholds: SubjectPolicyThresholds,
) -> str:
    if status != "pass" or baseline_score is None:
        return "low"
    improvement = score - baseline_score
    if abs(improvement) < thresholds.close_score_delta:
        return "low"
    if (
        improvement >= thresholds.minimum_consider_improvement
        and int(metrics["source_count"]) >= thresholds.low_source_count_threshold
        and chunk_count_ratio is not None
        and chunk_count_ratio <= thresholds.high_cost_ratio
        and float(metrics["duplicate_ratio"])
        <= thresholds.duplicate_ratio_review_threshold
        and int(metrics["empty_chunk_count"]) == 0
    ):
        return "high"
    return "medium"


def _rank_key(item: dict[str, Any]) -> tuple[int, float, int, str]:
    status_rank = {"pass": 0, "needs_review": 1, "fail": 2}
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    return (
        status_rank.get(str(item.get("status")), 3),
        -float(item.get("score") or 0.0),
        _integer(metrics.get("chunk_count")),
        str(item.get("policy_name") or ""),
    )


def _global_rank_key(item: dict[str, Any]) -> tuple[int, float, int, str]:
    status_rank = {"pass": 0, "needs_review": 1, "fail": 2}
    metrics = (
        item.get("global_metrics")
        if isinstance(item.get("global_metrics"), dict)
        else {}
    )
    return (
        status_rank.get(str(item.get("global_status")), 3),
        -float(item.get("global_score") or 0.0),
        _integer(metrics.get("chunk_count")),
        str(item.get("policy_name") or ""),
    )


def _recommended_action(
    *,
    baseline: dict[str, Any] | None,
    best: dict[str, Any],
    sampled: bool,
    thresholds: SubjectPolicyThresholds,
) -> tuple[str, str, str | None]:
    if baseline is None:
        return "needs_manual_review", "subject missing from baseline report", None
    if sampled:
        if best["policy_name"] == BASELINE_CANDIDATE_NAME:
            return "keep_current_default", "sampled run keeps default policy", None
        return (
            "needs_manual_review",
            "sampled policy optimizer run cannot recommend candidate adoption",
            None,
        )
    if baseline["status"] == "fail":
        return "needs_manual_review", "baseline policy failed subject checks", None
    if best["policy_name"] == BASELINE_CANDIDATE_NAME:
        return "keep_current_default", "baseline remains best policy", None
    if best["status"] != "pass":
        return "needs_manual_review", "best policy has review-level risks", None
    improvement = float(best["score"]) - float(baseline["score"])
    chunk_count_ratio = best["metrics"].get("chunk_count_ratio")
    if (
        improvement >= thresholds.minimum_consider_improvement
        and best.get("confidence") != "low"
        and chunk_count_ratio is not None
        and float(chunk_count_ratio) <= thresholds.high_cost_ratio
    ):
        return (
            "consider_candidate",
            "best policy improves subject score without high chunk cost",
            str(best["policy_name"]),
        )
    if improvement > 0:
        return (
            "needs_manual_review",
            "best policy improves score but confidence is not sufficient",
            None,
        )
    return "keep_current_default", "candidate improvement is insufficient", None


def _candidate_metadata(candidate: ChunkPolicyCandidate) -> dict[str, Any]:
    return {
        "policy_name": candidate.name,
        "splitter_mode": candidate.splitter_mode,
        "chunk_size": candidate.chunk_size,
        "chunk_overlap": candidate.chunk_overlap,
    }


def _global_result(
    *,
    candidate: ChunkPolicyCandidate,
    report: dict[str, Any] | None,
    baseline_metrics: dict[str, Any] | None,
    thresholds: SubjectPolicyThresholds,
    report_path: Path,
    project_root: Path,
    exc: Exception | None = None,
) -> dict[str, Any]:
    if exc is not None or report is None:
        payload = {
            **_candidate_metadata(candidate),
            "global_score": 0.0,
            "global_status": "fail",
            "global_score_components": _score_components_payload(None),
            "global_metrics": {
                "chunk_count": 0,
                "source_count": 0,
                "too_short_ratio": 0.0,
                "duplicate_ratio": 0.0,
                "chunk_count_ratio": None,
            },
            "report_path": _path_label(report_path, project_root),
            "reasons": ["candidate evaluation failed"],
        }
        if exc is not None:
            payload["error_type"] = type(exc).__name__
            payload["error_message"] = _sanitize_error(exc)
        return payload

    metrics = _global_metrics(report)
    components = _components(
        candidate=candidate,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        thresholds=thresholds,
    )
    status, reasons = _status(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        components=components,
        thresholds=thresholds,
    )
    score = 0.0 if status == "fail" else _score(components)
    score_components = (
        _score_components_payload(None)
        if status == "fail"
        else _score_components_payload(components)
    )
    metrics_payload = {
        **metrics,
        "chunk_count_ratio": components["chunk_count_ratio"],
    }
    return {
        **_candidate_metadata(candidate),
        "global_score": score,
        "global_status": status,
        "global_score_components": score_components,
        "global_metrics": metrics_payload,
        "report_path": _path_label(report_path, project_root),
        "reasons": reasons,
    }


def _subject_result(
    *,
    candidate: ChunkPolicyCandidate,
    subject: str,
    metrics: dict[str, Any] | None,
    baseline_metrics: dict[str, Any] | None,
    baseline_score: float | None,
    thresholds: SubjectPolicyThresholds,
) -> dict[str, Any]:
    if metrics is None:
        missing_metrics = {
            "chunk_count": 0,
            "source_count": 0,
            "too_short_count": 0,
            "too_short_ratio": 0.0,
            "empty_chunk_count": 0,
            "duplicate_chunk_count": 0,
            "duplicate_ratio": 0.0,
            "section_metadata_coverage": 0.0,
            "unique_section_count": 0,
            "chunk_count_ratio": None,
        }
        return {
            "policy_name": candidate.name,
            "score": 0.0,
            "status": "fail",
            "confidence": "low",
            "score_components": _score_components_payload(None),
            "metrics": missing_metrics,
            "reasons": ["subject missing from candidate report"],
        }

    components = _components(
        candidate=candidate,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        thresholds=thresholds,
    )
    status, reasons = _status(
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        components=components,
        thresholds=thresholds,
    )
    score = 0.0 if status == "fail" else _score(components)
    score_components = (
        _score_components_payload(None)
        if status == "fail"
        else _score_components_payload(components)
    )
    metrics_payload = {
        **metrics,
        "chunk_count_ratio": components["chunk_count_ratio"],
    }
    confidence = _confidence(
        status=status,
        score=score,
        baseline_score=baseline_score,
        metrics=metrics_payload,
        chunk_count_ratio=components["chunk_count_ratio"],
        thresholds=thresholds,
    )
    return {
        "policy_name": candidate.name,
        "score": score,
        "status": status,
        "confidence": confidence,
        "score_components": score_components,
        "metrics": metrics_payload,
        "reasons": reasons,
    }


def _subject_recommendation(
    *,
    baseline: dict[str, Any] | None,
    ranking: list[dict[str, Any]],
    sampled: bool,
    thresholds: SubjectPolicyThresholds,
) -> dict[str, Any]:
    best = sorted(ranking, key=_rank_key)[0]
    action, reason, recommended_policy = _recommended_action(
        baseline=baseline,
        best=best,
        sampled=sampled,
        thresholds=thresholds,
    )
    confidence = "low" if action == "needs_manual_review" else str(best["confidence"])
    return {
        "action": action,
        "recommended_policy": recommended_policy,
        "reason": reason,
        "confidence": confidence,
        "do_not_auto_apply": True,
    }


def _global_recommendation(
    *,
    baseline: dict[str, Any],
    ranking: list[dict[str, Any]],
    sampled: bool,
    thresholds: SubjectPolicyThresholds,
) -> dict[str, Any]:
    best = sorted(ranking, key=_global_rank_key)[0]
    if sampled:
        action = (
            "keep_current_default"
            if best["policy_name"] == BASELINE_CANDIDATE_NAME
            else "needs_manual_review"
        )
        reason = "sampled run cannot recommend candidate adoption"
    elif baseline["global_status"] == "fail":
        action = "needs_manual_review"
        reason = "baseline policy failed global checks"
    elif best["policy_name"] == BASELINE_CANDIDATE_NAME:
        action = "keep_current_default"
        reason = "baseline remains best global policy"
    elif best["global_status"] != "pass":
        action = "needs_manual_review"
        reason = "best global policy has review-level risks"
    elif float(best["global_score"]) - float(
        baseline["global_score"]
    ) >= thresholds.minimum_consider_improvement and (
        best["global_metrics"].get("chunk_count_ratio") is not None
        and float(best["global_metrics"]["chunk_count_ratio"])
        <= thresholds.high_cost_ratio
    ):
        action = "consider_candidate"
        reason = "best global policy improves score without high chunk cost"
    elif float(best["global_score"]) > float(baseline["global_score"]):
        action = "needs_manual_review"
        reason = "best global policy improves score but needs review"
    else:
        action = "keep_current_default"
        reason = "candidate improvement is insufficient"
    return {
        "action": action,
        "reason": reason,
        "recommended_policy": None
        if action != "consider_candidate"
        else best["policy_name"],
        "do_not_auto_apply": True,
    }


def _evaluate_candidate(
    *,
    candidate: ChunkPolicyCandidate,
    config: SplitterPolicyOptimizerConfig,
    project_root: Path,
    run_id: str,
    trace: PolicyTraceWriter,
    candidate_count: int,
) -> tuple[dict[str, Any] | None, Path, Exception | None]:
    report_path = _candidate_report_path(
        candidate, config.output_dir, project_root, config.sample_limit
    )
    trace.write(
        "policy_candidate_started",
        {
            "run_id": run_id,
            "policy_name": candidate.name,
            "splitter_mode": candidate.splitter_mode,
            "chunk_size": candidate.chunk_size,
            "chunk_overlap": candidate.chunk_overlap,
            "sampled": config.sample_limit is not None,
            "sample_limit": config.sample_limit,
            "candidate_count": candidate_count,
            "subject_count": 0,
            "baseline_policy": BASELINE_CANDIDATE_NAME,
            "global_best_policy": None,
            "global_action": None,
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
        return report, report_path, None
    except Exception as exc:
        trace.write(
            "policy_candidate_failed",
            {
                "run_id": run_id,
                "policy_name": candidate.name,
                "sampled": config.sample_limit is not None,
                "sample_limit": config.sample_limit,
                "candidate_count": candidate_count,
                "subject_count": 0,
                "baseline_policy": BASELINE_CANDIDATE_NAME,
                "global_best_policy": None,
                "global_action": None,
                "error_type": type(exc).__name__,
                "error_message": _sanitize_error(exc),
            },
        )
        return None, report_path, exc


def _build_subject_outputs(
    *,
    candidates: list[ChunkPolicyCandidate],
    reports_by_policy: dict[str, dict[str, Any]],
    candidate_global: dict[str, dict[str, Any]],
    config: SplitterPolicyOptimizerConfig,
    trace: PolicyTraceWriter,
    run_id: str,
) -> tuple[list[str], dict[str, Any], dict[str, dict[str, Any]]]:
    subjects: set[str] = set()
    subject_metrics_by_policy: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in candidates:
        report = reports_by_policy.get(candidate.name)
        subject_metrics = _subjects_by_name(report) if report is not None else {}
        subject_metrics_by_policy[candidate.name] = subject_metrics
        subjects.update(subject_metrics)

    ordered_subjects = sorted(subjects)
    baseline_subjects = subject_metrics_by_policy.get(BASELINE_CANDIDATE_NAME, {})
    baseline_scores: dict[str, float] = {}
    subject_report: dict[str, Any] = {}
    subject_scores_for_candidates: dict[str, dict[str, Any]] = {
        candidate.name: {} for candidate in candidates
    }

    for subject in ordered_subjects:
        baseline_metrics = baseline_subjects.get(subject)
        baseline_candidate = next(
            candidate
            for candidate in candidates
            if candidate.name == BASELINE_CANDIDATE_NAME
        )
        baseline_result = _subject_result(
            candidate=baseline_candidate,
            subject=subject,
            metrics=baseline_metrics,
            baseline_metrics=baseline_metrics,
            baseline_score=None,
            thresholds=config.thresholds,
        )
        baseline_scores[subject] = float(baseline_result["score"])

    for subject in ordered_subjects:
        ranking: list[dict[str, Any]] = []
        baseline_metrics = baseline_subjects.get(subject)
        baseline_score = baseline_scores.get(subject)
        for candidate in candidates:
            metrics = subject_metrics_by_policy[candidate.name].get(subject)
            result = _subject_result(
                candidate=candidate,
                subject=subject,
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                baseline_score=baseline_score,
                thresholds=config.thresholds,
            )
            ranking.append(result)
            subject_scores_for_candidates[candidate.name][subject] = {
                "score": result["score"],
                "status": result["status"],
                "confidence": result["confidence"],
                "score_components": result["score_components"],
                "metrics": result["metrics"],
                "reasons": result["reasons"],
            }
            trace.write(
                "subject_scored",
                {
                    "run_id": run_id,
                    "subject": subject,
                    "policy_name": candidate.name,
                    "score": result["score"],
                    "status": result["status"],
                    "confidence": result["confidence"],
                    "chunk_count": result["metrics"]["chunk_count"],
                    "source_count": result["metrics"]["source_count"],
                    "too_short_ratio": result["metrics"]["too_short_ratio"],
                    "duplicate_ratio": result["metrics"]["duplicate_ratio"],
                    "chunk_count_ratio": result["metrics"]["chunk_count_ratio"],
                    "sampled": config.sample_limit is not None,
                    "sample_limit": config.sample_limit,
                    "candidate_count": len(candidates),
                    "subject_count": len(ordered_subjects),
                    "baseline_policy": BASELINE_CANDIDATE_NAME,
                    "global_best_policy": None,
                    "global_action": None,
                },
            )

        sorted_ranking = sorted(ranking, key=_rank_key)
        baseline_entry = next(
            (
                item
                for item in ranking
                if item["policy_name"] == BASELINE_CANDIDATE_NAME
            ),
            None,
        )
        recommendation = _subject_recommendation(
            baseline=baseline_entry if baseline_metrics is not None else None,
            ranking=ranking,
            sampled=config.sample_limit is not None,
            thresholds=config.thresholds,
        )
        best_policy = sorted_ranking[0]["policy_name"] if sorted_ranking else None
        subject_report[subject] = {
            "source_count": baseline_metrics.get("source_count", 0)
            if baseline_metrics
            else 0,
            "baseline_policy": BASELINE_CANDIDATE_NAME,
            "best_policy": best_policy,
            "ranking": sorted_ranking,
            "recommendation": recommendation,
        }
        trace.write(
            "subject_recommendation_written",
            {
                "run_id": run_id,
                "subject": subject,
                "recommended_policy": recommendation["recommended_policy"],
                "action": recommendation["action"],
                "confidence": recommendation["confidence"],
                "sampled": config.sample_limit is not None,
                "sample_limit": config.sample_limit,
                "candidate_count": len(candidates),
                "subject_count": len(ordered_subjects),
                "baseline_policy": BASELINE_CANDIDATE_NAME,
                "global_best_policy": None,
                "global_action": None,
            },
        )

    return ordered_subjects, subject_report, subject_scores_for_candidates


def _global_ranking(
    candidates: list[ChunkPolicyCandidate],
    candidate_global: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for candidate in candidates:
        result = candidate_global[candidate.name]
        ranking.append(
            {
                "policy_name": candidate.name,
                "splitter_mode": candidate.splitter_mode,
                "chunk_size": candidate.chunk_size,
                "chunk_overlap": candidate.chunk_overlap,
                "score": result["global_score"],
                "status": result["global_status"],
                "score_components": result["global_score_components"],
                "metrics": result["global_metrics"],
                "reasons": result["reasons"],
                "report_path": result["report_path"],
            }
        )
    return sorted(ranking, key=_rank_key)


def _candidate_report_entry(
    *,
    candidate: ChunkPolicyCandidate,
    global_result: dict[str, Any],
    subject_scores: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        **_candidate_metadata(candidate),
        "global_score": global_result["global_score"],
        "global_status": global_result["global_status"],
        "global_score_components": global_result["global_score_components"],
        "global_metrics": global_result["global_metrics"],
        "subject_scores": subject_scores,
        "report_path": global_result["report_path"],
        "reasons": global_result["reasons"],
    }
    if "error_type" in global_result:
        payload["error_type"] = global_result["error_type"]
        payload["error_message"] = global_result["error_message"]
    return payload


def _build_reports(
    *,
    config: SplitterPolicyOptimizerConfig,
    project_root: Path,
    generated_at_utc: str,
    trace: PolicyTraceWriter,
    candidates: list[ChunkPolicyCandidate],
    candidate_global: dict[str, dict[str, Any]],
    subjects: list[str],
    subject_report: dict[str, Any],
    subject_scores_for_candidates: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    global_ranking = _global_ranking(candidates, candidate_global)
    baseline_global = candidate_global[BASELINE_CANDIDATE_NAME]
    global_recommendation = _global_recommendation(
        baseline=baseline_global,
        ranking=[
            {
                "policy_name": item["policy_name"],
                "global_score": item["score"],
                "global_status": item["status"],
                "global_metrics": item["metrics"],
            }
            for item in global_ranking
        ],
        sampled=config.sample_limit is not None,
        thresholds=config.thresholds,
    )
    global_best_policy = global_ranking[0]["policy_name"] if global_ranking else None
    global_best = global_ranking[0] if global_ranking else {}
    candidates_payload = [
        _candidate_report_entry(
            candidate=candidate,
            global_result=candidate_global[candidate.name],
            subject_scores=subject_scores_for_candidates[candidate.name],
        )
        for candidate in candidates
    ]

    candidates_report = {
        "generated_at_utc": generated_at_utc,
        "data_dir": _path_label(
            config.data_dir
            if config.data_dir.is_absolute()
            else project_root / config.data_dir,
            project_root,
        ),
        "sampled": config.sample_limit is not None,
        "sample_limit": config.sample_limit,
        "trace_enabled": trace.enabled,
        "trace_path": trace.path_label,
        "baseline_policy": BASELINE_CANDIDATE_NAME,
        "candidate_count": len(candidates),
        "subjects": subjects,
        "global_ranking": global_ranking,
        "candidates": candidates_payload,
    }
    subject_payload = {
        "generated_at_utc": generated_at_utc,
        "sampled": config.sample_limit is not None,
        "sample_limit": config.sample_limit,
        "baseline_policy": BASELINE_CANDIDATE_NAME,
        "candidate_count": len(candidates),
        "subject_count": len(subjects),
        "global_best_policy": global_best_policy,
        "global_action": global_recommendation["action"],
        "subjects": subject_report,
    }
    recommendation_payload = {
        "generated_at_utc": generated_at_utc,
        "sampled": config.sample_limit is not None,
        "sample_limit": config.sample_limit,
        "default_policy": BASELINE_CANDIDATE_NAME,
        "baseline_policy": BASELINE_CANDIDATE_NAME,
        "candidate_count": len(candidates),
        "subject_count": len(subjects),
        "global_best_policy": global_best_policy,
        "global_best_score": global_best.get("score", 0.0),
        "global_best_score_components": global_best.get(
            "score_components", _score_components_payload(None)
        ),
        "global_action": global_recommendation["action"],
        "global_recommendation": global_recommendation,
        "subject_policy_map": {
            subject: {
                "recommended_policy": payload["recommendation"]["recommended_policy"],
                "action": payload["recommendation"]["action"],
                "confidence": payload["recommendation"]["confidence"],
                "reason": payload["recommendation"]["reason"],
            }
            for subject, payload in subject_report.items()
        },
        "do_not_auto_apply": True,
        "warnings": [ADVISORY_WARNING],
    }
    return candidates_report, subject_payload, recommendation_payload


def optimize_splitter_policy(config: SplitterPolicyOptimizerConfig) -> dict[str, Any]:
    """Evaluate splitter policies and write subject-aware advisory reports."""

    project_root = _project_root(config.project_root)
    run_id = config.run_id or uuid4().hex[:12]
    trace = PolicyTraceWriter(
        enabled=config.trace_enabled,
        output_dir=config.output_dir,
        trace_output=config.trace_output,
        run_id=run_id,
        project_root=project_root,
    )
    if trace.enabled:
        trace.clear()

    try:
        candidates = generate_candidates(_candidate_config(config))
        generated_at_utc = _utc_now()
        trace.write(
            "policy_optimizer_started",
            {
                "run_id": run_id,
                "sampled": config.sample_limit is not None,
                "sample_limit": config.sample_limit,
                "candidate_count": len(candidates),
                "subject_count": 0,
                "baseline_policy": BASELINE_CANDIDATE_NAME,
                "global_best_policy": None,
                "global_action": None,
            },
        )

        reports_by_policy: dict[str, dict[str, Any]] = {}
        candidate_global: dict[str, dict[str, Any]] = {}
        baseline_metrics: dict[str, Any] | None = None

        for candidate in candidates:
            report, report_path, exc = _evaluate_candidate(
                candidate=candidate,
                config=config,
                project_root=project_root,
                run_id=run_id,
                trace=trace,
                candidate_count=len(candidates),
            )
            if candidate.name == BASELINE_CANDIDATE_NAME and report is not None:
                baseline_metrics = _global_metrics(report)
            if report is not None:
                reports_by_policy[candidate.name] = report
            global_result = _global_result(
                candidate=candidate,
                report=report,
                baseline_metrics=baseline_metrics,
                thresholds=config.thresholds,
                report_path=report_path,
                project_root=project_root,
                exc=exc,
            )
            candidate_global[candidate.name] = global_result
            trace.write(
                "policy_candidate_finished",
                {
                    "run_id": run_id,
                    "policy_name": candidate.name,
                    "global_score": global_result["global_score"],
                    "global_status": global_result["global_status"],
                    "subject_count": len(_subjects_by_name(report)) if report else 0,
                    "report_path": global_result["report_path"],
                    "sampled": config.sample_limit is not None,
                    "sample_limit": config.sample_limit,
                    "candidate_count": len(candidates),
                    "baseline_policy": BASELINE_CANDIDATE_NAME,
                    "global_best_policy": None,
                    "global_action": None,
                },
            )

        subjects, subject_report, subject_scores = _build_subject_outputs(
            candidates=candidates,
            reports_by_policy=reports_by_policy,
            candidate_global=candidate_global,
            config=config,
            trace=trace,
            run_id=run_id,
        )
        candidates_report, subject_payload, recommendation_payload = _build_reports(
            config=config,
            project_root=project_root,
            generated_at_utc=generated_at_utc,
            trace=trace,
            candidates=candidates,
            candidate_global=candidate_global,
            subjects=subjects,
            subject_report=subject_report,
            subject_scores_for_candidates=subject_scores,
        )

        report_paths = _policy_report_paths(
            config.output_dir, project_root, config.sample_limit
        )
        candidates_path = report_paths["candidates"]
        subject_path = report_paths["subject"]
        recommendation_path = report_paths["recommendation"]
        _write_json(candidates_report, candidates_path)
        _write_json(subject_payload, subject_path)
        _write_json(recommendation_payload, recommendation_path)
        if config.sample_limit is None:
            _write_json(candidates_report, report_paths["candidates_full"])
            _write_json(subject_payload, report_paths["subject_full"])
            _write_json(recommendation_payload, report_paths["recommendation_full"])

        global_best_policy = recommendation_payload["global_best_policy"]
        global_action = recommendation_payload["global_action"]
        trace.write(
            "policy_recommendation_written",
            {
                "run_id": run_id,
                "sampled": config.sample_limit is not None,
                "sample_limit": config.sample_limit,
                "candidate_count": len(candidates),
                "subject_count": len(subjects),
                "baseline_policy": BASELINE_CANDIDATE_NAME,
                "global_best_policy": global_best_policy,
                "global_action": global_action,
                "report_path": _path_label(recommendation_path, project_root),
            },
        )
        trace.write(
            "policy_optimizer_finished",
            {
                "run_id": run_id,
                "sampled": config.sample_limit is not None,
                "sample_limit": config.sample_limit,
                "candidate_count": len(candidates),
                "successful_candidates": sum(
                    1
                    for result in candidate_global.values()
                    if "error_type" not in result
                ),
                "failed_candidates": sum(
                    1
                    for result in candidate_global.values()
                    if result["global_status"] == "fail"
                ),
                "subject_count": len(subjects),
                "baseline_policy": BASELINE_CANDIDATE_NAME,
                "global_best_policy": global_best_policy,
                "global_action": global_action,
                "candidates_report_path": _path_label(candidates_path, project_root),
                "subject_report_path": _path_label(subject_path, project_root),
                "recommendation_report_path": _path_label(
                    recommendation_path, project_root
                ),
            },
        )
        result = {
            "candidates_report": candidates_report,
            "subject_report": subject_payload,
            "recommendation_report": recommendation_payload,
            "candidates_report_path": _path_label(candidates_path, project_root),
            "subject_report_path": _path_label(subject_path, project_root),
            "recommendation_report_path": _path_label(
                recommendation_path, project_root
            ),
            "trace_path": trace.path_label,
        }
        if config.sample_limit is None:
            result["candidates_full_report_path"] = _path_label(
                report_paths["candidates_full"], project_root
            )
            result["subject_full_report_path"] = _path_label(
                report_paths["subject_full"], project_root
            )
            result["recommendation_full_report_path"] = _path_label(
                report_paths["recommendation_full"], project_root
            )
        return result
    except Exception as exc:
        trace.write(
            "policy_optimizer_failed",
            {
                "run_id": run_id,
                "sampled": config.sample_limit is not None,
                "sample_limit": config.sample_limit,
                "candidate_count": 0,
                "subject_count": 0,
                "baseline_policy": BASELINE_CANDIDATE_NAME,
                "global_best_policy": None,
                "global_action": "needs_manual_review",
                "error_type": type(exc).__name__,
                "error_message": _sanitize_error(exc),
            },
        )
        raise
