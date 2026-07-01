from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import pytest

from scripts import optimize_splitter_policy as policy_script
from src.rag.eval import policy_optimizer
from src.rag.eval.chunk_optimizer import BASELINE_CANDIDATE_NAME
from src.rag.eval.policy_optimizer import (
    ADVISORY_WARNING,
    SplitterPolicyOptimizerConfig,
    optimize_splitter_policy,
)

ALLOWED_SCORE_COMPONENTS = {
    "metadata_score",
    "size_score",
    "section_score",
    "short_chunk_penalty",
    "duplicate_penalty",
    "chunk_count_penalty",
    "source_safety_score",
}
DISALLOWED_SCORE_COMPONENTS = {
    "chunk_count_ratio",
    "source_count",
    "chunk_count",
    "subject",
    "policy_name",
}


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def _subject(
    name: str,
    *,
    chunk_count: int = 100,
    source_count: int = 2,
    too_short_count: int = 0,
    empty_chunk_count: int = 0,
    duplicate_chunk_count: int = 0,
    section_metadata_coverage: float = 0.0,
    unique_section_count: int = 0,
) -> dict:
    return {
        "subject": name,
        "chunk_count": chunk_count,
        "source_count": source_count,
        "min_chars": 80,
        "max_chars": 1000,
        "avg_chars": 500.0,
        "median_chars": 500,
        "p10_chars": 120,
        "p90_chars": 900,
        "too_short_count": too_short_count,
        "too_long_count": 0,
        "empty_chunk_count": empty_chunk_count,
        "duplicate_chunk_count": duplicate_chunk_count,
        "section_metadata_coverage": section_metadata_coverage,
        "unique_section_count": unique_section_count,
    }


def _report(*, mode: str = "recursive", subjects: list[dict] | None = None) -> dict:
    subject_rows = subjects if subjects is not None else [_subject("alpha")]
    total_chunks = sum(int(row["chunk_count"]) for row in subject_rows)
    source_count = sum(int(row["source_count"]) for row in subject_rows)
    too_short_count = sum(int(row["too_short_count"]) for row in subject_rows)
    duplicate_count = sum(int(row["duplicate_chunk_count"]) for row in subject_rows)
    empty_count = sum(int(row["empty_chunk_count"]) for row in subject_rows)
    section_coverage = (
        sum(float(row["section_metadata_coverage"]) for row in subject_rows)
        / len(subject_rows)
        if subject_rows
        else 0.0
    )
    return {
        "mode": mode,
        "sampled": False,
        "sample_limit": None,
        "trace_enabled": False,
        "trace_path": None,
        "summary": {
            "total_chunks": total_chunks,
            "source_count": source_count,
            "too_short_count": too_short_count,
            "too_short_ratio": round(too_short_count / total_chunks, 4)
            if total_chunks
            else 0.0,
            "too_long_count": 0,
            "empty_chunk_count": empty_count,
            "duplicate_chunk_count": duplicate_count,
            "duplicate_ratio": round(duplicate_count / total_chunks, 4)
            if total_chunks
            else 0.0,
            "short_chunk_samples": [
                {"preview": "COMPLETE CHUNK TEXT THAT MUST NOT BE COPIED"}
            ],
        },
        "metadata": {
            "required_metadata_coverage": 1.0,
            "section_metadata_coverage": section_coverage,
            "missing_metadata_counts": {},
        },
        "structure": {
            "unique_section_count": sum(
                int(row["unique_section_count"]) for row in subject_rows
            )
        },
        "per_subject": subject_rows,
        "per_source": [],
        "warnings": [],
    }


def _install_fake_evaluator(monkeypatch, reports_by_name: dict[str, dict | Exception]):
    calls = []

    def fake_evaluate_mode(config):
        assert config.trace_enabled is False
        assert config.trace_output is None
        assert config.output_path is not None
        name = _candidate_name_from_report_path(config.output_path)
        calls.append(
            {
                "name": name,
                "mode": config.mode,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
        )
        payload = reports_by_name.get(name)
        if isinstance(payload, Exception):
            raise payload
        report = dict(payload or _report(mode=config.mode))
        report["sampled"] = config.sample_limit is not None
        report["sample_limit"] = config.sample_limit
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(policy_optimizer, "evaluate_mode", fake_evaluate_mode)
    return calls


def _candidate_name_from_report_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("policy_eval_sample_limit"):
        sample_part = stem.removeprefix("policy_eval_sample_limit")
        return sample_part.split("_", 1)[1]
    return stem.removeprefix("policy_eval_")


def _assert_score_components(payload: dict) -> None:
    assert set(payload) == ALLOWED_SCORE_COMPONENTS
    assert not (set(payload) & DISALLOWED_SCORE_COMPONENTS)
    for value in payload.values():
        assert isinstance(value, int | float)
        assert not isinstance(value, bool)
        assert value == round(value, 4)


def test_cli_trace_output_requires_trace_and_max_candidates_is_positive(
    local_tmp_path,
):
    with pytest.raises(SystemExit):
        policy_script.parse_args(
            ["--trace-output", str(local_tmp_path / "policy_trace.jsonl")]
        )
    with pytest.raises(SystemExit):
        policy_script.parse_args(["--max-candidates", "0"])


def test_subjects_are_discovered_from_reports_and_subject_ratios_use_baseline(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(
            subjects=[
                _subject("alpha", chunk_count=100),
                _subject("beta", chunk_count=50),
            ]
        ),
        "recursive_size700_overlap100": _report(
            subjects=[
                _subject("alpha", chunk_count=125),
                _subject("beta", chunk_count=50),
            ]
        ),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )

    subject_report = result["subject_report"]["subjects"]
    alpha_ranking = {
        item["policy_name"]: item for item in subject_report["alpha"]["ranking"]
    }

    assert result["candidates_report"]["subjects"] == ["alpha", "beta"]
    assert (
        alpha_ranking["recursive_size700_overlap100"]["metrics"]["chunk_count_ratio"]
        == 1.25
    )


def test_score_components_are_explainable_and_match_scoring_inputs(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(
            mode="recursive",
            subjects=[_subject("alpha", section_metadata_coverage=0.0)],
        ),
        "structure_size700_overlap100": _report(
            mode="structure",
            subjects=[
                _subject(
                    "alpha",
                    section_metadata_coverage=1.0,
                    unique_section_count=5,
                )
            ],
        ),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            modes=("structure",),
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )

    candidates = {
        item["policy_name"]: item for item in result["candidates_report"]["candidates"]
    }
    structure_candidate = candidates["structure_size700_overlap100"]
    subject_score = structure_candidate["subject_scores"]["alpha"]
    subject_ranking = {
        item["policy_name"]: item
        for item in result["subject_report"]["subjects"]["alpha"]["ranking"]
    }
    recommendation = result["recommendation_report"]

    _assert_score_components(structure_candidate["global_score_components"])
    _assert_score_components(subject_score["score_components"])
    _assert_score_components(
        subject_ranking["structure_size700_overlap100"]["score_components"]
    )
    _assert_score_components(recommendation["global_best_score_components"])
    assert "chunk_count_ratio" in subject_score["metrics"]
    assert "chunk_count_ratio" not in subject_score["score_components"]
    assert structure_candidate["global_score_components"]["section_score"] == 1.0
    assert subject_score["score_components"]["section_score"] == 1.0
    assert subject_score["score"] == 1.0
    assert recommendation["global_best_score"] == structure_candidate["global_score"]


def test_missing_subject_rules_do_not_skip_or_divide_by_zero(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(subjects=[_subject("alpha")]),
        "recursive_size700_overlap100": _report(subjects=[_subject("beta")]),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )
    subjects = result["subject_report"]["subjects"]
    alpha_candidate = next(
        item
        for item in subjects["alpha"]["ranking"]
        if item["policy_name"] == "recursive_size700_overlap100"
    )

    assert subjects["beta"]["recommendation"]["action"] == "needs_manual_review"
    assert (
        "subject missing from baseline report"
        in subjects["beta"]["recommendation"]["reason"]
    )
    assert alpha_candidate["status"] == "fail"
    assert alpha_candidate["reasons"] == ["subject missing from candidate report"]
    _assert_score_components(alpha_candidate["score_components"])
    assert all(value == 0.0 for value in alpha_candidate["score_components"].values())


def test_subject_status_rules_cover_source_empty_duplicate_and_chunk_ratio(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(
            subjects=[_subject("subject_a", chunk_count=100, source_count=2)]
        ),
        "recursive_size700_overlap100": _report(
            subjects=[_subject("subject_a", chunk_count=100, source_count=1)]
        ),
        "recursive_size900_overlap100": _report(
            subjects=[_subject("subject_a", chunk_count=100, empty_chunk_count=1)]
        ),
        "recursive_size1100_overlap100": _report(
            subjects=[
                _subject(
                    "subject_a",
                    chunk_count=100,
                    duplicate_chunk_count=4,
                )
            ]
        ),
        "recursive_size1200_overlap100": _report(
            subjects=[_subject("subject_a", chunk_count=150)]
        ),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            modes=("recursive",),
            chunk_sizes=(700, 900, 1100, 1200),
            overlaps=(100,),
            project_root=local_tmp_path,
        )
    )
    ranking = {
        item["policy_name"]: item
        for item in result["subject_report"]["subjects"]["subject_a"]["ranking"]
    }

    assert ranking["recursive_size700_overlap100"]["status"] == "fail"
    assert ranking["recursive_size900_overlap100"]["status"] == "fail"
    assert ranking["recursive_size1100_overlap100"]["status"] == "needs_review"
    assert ranking["recursive_size1200_overlap100"]["status"] == "needs_review"


def test_recursive_section_zero_does_not_fail_and_structure_section_scores(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(
            mode="recursive",
            subjects=[_subject("alpha", section_metadata_coverage=0.0)],
        ),
        "structure_size700_overlap100": _report(
            mode="structure",
            subjects=[
                _subject(
                    "alpha",
                    section_metadata_coverage=1.0,
                    unique_section_count=10,
                )
            ],
        ),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            modes=("structure",),
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )
    ranking = {
        item["policy_name"]: item
        for item in result["subject_report"]["subjects"]["alpha"]["ranking"]
    }

    assert ranking[BASELINE_CANDIDATE_NAME]["status"] == "pass"
    assert (
        ranking["structure_size700_overlap100"]["score"]
        > ranking[BASELINE_CANDIDATE_NAME]["score"]
    )


def test_sampled_run_never_considers_global_or_subject_candidate(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(
            subjects=[_subject("alpha", too_short_count=9)]
        ),
        "structure_size700_overlap100": _report(
            mode="structure",
            subjects=[
                _subject(
                    "alpha",
                    section_metadata_coverage=1.0,
                    unique_section_count=10,
                )
            ],
        ),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            modes=("structure",),
            max_candidates=2,
            sample_limit=20,
            project_root=local_tmp_path,
        )
    )

    assert result["recommendation_report"]["global_action"] != "consider_candidate"
    assert (
        result["subject_report"]["subjects"]["alpha"]["recommendation"]["action"]
        != "consider_candidate"
    )


def test_trace_is_policy_level_only_and_overwrites_stale_content(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(subjects=[_subject("alpha")]),
        "recursive_size700_overlap100": _report(subjects=[_subject("alpha")]),
    }
    _install_fake_evaluator(monkeypatch, reports)
    trace_output = local_tmp_path / "reports" / "policy_trace.jsonl"
    trace_output.parent.mkdir(parents=True)
    trace_output.write_text("stale trace\n", encoding="utf-8")

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            sample_limit=5,
            trace_enabled=True,
            trace_output=trace_output,
            run_id="tracecase",
            project_root=local_tmp_path,
        )
    )

    text = trace_output.read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines()]
    events = [record["event"] for record in records]
    serialized = json.dumps(records, ensure_ascii=False)

    assert "stale trace" not in text
    assert "chunk_evaluated" not in events
    assert "policy_optimizer_started" in events
    assert "subject_scored" in events
    assert "policy_recommendation_written" in events
    assert events[-1] == "policy_optimizer_finished"
    recommendation_events = [
        record
        for record in records
        if record["event"] == "policy_recommendation_written"
    ]
    assert (
        recommendation_events[-1]["report_path"] == result["recommendation_report_path"]
    )
    assert records[-1]["candidates_report_path"] == result["candidates_report_path"]
    assert records[-1]["subject_report_path"] == result["subject_report_path"]
    assert (
        records[-1]["recommendation_report_path"]
        == result["recommendation_report_path"]
    )
    for record in records:
        assert "sampled" in record
        assert "sample_limit" in record
        assert "candidate_count" in record
        assert "subject_count" in record
        assert "baseline_policy" in record
        assert "global_best_policy" in record
        assert "global_action" in record
    assert "COMPLETE CHUNK TEXT" not in serialized
    assert "preview" not in serialized
    assert "C:\\Users" not in serialized
    assert "API_KEY=secret" not in serialized
    assert "Traceback" not in serialized


def test_reports_do_not_copy_chunk_samples_or_overwrite_default_reports(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(subjects=[_subject("alpha")]),
        "recursive_size700_overlap100": _report(subjects=[_subject("alpha")]),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )

    output_dir = local_tmp_path / "reports"
    report_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            output_dir / "splitter_policy_candidates.json",
            output_dir / "splitter_policy_subject_report.json",
            output_dir / "splitter_policy_recommendation.json",
        ]
    )
    canonical_recommendation = json.loads(
        (output_dir / "splitter_policy_recommendation.json").read_text(encoding="utf-8")
    )
    full_recommendation = json.loads(
        (output_dir / "splitter_policy_recommendation_full.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["recommendation_full_report_path"].endswith(
        "splitter_policy_recommendation_full.json"
    )
    assert canonical_recommendation == full_recommendation
    assert (output_dir / f"policy_eval_{BASELINE_CANDIDATE_NAME}.json").exists()
    assert (output_dir / "policy_eval_recursive_size700_overlap100.json").exists()
    assert (output_dir / "splitter_policy_candidates_full.json").exists()
    assert (output_dir / "splitter_policy_subject_report_full.json").exists()
    assert not (output_dir / "chunk_eval_recursive.json").exists()
    assert not (output_dir / "chunk_eval_structure.json").exists()
    assert not (output_dir / "chunk_eval_compare.json").exists()
    assert not (output_dir / "chunk_optimizer_candidates.json").exists()
    assert not (output_dir / "chunk_optimizer_report.json").exists()
    assert "short_chunk_samples" not in report_text
    assert "COMPLETE CHUNK TEXT" not in report_text
    assert "preview" not in report_text


def test_sample_run_writes_sample_reports_and_preserves_full_outputs(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(subjects=[_subject("alpha")]),
        "recursive_size700_overlap100": _report(subjects=[_subject("alpha")]),
    }
    _install_fake_evaluator(monkeypatch, reports)

    full_result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )
    output_dir = local_tmp_path / "reports"
    canonical_before = (output_dir / "splitter_policy_recommendation.json").read_text(
        encoding="utf-8"
    )

    sample_result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            sample_limit=5,
            project_root=local_tmp_path,
        )
    )

    assert sample_result["candidates_report_path"].endswith(
        "splitter_policy_candidates_sample_limit5.json"
    )
    assert sample_result["subject_report_path"].endswith(
        "splitter_policy_subject_report_sample_limit5.json"
    )
    assert sample_result["recommendation_report_path"].endswith(
        "splitter_policy_recommendation_sample_limit5.json"
    )
    assert "recommendation_full_report_path" not in sample_result
    assert (output_dir / f"policy_eval_{BASELINE_CANDIDATE_NAME}.json").exists()
    assert (
        output_dir / f"policy_eval_sample_limit5_{BASELINE_CANDIDATE_NAME}.json"
    ).exists()
    sample_eval = json.loads(
        (
            output_dir / f"policy_eval_sample_limit5_{BASELINE_CANDIDATE_NAME}.json"
        ).read_text(encoding="utf-8")
    )
    assert sample_eval["sampled"] is True
    assert sample_eval["sample_limit"] == 5
    assert (output_dir / "splitter_policy_recommendation.json").read_text(
        encoding="utf-8"
    ) == canonical_before
    assert full_result["recommendation_report_path"].endswith(
        "splitter_policy_recommendation.json"
    )


def test_cli_prints_actual_report_paths(monkeypatch, local_tmp_path, capsys):
    def fake_optimize(config):
        return {
            "recommendation_report": {"global_action": "keep_current_default"},
            "candidates_report_path": "reports/splitter_policy_candidates_sample_limit5.json",
            "subject_report_path": "reports/splitter_policy_subject_report_sample_limit5.json",
            "recommendation_report_path": "reports/splitter_policy_recommendation_sample_limit5.json",
            "trace_path": "reports/manual_policy_trace.jsonl",
        }

    monkeypatch.setattr(policy_script, "optimize_splitter_policy", fake_optimize)

    policy_script.main(
        [
            "--data-dir",
            str(local_tmp_path / "data"),
            "--output-dir",
            str(local_tmp_path / "reports"),
            "--sample-limit",
            "5",
            "--trace",
            "--trace-output",
            str(local_tmp_path / "reports" / "manual_policy_trace.jsonl"),
        ]
    )

    output = capsys.readouterr().out
    assert "reports/splitter_policy_candidates_sample_limit5.json" in output
    assert "reports/splitter_policy_subject_report_sample_limit5.json" in output
    assert "reports/splitter_policy_recommendation_sample_limit5.json" in output
    assert "reports/manual_policy_trace.jsonl" in output


def test_recommendations_are_advisory_only(monkeypatch, local_tmp_path):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(subjects=[_subject("alpha")]),
        "recursive_size700_overlap100": _report(subjects=[_subject("alpha")]),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )

    recommendation = result["recommendation_report"]
    assert recommendation["do_not_auto_apply"] is True
    assert recommendation["warnings"] == [ADVISORY_WARNING]
    for payload in result["subject_report"]["subjects"].values():
        assert payload["recommendation"]["do_not_auto_apply"] is True


def test_candidate_failure_continues_and_sanitizes_trace(monkeypatch, local_tmp_path):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(subjects=[_subject("alpha")]),
        "recursive_size700_overlap100": RuntimeError(
            "candidate failed API_KEY=secret C:\\Users\\alpha\\traceback"
        ),
    }
    _install_fake_evaluator(monkeypatch, reports)
    trace_output = local_tmp_path / "reports" / "policy_trace.jsonl"

    result = optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            trace_enabled=True,
            trace_output=trace_output,
            project_root=local_tmp_path,
        )
    )

    failed = [
        candidate
        for candidate in result["candidates_report"]["candidates"]
        if candidate["global_status"] == "fail"
    ]
    records = [
        json.loads(line)
        for line in trace_output.read_text(encoding="utf-8").splitlines()
    ]
    serialized = json.dumps(records, ensure_ascii=False)

    assert failed[0]["policy_name"] == "recursive_size700_overlap100"
    assert any(record["event"] == "policy_candidate_failed" for record in records)
    assert "API_KEY=secret" not in serialized
    assert "C:\\Users" not in serialized
    assert "Traceback" not in serialized


def test_optimizer_failure_writes_failed_trace(monkeypatch, local_tmp_path):
    trace_output = local_tmp_path / "reports" / "policy_trace.jsonl"

    def broken_generate_candidates(config):
        raise RuntimeError("optimizer failed API_KEY=secret C:\\Users\\alpha")

    monkeypatch.setattr(
        policy_optimizer, "generate_candidates", broken_generate_candidates
    )

    with pytest.raises(RuntimeError, match="optimizer failed"):
        optimize_splitter_policy(
            SplitterPolicyOptimizerConfig(
                data_dir=local_tmp_path / "data",
                output_dir=local_tmp_path / "reports",
                trace_enabled=True,
                trace_output=trace_output,
                project_root=local_tmp_path,
            )
        )

    records = [
        json.loads(line)
        for line in trace_output.read_text(encoding="utf-8").splitlines()
    ]
    serialized = json.dumps(records, ensure_ascii=False)

    assert records[-1]["event"] == "policy_optimizer_failed"
    assert "API_KEY=secret" not in serialized
    assert "C:\\Users" not in serialized


def test_optimizer_does_not_modify_environment(monkeypatch, local_tmp_path):
    monkeypatch.setenv("RAG_SPLITTER_MODE", "invalid")
    before = os.environ.copy()
    _install_fake_evaluator(
        monkeypatch,
        {
            BASELINE_CANDIDATE_NAME: _report(subjects=[_subject("alpha")]),
            "recursive_size700_overlap100": _report(subjects=[_subject("alpha")]),
        },
    )

    optimize_splitter_policy(
        SplitterPolicyOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )

    assert os.environ["RAG_SPLITTER_MODE"] == "invalid"
    for key, value in before.items():
        assert os.environ.get(key) == value
