from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import pytest

from scripts import optimize_chunking as optimize_script
from src.rag.eval import chunk_optimizer
from src.rag.eval.chunk_optimizer import (
    BASELINE_CANDIDATE_NAME,
    ChunkOptimizerConfig,
    generate_candidates,
    optimize_chunking,
)


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def _report(
    *,
    mode: str = "recursive",
    total_chunks: int = 100,
    source_count: int = 2,
    empty_count: int = 0,
    too_short_ratio: float = 0.0,
    duplicate_ratio: float = 0.0,
    required_metadata_coverage: float = 1.0,
    section_metadata_coverage: float = 0.0,
) -> dict:
    return {
        "mode": mode,
        "sampled": False,
        "sample_limit": None,
        "trace_enabled": False,
        "trace_path": None,
        "summary": {
            "total_chunks": total_chunks,
            "source_count": source_count,
            "too_short_count": int(total_chunks * too_short_ratio),
            "too_short_ratio": too_short_ratio,
            "too_long_count": 0,
            "empty_chunk_count": empty_count,
            "duplicate_chunk_count": int(total_chunks * duplicate_ratio),
            "duplicate_ratio": duplicate_ratio,
            "short_chunk_samples": [
                {"preview": "COMPLETE CHUNK TEXT THAT MUST NOT ENTER OPTIMIZER REPORT"}
            ],
        },
        "metadata": {
            "required_metadata_coverage": required_metadata_coverage,
            "section_metadata_coverage": section_metadata_coverage,
            "missing_metadata_counts": {}
            if required_metadata_coverage == 1.0
            else {"doc_id": 1},
        },
        "structure": {},
        "per_subject": [],
        "per_source": [],
        "warnings": [],
    }


def _install_fake_evaluator(monkeypatch, reports_by_name: dict[str, dict]):
    calls = []

    def fake_evaluate_mode(config):
        assert config.trace_enabled is False
        assert config.trace_output is None
        assert config.output_path is not None
        name = config.output_path.stem.removeprefix("chunk_eval_")
        calls.append(
            {
                "name": name,
                "mode": config.mode,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
        )
        if name in reports_by_name and isinstance(reports_by_name[name], Exception):
            raise reports_by_name[name]
        report = reports_by_name.get(
            name,
            _report(
                mode=config.mode,
                section_metadata_coverage=1.0 if config.mode == "structure" else 0.0,
            ),
        )
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(chunk_optimizer, "evaluate_mode", fake_evaluate_mode)
    return calls


def test_generate_candidates_default_names_and_baseline_first():
    candidates = generate_candidates(ChunkOptimizerConfig(data_dir=Path("data")))
    names = [candidate.name for candidate in candidates]

    assert names == [
        "recursive_size1000_overlap200",
        "recursive_size700_overlap100",
        "recursive_size900_overlap150",
        "recursive_size1200_overlap200",
        "structure_size700_overlap100",
        "structure_size900_overlap150",
        "structure_size1000_overlap200",
        "structure_size1200_overlap200",
    ]


def test_generate_candidates_skips_invalid_overlap_and_adds_baseline():
    candidates = generate_candidates(
        ChunkOptimizerConfig(
            data_dir=Path("data"),
            modes=("recursive",),
            chunk_sizes=(100,),
            overlaps=(50, 100),
        )
    )
    names = [candidate.name for candidate in candidates]

    assert names == [BASELINE_CANDIDATE_NAME, "recursive_size100_overlap50"]


def test_generate_candidates_max_candidates_rules():
    one = generate_candidates(
        ChunkOptimizerConfig(data_dir=Path("data"), max_candidates=1)
    )
    three = generate_candidates(
        ChunkOptimizerConfig(data_dir=Path("data"), max_candidates=3)
    )

    assert [candidate.name for candidate in one] == [BASELINE_CANDIDATE_NAME]
    assert [candidate.name for candidate in three] == [
        BASELINE_CANDIDATE_NAME,
        "recursive_size700_overlap100",
        "recursive_size900_overlap150",
    ]
    with pytest.raises(ValueError, match="max_candidates"):
        generate_candidates(
            ChunkOptimizerConfig(data_dir=Path("data"), max_candidates=0)
        )


def test_cli_trace_output_requires_trace(local_tmp_path):
    with pytest.raises(SystemExit):
        optimize_script.parse_args(
            ["--trace-output", str(local_tmp_path / "optimizer_trace.jsonl")]
        )


def test_cli_passes_config_without_enabling_trace(monkeypatch, local_tmp_path, capsys):
    captured: dict[str, object] = {}

    def fake_optimize(config):
        captured["trace_enabled"] = config.trace_enabled
        captured["max_candidates"] = config.max_candidates
        return {
            "candidates_report_path": "reports/chunk_optimizer_candidates.json",
            "recommendation_report_path": "reports/chunk_optimizer_report.json",
            "trace_path": None,
            "recommendation_report": {
                "recommendation": {"action": "keep_current_default"}
            },
        }

    monkeypatch.setattr(optimize_script, "optimize_chunking", fake_optimize)

    optimize_script.main(
        [
            "--data-dir",
            str(local_tmp_path / "data"),
            "--output-dir",
            str(local_tmp_path / "reports"),
            "--max-candidates",
            "1",
        ]
    )

    assert captured == {"trace_enabled": False, "max_candidates": 1}
    assert "Recommendation action" in capsys.readouterr().out


def test_optimizer_writes_candidate_reports_without_overwriting_phase4d_defaults(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(mode="recursive"),
        "recursive_size700_overlap100": _report(mode="recursive"),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_chunking(
        ChunkOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )

    output_dir = local_tmp_path / "reports"
    assert (output_dir / f"chunk_eval_{BASELINE_CANDIDATE_NAME}.json").exists()
    assert (output_dir / "chunk_eval_recursive_size700_overlap100.json").exists()
    assert not (output_dir / "chunk_eval_recursive.json").exists()
    assert not (output_dir / "chunk_eval_structure.json").exists()
    assert not (output_dir / "chunk_eval_compare.json").exists()
    assert result["trace_path"] is None


def test_optimizer_trace_is_candidate_level_only(monkeypatch, local_tmp_path):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(mode="recursive"),
        "recursive_size700_overlap100": _report(mode="recursive"),
    }
    _install_fake_evaluator(monkeypatch, reports)
    trace_output = local_tmp_path / "reports" / "trace.jsonl"
    trace_output.parent.mkdir(parents=True)
    trace_output.write_text("stale trace\n", encoding="utf-8")

    optimize_chunking(
        ChunkOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            trace_enabled=True,
            trace_output=trace_output,
            run_id="tracecase",
            project_root=local_tmp_path,
        )
    )

    text = trace_output.read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines()]
    events = [record["event"] for record in records]

    assert "stale trace" not in text
    assert "chunk_evaluated" not in events
    assert events[0] == "optimizer_started"
    assert "candidate_started" in events
    assert "candidate_finished" in events
    assert "recommendation_written" in events
    assert events[-1] == "optimizer_finished"


def test_sampled_optimizer_never_considers_candidate(monkeypatch, local_tmp_path):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(mode="recursive", too_short_ratio=0.08),
        "recursive_size700_overlap100": _report(mode="recursive"),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_chunking(
        ChunkOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            sample_limit=10,
            project_root=local_tmp_path,
        )
    )

    action = result["recommendation_report"]["recommendation"]["action"]
    assert action in {"needs_manual_review", "keep_current_default"}
    assert action != "consider_candidate"


def test_status_rules_cover_empty_metadata_missing_and_source_lost(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(mode="recursive", source_count=3),
        "recursive_size700_overlap100": _report(mode="recursive", empty_count=1),
        "recursive_size900_overlap100": _report(
            mode="recursive", required_metadata_coverage=0.9
        ),
        "recursive_size1100_overlap100": _report(mode="recursive", source_count=2),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_chunking(
        ChunkOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            modes=("recursive",),
            chunk_sizes=(700, 900, 1100),
            overlaps=(100,),
            project_root=local_tmp_path,
        )
    )
    by_name = {
        candidate["name"]: candidate
        for candidate in result["candidates_report"]["candidates"]
    }

    assert by_name["recursive_size700_overlap100"]["status"] == "fail"
    assert by_name["recursive_size900_overlap100"]["status"] == "fail"
    assert by_name["recursive_size1100_overlap100"]["status"] == "fail"


def test_structure_section_coverage_adds_score_without_recursive_failure(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(mode="recursive"),
        "structure_size700_overlap100": _report(
            mode="structure", section_metadata_coverage=1.0
        ),
    }
    _install_fake_evaluator(monkeypatch, reports)

    result = optimize_chunking(
        ChunkOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            modes=("structure",),
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )
    by_name = {
        candidate["name"]: candidate
        for candidate in result["candidates_report"]["candidates"]
    }

    assert by_name[BASELINE_CANDIDATE_NAME]["status"] == "pass"
    assert (
        by_name["structure_size700_overlap100"]["score"]
        > by_name[BASELINE_CANDIDATE_NAME]["score"]
    )


def test_candidate_failure_continues_and_writes_failed_trace(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(mode="recursive"),
        "recursive_size700_overlap100": RuntimeError(
            "boom API_KEY=secret C:\\Users\\kyle\\traceback"
        ),
    }
    _install_fake_evaluator(monkeypatch, reports)
    trace_output = local_tmp_path / "reports" / "trace.jsonl"

    result = optimize_chunking(
        ChunkOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            trace_enabled=True,
            trace_output=trace_output,
            project_root=local_tmp_path,
        )
    )

    candidates = result["candidates_report"]["candidates"]
    failed = [candidate for candidate in candidates if candidate["status"] == "fail"]
    records = [
        json.loads(line)
        for line in trace_output.read_text(encoding="utf-8").splitlines()
    ]
    serialized = json.dumps(records, ensure_ascii=False)

    assert len(candidates) == 2
    assert failed[0]["name"] == "recursive_size700_overlap100"
    assert any(record["event"] == "candidate_failed" for record in records)
    assert "secret" not in serialized.casefold()
    assert "C:\\Users" not in serialized
    assert "Traceback" not in serialized


def test_optimizer_failed_event_for_unrecoverable_error(monkeypatch, local_tmp_path):
    trace_output = local_tmp_path / "reports" / "trace.jsonl"

    def broken_generate_candidates(config):
        raise RuntimeError("optimizer exploded C:\\Users\\kyle\\secret")

    monkeypatch.setattr(
        chunk_optimizer, "generate_candidates", broken_generate_candidates
    )

    with pytest.raises(RuntimeError, match="optimizer exploded"):
        optimize_chunking(
            ChunkOptimizerConfig(
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
    assert records[-1]["event"] == "optimizer_failed"
    serialized = json.dumps(records, ensure_ascii=False)
    assert "C:\\Users" not in serialized
    assert "Traceback" not in serialized


def test_optimizer_does_not_modify_env(monkeypatch, local_tmp_path):
    monkeypatch.setenv("RAG_SPLITTER_MODE", "invalid")
    before = os.environ.copy()
    _install_fake_evaluator(
        monkeypatch,
        {
            BASELINE_CANDIDATE_NAME: _report(mode="recursive"),
            "recursive_size700_overlap100": _report(mode="recursive"),
        },
    )

    optimize_chunking(
        ChunkOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )

    assert os.environ["RAG_SPLITTER_MODE"] == "invalid"
    for key, value in before.items():
        assert os.environ.get(key) == value


def test_optimizer_reports_exclude_chunk_text_preview_paths_and_secrets(
    monkeypatch, local_tmp_path
):
    reports = {
        BASELINE_CANDIDATE_NAME: _report(mode="recursive"),
        "recursive_size700_overlap100": RuntimeError(
            "candidate failed API_KEY=secret C:\\Users\\kyle\\secret"
        ),
    }
    _install_fake_evaluator(monkeypatch, reports)

    optimize_chunking(
        ChunkOptimizerConfig(
            data_dir=local_tmp_path / "data",
            output_dir=local_tmp_path / "reports",
            max_candidates=2,
            project_root=local_tmp_path,
        )
    )

    candidates_text = (
        local_tmp_path / "reports" / "chunk_optimizer_candidates.json"
    ).read_text(encoding="utf-8")
    report_text = (
        local_tmp_path / "reports" / "chunk_optimizer_report.json"
    ).read_text(encoding="utf-8")
    serialized = candidates_text + report_text

    assert "COMPLETE CHUNK TEXT" not in serialized
    assert "API_KEY=secret" not in serialized
    assert "C:\\Users" not in serialized
    assert ".env" not in serialized
    assert "Authorization" not in serialized
    assert "Cookie" not in serialized
    assert "postgresql://" not in serialized
