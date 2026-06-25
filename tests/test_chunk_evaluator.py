from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from langchain_core.documents import Document
import pytest

from scripts import evaluate_chunking as evaluate_script
from src.rag.eval import chunk_evaluator
from src.rag.eval.chunk_evaluator import (
    ChunkEvaluationConfig,
    build_compare_report,
    compare_modes,
    evaluate_mode,
)


@pytest.fixture
def local_tmp_path():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        yield Path(tmpdir)


def _metadata(
    *,
    mode: str = "recursive",
    subject: str = "alpha",
    chunk_index: int = 0,
    source_relpath: str = "data/alpha/source.txt",
    content_sha1: str = "content_1",
    section: bool = False,
    section_title: str = "Overview",
):
    metadata = {
        "doc_id": f"doc_{subject}",
        "chunk_id": f"chunk_{subject}_{chunk_index}",
        "source_relpath": source_relpath,
        "source_file": Path(source_relpath).name,
        "source_file_sha1": f"file_{subject}",
        "source_file_size": 100,
        "subject": subject,
        "doc_type": "course_material",
        "chunk_index": chunk_index,
        "chunk_policy_version": "structure_v1"
        if mode == "structure"
        else "recursive_v1",
        "index_version": "a3_rag_v1",
        "content_sha1": content_sha1,
        "chunk_chars": 20,
    }
    if mode == "structure" or section:
        metadata.update(
            {
                "splitter_mode": "structure",
                "section_id": f"sec_{subject}",
                "section_title": section_title,
                "section_chunk_index": chunk_index,
            }
        )
    return metadata


def _doc(text: str, **metadata_overrides) -> Document:
    metadata = _metadata(**metadata_overrides)
    metadata["chunk_chars"] = len(text)
    return Document(page_content=text, metadata=metadata)


def _prepare_data_dir(root: Path, subject: str = "alpha") -> Path:
    data_dir = root / "data"
    subject_dir = data_dir / subject
    subject_dir.mkdir(parents=True)
    (subject_dir / "placeholder.txt").write_text("placeholder", encoding="utf-8")
    return data_dir


def test_evaluate_mode_writes_report_and_no_trace_when_disabled(local_tmp_path):
    data_dir = _prepare_data_dir(local_tmp_path)
    output_dir = local_tmp_path / "reports"

    report = evaluate_mode(
        ChunkEvaluationConfig(
            mode="recursive",
            data_dir=data_dir,
            output_dir=output_dir,
            subjects=("alpha",),
            trace_enabled=False,
            project_root=local_tmp_path,
        )
    )

    assert report["mode"] == "recursive"
    assert report["sampled"] is False
    assert report["sample_limit"] is None
    assert report["trace_enabled"] is False
    assert report["trace_path"] is None
    assert (output_dir / "chunk_eval_recursive.json").exists()
    assert not list(output_dir.glob("chunk_eval_trace_*.jsonl"))


def test_evaluate_mode_uses_explicit_mode_without_changing_env(
    local_tmp_path, monkeypatch
):
    data_dir = _prepare_data_dir(local_tmp_path)
    before_env = os.environ.copy()
    monkeypatch.setenv("RAG_SPLITTER_MODE", "invalid")

    report = evaluate_mode(
        ChunkEvaluationConfig(
            mode="recursive",
            data_dir=data_dir,
            output_dir=local_tmp_path / "reports",
            subjects=("alpha",),
            trace_enabled=False,
            project_root=local_tmp_path,
        )
    )

    assert report["summary"]["total_chunks"] >= 1
    assert os.environ["RAG_SPLITTER_MODE"] == "invalid"
    for key, value in before_env.items():
        if key != "RAG_SPLITTER_MODE":
            assert os.environ.get(key) == value


def test_trace_records_chunk_ids_and_redacts_preview(local_tmp_path, monkeypatch):
    data_dir = _prepare_data_dir(local_tmp_path)
    output_dir = local_tmp_path / "reports"

    def fake_load_documents(directory, **kwargs):
        return [
            _doc(
                "FAKE_API_KEY=secret_token C:\\Users\\kyle\\secret full text",
                mode=kwargs["splitter_mode"],
                source_relpath="C:\\Users\\kyle\\secret\\source.txt",
                section=True,
                section_title="C:\\Users\\kyle\\section",
            )
        ]

    monkeypatch.setattr(chunk_evaluator, "load_documents", fake_load_documents)

    report = evaluate_mode(
        ChunkEvaluationConfig(
            mode="recursive",
            data_dir=data_dir,
            output_dir=output_dir,
            subjects=("alpha",),
            run_id="tracecase",
            project_root=local_tmp_path,
        )
    )

    trace_path = local_tmp_path / report["trace_path"]
    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    chunk_records = [
        record for record in records if record["event"] == "chunk_evaluated"
    ]

    assert chunk_records
    chunk_record = chunk_records[0]
    assert chunk_record["doc_id"] == "doc_alpha"
    assert chunk_record["chunk_id"] == "chunk_alpha_0"
    assert chunk_record["source_relpath"] == "source.txt"
    assert chunk_record["section_title"] == "<redacted>"
    assert len(chunk_record["preview"]) <= 120
    serialized = json.dumps(records, ensure_ascii=False)
    assert "secret_token" not in serialized
    assert "FAKE_API_KEY=" not in serialized
    assert "C:\\Users" not in serialized
    assert str(local_tmp_path.resolve()) not in serialized


def test_evaluate_mode_writes_run_failed_and_reraises(local_tmp_path, monkeypatch):
    data_dir = _prepare_data_dir(local_tmp_path)
    trace_output = local_tmp_path / "reports" / "trace.jsonl"

    def broken_evaluate_documents(documents, *, config):
        raise RuntimeError("boom secret_token C:\\Users\\kyle\\traceback")

    monkeypatch.setattr(
        chunk_evaluator, "evaluate_documents", broken_evaluate_documents
    )

    with pytest.raises(RuntimeError, match="boom"):
        evaluate_mode(
            ChunkEvaluationConfig(
                mode="recursive",
                data_dir=data_dir,
                output_dir=local_tmp_path / "reports",
                subjects=("alpha",),
                trace_output=trace_output,
                project_root=local_tmp_path,
            )
        )

    records = [
        json.loads(line)
        for line in trace_output.read_text(encoding="utf-8").splitlines()
    ]
    failed = [record for record in records if record["event"] == "run_failed"]

    assert failed
    assert failed[-1]["mode"] == "recursive"
    assert failed[-1]["error_type"] == "RuntimeError"
    serialized = json.dumps(failed, ensure_ascii=False)
    assert "secret_token" not in serialized
    assert "C:\\Users" not in serialized
    assert "Traceback" not in serialized


def test_evaluate_mode_overwrites_explicit_trace_output(local_tmp_path):
    data_dir = _prepare_data_dir(local_tmp_path)
    trace_output = local_tmp_path / "reports" / "trace.jsonl"
    trace_output.parent.mkdir(parents=True)
    trace_output.write_text("stale trace\n", encoding="utf-8")

    evaluate_mode(
        ChunkEvaluationConfig(
            mode="recursive",
            data_dir=data_dir,
            output_dir=local_tmp_path / "reports",
            subjects=("alpha",),
            trace_output=trace_output,
            project_root=local_tmp_path,
        )
    )

    text = trace_output.read_text(encoding="utf-8")
    assert "stale trace" not in text
    assert '"event": "run_started"' in text


def test_compare_modes_marks_sampled_reports(local_tmp_path, monkeypatch):
    data_dir = _prepare_data_dir(local_tmp_path)
    output_dir = local_tmp_path / "reports"

    def fake_load_documents(directory, **kwargs):
        mode = kwargs["splitter_mode"]
        return [
            _doc("chunk one body", mode=mode, chunk_index=0, content_sha1="a"),
            _doc("chunk two body", mode=mode, chunk_index=1, content_sha1="b"),
            _doc("chunk three body", mode=mode, chunk_index=2, content_sha1="c"),
        ]

    monkeypatch.setattr(chunk_evaluator, "load_documents", fake_load_documents)

    compare = compare_modes(
        baseline_mode="recursive",
        candidate_mode="structure",
        data_dir=data_dir,
        output_dir=output_dir,
        subjects=("alpha",),
        sample_limit=1,
        trace_enabled=False,
        project_root=local_tmp_path,
    )

    recursive_report = json.loads(
        (output_dir / "chunk_eval_recursive.json").read_text(encoding="utf-8")
    )
    structure_report = json.loads(
        (output_dir / "chunk_eval_structure.json").read_text(encoding="utf-8")
    )
    compare_report = json.loads(
        (output_dir / "chunk_eval_compare.json").read_text(encoding="utf-8")
    )

    assert compare["sampled"] is True
    assert recursive_report["sampled"] is True
    assert structure_report["sampled"] is True
    assert compare_report["sample_limit"] == 1
    assert recursive_report["summary"]["total_chunks"] == 1


def test_compare_modes_reuses_explicit_trace_without_clearing_candidate(
    local_tmp_path, monkeypatch
):
    data_dir = _prepare_data_dir(local_tmp_path)
    output_dir = local_tmp_path / "reports"
    trace_output = output_dir / "trace.jsonl"
    trace_output.parent.mkdir(parents=True)
    trace_output.write_text("stale trace\n", encoding="utf-8")

    def fake_load_documents(directory, **kwargs):
        mode = kwargs["splitter_mode"]
        return [_doc("chunk body", mode=mode, content_sha1=mode)]

    monkeypatch.setattr(chunk_evaluator, "load_documents", fake_load_documents)

    compare_modes(
        baseline_mode="recursive",
        candidate_mode="structure",
        data_dir=data_dir,
        output_dir=output_dir,
        subjects=("alpha",),
        trace_output=trace_output,
        run_id="comparetrace",
        project_root=local_tmp_path,
    )

    text = trace_output.read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines()]
    run_started_modes = [
        record["mode"] for record in records if record["event"] == "run_started"
    ]

    assert "stale trace" not in text
    assert run_started_modes == ["recursive", "structure"]
    assert any(record["event"] == "comparison_written" for record in records)


def test_compare_modes_writes_run_failed_and_reraises(local_tmp_path, monkeypatch):
    data_dir = _prepare_data_dir(local_tmp_path)
    trace_output = local_tmp_path / "reports" / "trace.jsonl"

    def broken_evaluate_mode(config, *, reset_trace):
        raise ValueError("compare failed C:\\Users\\kyle\\secret")

    monkeypatch.setattr(chunk_evaluator, "_evaluate_mode", broken_evaluate_mode)

    with pytest.raises(ValueError, match="compare failed"):
        compare_modes(
            baseline_mode="recursive",
            candidate_mode="structure",
            data_dir=data_dir,
            output_dir=local_tmp_path / "reports",
            subjects=("alpha",),
            trace_output=trace_output,
            project_root=local_tmp_path,
        )

    records = [
        json.loads(line)
        for line in trace_output.read_text(encoding="utf-8").splitlines()
    ]
    failed = [record for record in records if record["event"] == "run_failed"]

    assert failed
    assert failed[-1]["baseline_mode"] == "recursive"
    assert failed[-1]["candidate_mode"] == "structure"
    assert failed[-1]["error_type"] == "ValueError"
    assert "C:\\Users" not in json.dumps(failed, ensure_ascii=False)


def _report(
    *,
    mode: str,
    total_chunks: int = 10,
    source_count: int = 1,
    empty_count: int = 0,
    short_ratio: float = 0.0,
    duplicate_ratio: float = 0.0,
    missing: dict[str, int] | None = None,
):
    return {
        "mode": mode,
        "sampled": False,
        "sample_limit": None,
        "trace_enabled": False,
        "trace_path": None,
        "summary": {
            "total_chunks": total_chunks,
            "source_count": source_count,
            "too_short_count": int(total_chunks * short_ratio),
            "too_short_ratio": short_ratio,
            "empty_chunk_count": empty_count,
            "duplicate_chunk_count": int(total_chunks * duplicate_ratio),
            "duplicate_ratio": duplicate_ratio,
        },
        "metadata": {
            "section_metadata_coverage": 1.0 if mode == "structure" else 0.0,
            "missing_metadata_counts": missing or {},
        },
    }


def test_compare_judgement_is_conservative():
    baseline = _report(mode="recursive")

    fail_report = build_compare_report(
        baseline,
        _report(mode="structure", empty_count=1),
    )
    review_report = build_compare_report(
        baseline,
        _report(mode="structure", total_chunks=13),
    )
    pass_report = build_compare_report(
        baseline,
        _report(mode="structure", total_chunks=10),
    )

    assert fail_report["judgement"]["status"] == "fail"
    assert review_report["judgement"]["status"] == "needs_review"
    assert pass_report["judgement"]["status"] == "pass"


def test_cli_rejects_invalid_mode():
    with pytest.raises(SystemExit):
        evaluate_script.parse_args(["--mode", "invalid"])


def test_cli_rejects_output_with_compare(local_tmp_path):
    with pytest.raises(SystemExit):
        evaluate_script.parse_args(
            [
                "--compare",
                "recursive",
                "structure",
                "--output",
                str(local_tmp_path / "custom.json"),
            ]
        )


def test_cli_no_trace_passes_trace_disabled(monkeypatch, local_tmp_path):
    captured: dict[str, object] = {}

    def fake_evaluate_mode(config):
        captured["trace_enabled"] = config.trace_enabled
        return {"trace_enabled": config.trace_enabled}

    monkeypatch.setattr(evaluate_script, "evaluate_mode", fake_evaluate_mode)

    evaluate_script.main(
        [
            "--mode",
            "recursive",
            "--data-dir",
            str(local_tmp_path / "data"),
            "--output-dir",
            str(local_tmp_path / "reports"),
            "--no-trace",
        ]
    )

    assert captured == {"trace_enabled": False}


def test_cli_sample_limit_writes_sampled_report(monkeypatch, local_tmp_path):
    output_dir = local_tmp_path / "reports"

    def fake_evaluate_mode(config):
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "mode": config.mode,
            "sampled": config.sample_limit is not None,
            "sample_limit": config.sample_limit,
            "trace_enabled": config.trace_enabled,
        }
        (output_dir / f"chunk_eval_{config.mode}.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(evaluate_script, "evaluate_mode", fake_evaluate_mode)

    evaluate_script.main(
        [
            "--mode",
            "recursive",
            "--data-dir",
            str(local_tmp_path / "data"),
            "--output-dir",
            str(output_dir),
            "--sample-limit",
            "7",
        ]
    )

    payload = json.loads(
        (output_dir / "chunk_eval_recursive.json").read_text(encoding="utf-8")
    )
    assert payload["sampled"] is True
    assert payload["sample_limit"] == 7
