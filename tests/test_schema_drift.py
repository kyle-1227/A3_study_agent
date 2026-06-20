import pytest
from pydantic import ValidationError

from src.graph.web_research import WebSourceSummaryBatch
from src.llm.schema_drift import analyze_schema_drift_trace_only
from src.llm.schema_manifest import build_canonical_manifest, load_drift_guard_config


def _config() -> dict:
    return {
        "drift_guards": {
            "default": {
                "global_forbidden_output_fields": ["metadata", "debug", "raw_output"],
                "global_forbidden_aliases": {"reason": ["rationale", "explanation"]},
            },
            "web_source_summarizer": {
                "forbidden_output_fields": ["task_id", "subject", "url", "title"],
                "canonical_aliases": {
                    "coverage_points": ["key_points", "coverage"],
                    "use_case": ["purpose", "role"],
                },
            },
        }
    }


def test_drift_analyzer_reports_path_aware_aliases_and_metadata_leaks():
    manifest = build_canonical_manifest(
        WebSourceSummaryBatch,
        node_name="web_source_summarizer",
        output_mode="deepseek_tool_call_strict",
        config=_config(),
    )
    drift_guard = load_drift_guard_config("web_source_summarizer", config=_config())
    raw = {
        "summaries": [
            {
                "source_id": "websrc:0",
                "keep": True,
                "summary": "Useful source.",
                "coverage": ["Python loops"],
                "rationale": "Useful for practice.",
                "task_id": "task_python_0",
                "use_case": "exercise_material",
                "evidence_type": "tutorial",
                "relevance": "high",
                "usefulness": "high",
                "risk": "low",
            }
        ]
    }

    report = analyze_schema_drift_trace_only(
        raw,
        manifest=manifest,
        drift_guard=drift_guard,
        node_name="web_source_summarizer",
    )

    assert report.parsed_ok is True
    assert "summaries[0].coverage" in report.extra_fields_by_path
    assert report.alias_hits_by_path["summaries[0].coverage"] == "coverage_points"
    assert report.alias_hits_by_path["summaries[0].rationale"] == "reason"
    assert "summaries[0].task_id" in report.input_metadata_leak_by_path
    assert "summaries[0].coverage_points" in report.missing_required_by_path
    assert "summaries[0].reason" in report.missing_required_by_path


def test_drift_analyzer_redacts_and_bounds_raw_preview():
    manifest = build_canonical_manifest(
        WebSourceSummaryBatch,
        node_name="web_source_summarizer",
        output_mode="deepseek_tool_call_strict",
        config=_config(),
    )
    drift_guard = load_drift_guard_config("web_source_summarizer", config=_config())
    raw = (
        '{"Authorization": "Bearer secret-token-1234567890", '
        '"api_key": "sk-or-v1-secretvalue", "cookie": "session=abcdef", '
        '"summaries": []}'
    )

    report = analyze_schema_drift_trace_only(
        raw,
        manifest=manifest,
        drift_guard=drift_guard,
        node_name="web_source_summarizer",
    )

    assert "secret-token" not in report.raw_preview
    assert "sk-or-v1-secretvalue" not in report.raw_preview
    assert "session=abcdef" not in report.raw_preview
    assert len(report.raw_preview) <= 1014


def test_schema_drift_does_not_auto_normalize_aliases():
    with pytest.raises(ValidationError):
        WebSourceSummaryBatch.model_validate(
            {
                "summaries": [
                    {
                        "source_id": "websrc:0",
                        "keep": True,
                        "summary": "Useful source.",
                        "coverage": ["Python loops"],
                        "rationale": "Useful for practice.",
                        "evidence_type": "tutorial",
                        "use_case": "exercise_material",
                        "relevance": "high",
                        "usefulness": "high",
                        "risk": "low",
                    }
                ]
            }
        )
