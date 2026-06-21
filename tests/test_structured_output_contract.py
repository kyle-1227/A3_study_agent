import pytest
from pydantic import BaseModel, Field, ValidationError

from src.graph.evidence import EvidenceSufficiencyOutput
from src.graph.web_research import WebSourceSummaryBatch
from src.llm.structured_output import _json_output_contract_with_debug


class TinyContractModel(BaseModel):
    result: str = Field(..., description="Canonical result text.")
    reason: str = Field("")


def test_json_output_contract_injects_manifest_and_drift_guard():
    contract, debug = _json_output_contract_with_debug(
        WebSourceSummaryBatch,
        "web_source_summarizer",
        "deepseek_tool_call_strict",
    )

    assert "Canonical schema manifest: WebSourceSummaryBatch" in contract
    assert "summaries[].source_id" in contract
    assert "Use canonical field names exactly" in contract
    assert "Forbidden output fields:" in contract
    assert "task_id" in contract
    assert "coverage_points <-" in contract
    assert debug["manifest_injected"] is True
    assert debug["drift_guard_source"] == "default+web_source_summarizer"
    assert debug["drift_guard_config_validated"] is True
    assert debug["schema_manifest"]["schema_name"] == "WebSourceSummaryBatch"


def test_manifest_disabled_is_reported_in_contract_debug(monkeypatch):
    def fake_get_setting(key, default=None):
        if key == "structured_output":
            return {
                "manifest": {"enabled": False, "max_chars": 6000},
                "drift_guards": {},
            }
        return default

    monkeypatch.setattr("src.llm.schema_manifest.get_setting", fake_get_setting)

    contract, debug = _json_output_contract_with_debug(
        TinyContractModel,
        "tiny_node",
        "deepseek_tool_call_strict",
    )

    assert "Canonical schema manifest injection is disabled" in contract
    assert "Canonical schema manifest: TinyContractModel" not in contract
    assert debug["manifest_enabled"] is False
    assert debug["manifest_injected"] is False
    assert debug["manifest_truncated"] is False


def test_manifest_truncation_is_reported(monkeypatch):
    def fake_get_setting(key, default=None):
        if key == "structured_output":
            return {
                "manifest": {"enabled": True, "max_chars": 220},
                "drift_guards": {},
            }
        return default

    monkeypatch.setattr("src.llm.schema_manifest.get_setting", fake_get_setting)

    contract, debug = _json_output_contract_with_debug(
        EvidenceSufficiencyOutput,
        "evidence_sufficiency_judge",
        "deepseek_tool_call_strict",
    )

    assert "[manifest truncated]" in contract
    assert debug["manifest_truncated"] is True
    assert debug["manifest_max_chars"] == 220


def test_contract_does_not_make_aliases_validate():
    with pytest.raises(ValidationError):
        EvidenceSufficiencyOutput.model_validate(
            {
                "overall_evidence_state": "insufficient",
                "answerability": "cannot_answer",
                "need_more_local_rag": True,
                "need_more_web_research": True,
                "coverage_gaps": [
                    {
                        "topic": "Missing Python exercises",
                        "query": "Python practice questions",
                    }
                ],
                "decision_summary": "Need more evidence.",
            }
        )
