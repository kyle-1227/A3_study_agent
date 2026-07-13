"""First-class QA agent, capability context, and qa_final tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from src.context_engineering.input_manifest import build_llm_input_manifest
from src.graph.capability_registry import (
    build_safe_capability_context,
    get_registered_resource_types,
)
from src.graph.qa import (
    QAFinalEvent,
    QAResponse,
    QASuggestion,
    build_general_qa_node_output,
    build_qa_final_payload,
    qa_agent,
    qa_final_payload,
    validate_qa_response,
)
from src.graph.qa_suggestion_registry import (
    build_safe_qa_suggestion_registry,
    get_qa_suggestion_resource_types,
)


def _response(
    *,
    grounding_status: str,
    uncertainty_note: str = "",
    action: str = "continue_qa",
    resource_type: str = "",
    answer: str = "A bounded answer.",
    suggestions: list[QASuggestion] | None = None,
) -> QAResponse:
    return QAResponse(
        answer=answer,
        uncertainty_note=uncertainty_note,
        grounding_status=grounding_status,
        suggestions=(
            suggestions
            if suggestions is not None
            else [
                QASuggestion(
                    label="Continue",
                    action=action,
                    resource_type=resource_type,
                )
            ]
        ),
    )


def test_academic_qa_requires_judged_evidence_grounding():
    valid = _response(grounding_status="judged_evidence")
    invalid = _response(grounding_status="general_knowledge")

    assert (
        validate_qa_response(
            valid,
            qa_scope="academic",
            kept_evidence_count=2,
            requires_live_verification=False,
        )
        == ""
    )
    assert "judged_evidence" in validate_qa_response(
        invalid,
        qa_scope="academic",
        kept_evidence_count=2,
        requires_live_verification=False,
    )


def test_academic_qa_without_kept_evidence_requires_insufficient_status_and_note():
    valid = _response(
        grounding_status="insufficient_evidence",
        uncertainty_note="The approved evidence set is empty.",
    )
    missing_note = _response(grounding_status="insufficient_evidence")

    assert (
        validate_qa_response(
            valid,
            qa_scope="academic",
            kept_evidence_count=0,
            requires_live_verification=False,
        )
        == ""
    )
    assert "uncertainty_note" in validate_qa_response(
        missing_note,
        qa_scope="academic",
        kept_evidence_count=0,
        requires_live_verification=False,
    )


def test_general_live_qa_must_disclose_missing_live_verification():
    valid = _response(
        grounding_status="not_live_verified",
        uncertainty_note="This response was not live-verified.",
    )
    invalid = _response(grounding_status="general_knowledge")

    assert (
        validate_qa_response(
            valid,
            qa_scope="general",
            kept_evidence_count=0,
            requires_live_verification=True,
        )
        == ""
    )
    assert "not_live_verified" in validate_qa_response(
        invalid,
        qa_scope="general",
        kept_evidence_count=0,
        requires_live_verification=True,
    )


@pytest.mark.parametrize(
    "action",
    ["explain_algorithm_complexity", "learn", "explain_algorithm"],
)
def test_deepseek_style_unknown_suggestion_action_is_rejected_with_correction_values(
    action: str,
):
    unknown_action = _response(grounding_status="general_knowledge", action=action)

    error = validate_qa_response(
        unknown_action,
        qa_scope="general",
        kept_evidence_count=0,
        requires_live_verification=False,
    )

    assert "suggestions[0].action is invalid" in error
    assert "ask_followup" in error
    assert "continue_qa" in error
    assert "generate_resource" in error
    assert "review_doc" in error
    assert "study_plan" in error


def test_unknown_suggestion_resource_is_rejected_with_correction_values():
    unknown_resource = _response(
        grounding_status="general_knowledge",
        action="generate_resource",
        resource_type="invented_resource",
    )

    error = validate_qa_response(
        unknown_resource,
        qa_scope="general",
        kept_evidence_count=0,
        requires_live_verification=False,
    )
    assert "suggestions[0].resource_type is invalid" in error
    assert "allowed resource_type values" in error
    assert "quiz" in error


def test_qa_suggestion_correction_is_preserved_in_structured_reask():
    from src.llm.structured_output import (
        StructuredLLMResult,
        _build_reask_instruction,
    )

    business_error = validate_qa_response(
        _response(
            grounding_status="general_knowledge",
            action="explain_algorithm_complexity",
        ),
        qa_scope="general",
        kept_evidence_count=0,
        requires_live_verification=False,
    )
    result = StructuredLLMResult(
        success=False,
        parsed=None,
        node_name="qa_agent",
        llm_node="qa_agent",
        schema_name="QAResponse",
        provider="configured-provider",
        model="configured-model",
        output_mode="deepseek_tool_call_strict",
        failure_phase="business_validation_error",
        business_validation_error=business_error,
    )

    correction = _build_reask_instruction(
        result=result,
        schema_name="QAResponse",
        previous_error_summary=business_error,
    )

    assert (
        "allowed action values: [ask_followup, continue_qa, generate_resource]"
        in correction
    )
    assert "allowed resource_type values" in correction
    assert "review_doc" in correction


def test_empty_suggestions_are_valid_for_direct_qa_answer():
    response = _response(
        grounding_status="general_knowledge",
        suggestions=[],
    )

    assert (
        validate_qa_response(
            response,
            qa_scope="general",
            kept_evidence_count=0,
            requires_live_verification=False,
        )
        == ""
    )


def test_general_qa_node_output_requires_identity_and_validates_contract():
    result = build_general_qa_node_output(
        answer="A validated emotional-support response.",
        state={"thread_id": "thread-1", "request_id": "request-1"},
    )

    final = QAFinalEvent.model_validate(result["last_qa_response"])
    assert final.qa_scope == "general"
    assert final.response.grounding_status == "general_knowledge"
    assert final.response.suggestions == []
    with pytest.raises(ValueError, match="explicit thread_id and request_id"):
        build_general_qa_node_output(answer="Missing identity", state={})


def test_qa_suggestion_registry_is_safe_minimal_and_uses_canonical_resources():
    registry = build_safe_qa_suggestion_registry()

    assert registry.startswith("<QA_SUGGESTION_REGISTRY>")
    assert registry.endswith("</QA_SUGGESTION_REGISTRY>")
    assert "<CAPABILITY_CONTEXT>" not in registry
    payload = json.loads(
        registry.removeprefix("<QA_SUGGESTION_REGISTRY>\n").removesuffix(
            "\n</QA_SUGGESTION_REGISTRY>"
        )
    )
    assert payload["suggestions_optional"] is True
    assert tuple(payload["resource_types"]) == get_qa_suggestion_resource_types()


def test_capability_context_uses_runtime_registries_and_excludes_secrets():
    context = build_safe_capability_context(
        context_policy_mode="strict",
        runtime_metadata={
            "checkpointer_enabled": True,
            "checkpointer_type": "postgres",
            "db_uri": "postgresql://user:secret@host/db",
            "api_token": "secret-token",
        },
    )

    assert context.startswith("<CAPABILITY_CONTEXT>")
    assert context.endswith("</CAPABILITY_CONTEXT>")
    assert "postgresql://" not in context
    assert "secret-token" not in context
    payload = json.loads(
        context.removeprefix("<CAPABILITY_CONTEXT>\n").removesuffix(
            "\n</CAPABILITY_CONTEXT>"
        )
    )
    assert tuple(payload["resource_types"]) == get_registered_resource_types()
    assert payload["context_engineering"]["policy_mode"] == "strict"
    assert payload["persistence"] == {
        "checkpointer_enabled": True,
        "checkpointer_type": "postgres",
    }


@pytest.mark.anyio
async def test_general_qa_agent_uses_one_structured_contract(monkeypatch):
    from src.graph import qa as qa_module

    mock_invoke = AsyncMock(
        return_value=SimpleNamespace(
            parsed=_response(grounding_status="general_knowledge")
        )
    )
    monkeypatch.setattr(qa_module, "invoke_structured_llm", mock_invoke)

    result = await qa_agent(
        {
            "messages": [HumanMessage(content="Explain a stable general concept")],
            "response_mode": "qa",
            "qa_scope": "general",
            "requires_live_verification": False,
            "thread_id": "thread-1",
            "request_id": "request-1",
        }
    )

    kwargs = mock_invoke.await_args.kwargs
    assert kwargs["node_name"] == "qa_agent"
    assert kwargs["llm_node"] == "qa_agent"
    assert kwargs["schema"] is QAResponse
    assert kwargs["fallback_modes"] == []
    assert len(kwargs["messages"]) == 3
    assert "<QA_SUGGESTION_REGISTRY>" in kwargs["messages"][1].content
    assert not any(
        "<CAPABILITY_CONTEXT>" in message.content for message in kwargs["messages"]
    )
    assert result["final_response_type"] == "qa"
    assert QAFinalEvent.model_validate(result["last_qa_response"])


@pytest.mark.anyio
async def test_a3_qa_agent_inserts_capability_context_before_user(monkeypatch):
    from src.graph import qa as qa_module

    mock_invoke = AsyncMock(
        return_value=SimpleNamespace(
            parsed=_response(grounding_status="capability_registry")
        )
    )
    monkeypatch.setattr(qa_module, "invoke_structured_llm", mock_invoke)
    monkeypatch.setattr(
        qa_module,
        "resolve_context_runtime_policy",
        lambda: SimpleNamespace(mode="strict"),
    )

    await qa_agent(
        {
            "messages": [HumanMessage(content="Describe the available capabilities")],
            "response_mode": "qa",
            "qa_scope": "a3_agent",
            "requires_live_verification": False,
            "runtime_capability_metadata": {
                "checkpointer_enabled": False,
                "checkpointer_type": "none",
            },
            "thread_id": "thread-1",
            "request_id": "request-1",
        }
    )

    messages = mock_invoke.await_args.kwargs["messages"]
    assert len(messages) == 4
    assert "<QA_SUGGESTION_REGISTRY>" in messages[1].content
    assert "<CAPABILITY_CONTEXT>" in messages[2].content
    assert isinstance(messages[-1], HumanMessage)


@pytest.mark.anyio
async def test_academic_qa_agent_validator_observes_kept_evidence_count(monkeypatch):
    from src.graph import qa as qa_module

    mock_invoke = AsyncMock(
        return_value=SimpleNamespace(
            parsed=_response(grounding_status="judged_evidence")
        )
    )
    monkeypatch.setattr(qa_module, "invoke_structured_llm", mock_invoke)

    await qa_agent(
        {
            "messages": [HumanMessage(content="Explain the judged topic")],
            "response_mode": "qa",
            "qa_scope": "academic",
            "requires_live_verification": False,
            "graded_evidence": [{"evidence_id": "kept-1"}],
            "thread_id": "thread-1",
            "request_id": "request-1",
        }
    )

    validator = mock_invoke.await_args.kwargs["business_validator"]
    assert validator(_response(grounding_status="judged_evidence")) == ""
    assert "judged_evidence" in validator(
        _response(grounding_status="general_knowledge")
    )


def test_qa_final_is_stable_bounded_and_current_request_only():
    response = _response(
        grounding_status="general_knowledge",
        answer="a" * 10000,
    )
    first = build_qa_final_payload(
        response=response,
        qa_scope="general",
        thread_id="thread-1",
        request_id="request-1",
    )
    second = build_qa_final_payload(
        response=response,
        qa_scope="general",
        thread_id="thread-1",
        request_id="request-1",
    )

    assert first["qa_id"] == second["qa_id"]
    assert first["payload_hash"] == second["payload_hash"]
    assert len(first["response"]["answer"]) == 6000
    assert qa_final_payload({"request_id": "request-1", "last_qa_response": first})
    assert (
        qa_final_payload({"request_id": "request-2", "last_qa_response": first}) is None
    )
    assert qa_final_payload({"request_id": "request-1", "last_qa_response": {}}) is None


def test_manifest_identifies_capability_context_without_storing_content():
    capability = build_safe_capability_context(
        context_policy_mode="strict",
        runtime_metadata={
            "checkpointer_enabled": False,
            "checkpointer_type": "none",
        },
    )
    manifest = build_llm_input_manifest(
        node_name="qa_agent",
        llm_node="qa_agent",
        provider="configured-provider",
        model="configured-model",
        messages=[
            {"role": "system", "content": "Structured output contract"},
            {"role": "system", "content": build_safe_qa_suggestion_registry()},
            {"role": "system", "content": capability},
            {"role": "user", "content": "question"},
        ],
        state={"request_id": "request-1", "thread_id": "thread-1"},
        call_purpose="structured_llm",
        output_mode="strict",
        schema_name="QAResponse",
        schema_contract_first=True,
    )

    assert "capability_context" in manifest["section_names"]
    assert "qa_suggestion_registry" in manifest["section_names"]
    section = next(
        item for item in manifest["sections"] if item["section"] == "capability_context"
    )
    assert section["char_count"] == len(capability)
    assert "resource_types" not in json.dumps(manifest, ensure_ascii=False)


def test_qa_agent_is_in_structured_active_rollout():
    from src.llm.structured_output import _structured_context_apply_config

    config = _structured_context_apply_config()

    assert config["enabled"] is True
    assert config["mode"] == "active"
    assert config["allow_structured_output"] is True
    assert "qa_agent" in config["active_nodes"]


def test_qa_structured_message_order_is_contract_registry_capability_ce_then_user():
    from src.llm.structured_output import _prepare_structured_messages_with_context

    capability = build_safe_capability_context(
        context_policy_mode="strict",
        runtime_metadata={
            "checkpointer_enabled": False,
            "checkpointer_type": "none",
        },
    )
    registry = build_safe_qa_suggestion_registry()
    result = _prepare_structured_messages_with_context(
        node_name="qa_agent",
        llm_node="qa_agent",
        messages=[
            {"role": "system", "content": "Structured output contract"},
            {"role": "system", "content": "QA business prompt"},
            {"role": "system", "content": registry},
            {"role": "system", "content": capability},
            {"role": "user", "content": "question"},
        ],
        state={
            "request_id": "request-1",
            "thread_id": "thread-1",
            "response_mode": "qa",
            "qa_scope": "a3_agent",
        },
    )

    assert result.debug["structured_context_apply_status"] == "applied"
    assert result.messages[0]["content"] == "Structured output contract"
    assert result.messages[1]["content"] == "QA business prompt"
    assert "<QA_SUGGESTION_REGISTRY>" in result.messages[2]["content"]
    assert "<CAPABILITY_CONTEXT>" in result.messages[3]["content"]
    assert "<INJECTED_CONTEXT>" in result.messages[-2]["content"]
    assert result.messages[-1] == {"role": "user", "content": "question"}
