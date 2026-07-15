"""First-class structured QA agent and stable QA final contract."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal, Mapping, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from src.config import load_prompt
from src.context_engineering.policy_mode import resolve_context_runtime_policy
from src.context_engineering.workspace import sanitize_workspace_text, utc_now_iso
from src.graph.capability_registry import (
    build_safe_capability_context,
)
from src.graph.qa_suggestion_registry import (
    build_safe_qa_suggestion_registry,
    get_qa_suggestion_action,
    get_qa_suggestion_resource_types,
    qa_suggestion_validation_guidance,
)
from src.graph.state import LearningState
from src.llm.structured_output import (
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.a3_trace import emit_a3_trace
from src.tracing import traced_node

logger = logging.getLogger(__name__)

QAScope = Literal["academic", "general", "a3_agent"]
QAGroundingStatus = Literal[
    "judged_evidence",
    "general_knowledge",
    "capability_registry",
    "insufficient_evidence",
    "not_live_verified",
]


class QASuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=160)
    action: str = Field(min_length=1, max_length=80)
    resource_type: str = Field(default="", max_length=80)


class QAResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=16000)
    uncertainty_note: str = Field(default="", max_length=2000)
    grounding_status: QAGroundingStatus
    suggestions: list[QASuggestion] = Field(default_factory=list, max_length=3)


class QAFinalEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["qa_final"]
    schema_version: Literal[1]
    qa_id: str = Field(pattern=r"^qa:v1:[0-9a-f]{64}$")
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    qa_scope: QAScope
    response: QAResponse
    thread_id: str = Field(min_length=1, max_length=120)
    request_id: str = Field(min_length=1, max_length=120)
    created_at: str = Field(min_length=1, max_length=80)


def validate_qa_response(
    parsed: BaseModel,
    *,
    qa_scope: QAScope,
    kept_evidence_count: int,
    requires_live_verification: bool,
) -> str:
    """Validate grounding and registered suggestion actions for one QA scope."""
    if not isinstance(parsed, QAResponse):
        return "root expected QAResponse"
    if not parsed.answer.strip():
        return "answer must not be empty"
    if len(parsed.suggestions) > 3:
        return "suggestions must contain at most 3 items"

    registered_resources = set(get_qa_suggestion_resource_types())
    for index, suggestion in enumerate(parsed.suggestions):
        if not suggestion.label.strip():
            return f"suggestions[{index}].label must not be blank"
        action = get_qa_suggestion_action(suggestion.action)
        if action is None:
            return (
                f"suggestions[{index}].action is invalid; "
                f"{qa_suggestion_validation_guidance()}"
            )
        resource_type = suggestion.resource_type
        if action.requires_resource_type:
            if resource_type not in registered_resources:
                return (
                    f"suggestions[{index}].resource_type is invalid; "
                    f"{qa_suggestion_validation_guidance()}"
                )
        elif resource_type:
            return (
                f"suggestions[{index}].resource_type is not allowed for "
                f"action={action.action_id}; {qa_suggestion_validation_guidance()}"
            )

    if qa_scope == "academic":
        if kept_evidence_count > 0:
            if parsed.grounding_status != "judged_evidence":
                return (
                    "academic QA with kept evidence requires judged_evidence grounding"
                )
        else:
            if parsed.grounding_status != "insufficient_evidence":
                return (
                    "academic QA without kept evidence requires insufficient_evidence"
                )
            if not parsed.uncertainty_note.strip():
                return "insufficient academic evidence requires uncertainty_note"
    elif qa_scope == "general":
        expected = (
            "not_live_verified" if requires_live_verification else "general_knowledge"
        )
        if parsed.grounding_status != expected:
            return f"general QA requires grounding_status={expected}"
        if requires_live_verification and not parsed.uncertainty_note.strip():
            return "unverified live information requires uncertainty_note"
    elif qa_scope == "a3_agent":
        if parsed.grounding_status != "capability_registry":
            return "A3 capability QA requires capability_registry grounding"
    else:
        return "qa_scope is invalid"
    return ""


@traced_node
async def qa_agent(state: LearningState) -> dict:
    """Answer academic, general, or A3 capability questions through one contract."""
    if state.get("response_mode") != "qa":
        raise ValueError("qa_agent requires response_mode=qa")
    qa_scope = str(state.get("qa_scope") or "").strip()
    if qa_scope not in {"academic", "general", "a3_agent"}:
        raise ValueError("qa_agent requires a valid qa_scope")
    typed_scope = cast(QAScope, qa_scope)
    requires_live_verification = state.get("requires_live_verification") is True
    kept_evidence = [
        item for item in (state.get("graded_evidence") or []) if isinstance(item, dict)
    ]
    kept_evidence_count = len(kept_evidence)
    question = _last_human_query(state)
    if not question:
        raise ValueError("qa_agent requires a current user question")

    messages: list[Any] = [
        SystemMessage(content=load_prompt("qa_agent")),
        SystemMessage(content=build_safe_qa_suggestion_registry()),
    ]
    capability_context_present = typed_scope == "a3_agent"
    if capability_context_present:
        runtime_metadata = state.get("runtime_capability_metadata")
        if not isinstance(runtime_metadata, Mapping):
            raise ValueError("A3 capability QA requires runtime capability metadata")
        runtime_policy = resolve_context_runtime_policy()
        messages.append(
            SystemMessage(
                content=build_safe_capability_context(
                    context_policy_mode=runtime_policy.mode,
                    runtime_metadata=runtime_metadata,
                )
            )
        )
    messages.append(
        HumanMessage(
            content=_qa_request_payload(
                question=question,
                qa_scope=typed_scope,
                kept_evidence_count=kept_evidence_count,
                requires_live_verification=requires_live_verification,
            )
        )
    )

    result = await invoke_structured_llm(
        node_name="qa_agent",
        llm_node="qa_agent",
        schema=QAResponse,
        messages=messages,
        output_mode=get_llm_output_mode("qa_agent"),
        business_validator=lambda parsed: validate_qa_response(
            parsed,
            qa_scope=typed_scope,
            kept_evidence_count=kept_evidence_count,
            requires_live_verification=requires_live_verification,
        ),
        state=state,
        max_raw_chars=get_max_raw_chars("qa_agent"),
    )
    parsed = result.parsed
    if not isinstance(parsed, QAResponse):
        raise TypeError("qa_agent parsed result is not QAResponse")

    final_payload = build_qa_final_payload(
        response=parsed,
        qa_scope=typed_scope,
        thread_id=str(state.get("thread_id") or state.get("session_id") or ""),
        request_id=str(state.get("request_id") or ""),
    )
    emit_a3_trace(
        logger,
        "qa_final.prepared",
        qa_final_trace_payload(final_payload),
        state=state,
        env_flag="LOG_A3_TRACE",
    )
    return {
        "messages": [AIMessage(content=_render_qa_message(parsed))],
        "last_qa_response": final_payload,
        "final_response_type": "qa",
    }


def build_qa_final_payload(
    *,
    response: QAResponse,
    qa_scope: QAScope,
    thread_id: str,
    request_id: str,
) -> dict[str, Any]:
    """Build a bounded, deterministic payload for SSE and checkpoint restore."""
    normalized_thread_id = sanitize_workspace_text(thread_id, max_chars=120)
    normalized_request_id = sanitize_workspace_text(request_id, max_chars=120)
    if not normalized_thread_id or not normalized_request_id:
        raise ValueError("qa_final requires explicit thread_id and request_id")
    safe_response = {
        "answer": sanitize_workspace_text(
            response.answer,
            max_chars=6000,
        ),
        "uncertainty_note": sanitize_workspace_text(
            response.uncertainty_note,
            max_chars=1000,
            fallback="",
        ),
        "grounding_status": response.grounding_status,
        "suggestions": [
            {
                "label": sanitize_workspace_text(
                    suggestion.label,
                    max_chars=160,
                ),
                "action": suggestion.action,
                "resource_type": suggestion.resource_type,
            }
            for suggestion in response.suggestions[:3]
        ],
    }
    validated_response = QAResponse.model_validate(safe_response)
    payload_hash = _stable_qa_payload_hash(validated_response)
    qa_id = _stable_qa_id(
        thread_id=normalized_thread_id,
        request_id=normalized_request_id,
        qa_scope=qa_scope,
        payload_hash=payload_hash,
    )
    return QAFinalEvent(
        type="qa_final",
        schema_version=1,
        qa_id=qa_id,
        payload_hash=payload_hash,
        qa_scope=qa_scope,
        response=validated_response,
        thread_id=normalized_thread_id,
        request_id=normalized_request_id,
        created_at=utc_now_iso(),
    ).model_dump(mode="json")


def build_general_qa_node_output(
    *,
    answer: str,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a non-structured node answer into the shared QA final contract."""

    response = QAResponse.model_validate(
        {
            "answer": answer,
            "uncertainty_note": "",
            "grounding_status": "general_knowledge",
            "suggestions": [],
        }
    )
    business_error = validate_qa_response(
        response,
        qa_scope="general",
        kept_evidence_count=0,
        requires_live_verification=False,
    )
    if business_error:
        raise ValueError(f"QAResponse business validation failed: {business_error}")

    thread_id = str(state.get("thread_id") or state.get("session_id") or "").strip()
    request_id = str(state.get("request_id") or "").strip()
    if not thread_id or not request_id:
        raise ValueError("qa_final requires explicit thread_id and request_id")
    final_payload = build_qa_final_payload(
        response=response,
        qa_scope="general",
        thread_id=thread_id,
        request_id=request_id,
    )
    return {
        "messages": [AIMessage(content=_render_qa_message(response))],
        "last_qa_response": final_payload,
        "final_response_type": "qa",
    }


def qa_final_payload(state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the current request's strictly bound QA final event, if present."""

    payload = state.get("last_qa_response")
    if payload in ({}, None):
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("qa_final payload must be an object")

    runtime_request_id = str(state.get("request_id") or "").strip()
    if not runtime_request_id:
        raise ValueError("qa_final requires runtime request_id")
    payload_request_id = payload.get("request_id")
    if isinstance(payload_request_id, str) and payload_request_id != runtime_request_id:
        # A durable final from an earlier request is history, not this request's
        # authoritative terminal. Malformed current-request payloads still fail closed.
        return None

    validated = validate_qa_final_event(payload)
    runtime_thread_id = str(state.get("thread_id") or "").strip()
    if not runtime_thread_id:
        raise ValueError("qa_final requires runtime thread_id")
    if validated.thread_id != runtime_thread_id:
        raise ValueError("qa_final thread_id does not match runtime state")
    if validated.request_id != runtime_request_id:
        raise ValueError("qa_final request_id does not match runtime state")

    runtime_scope = str(state.get("qa_scope") or "").strip()
    if runtime_scope and validated.qa_scope != runtime_scope:
        raise ValueError("qa_final qa_scope does not match runtime state")
    kept_evidence_count = sum(
        1 for item in (state.get("graded_evidence") or []) if isinstance(item, Mapping)
    )
    business_error = validate_qa_response(
        validated.response,
        qa_scope=validated.qa_scope,
        kept_evidence_count=kept_evidence_count,
        requires_live_verification=state.get("requires_live_verification") is True,
    )
    if business_error:
        raise ValueError(f"qa_final business validation failed: {business_error}")
    return validated.model_dump(mode="json")


def validate_qa_final_event(value: object) -> QAFinalEvent:
    """Validate QA Final identity and content hashes without accepting drift."""

    if isinstance(value, QAFinalEvent):
        payload = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        _reject_non_json_qa_sequences(value)
        payload = dict(value)
    else:
        raise TypeError("qa_final payload must be an object")
    validated = QAFinalEvent.model_validate_json(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        strict=True,
    )
    expected_hash = _stable_qa_payload_hash(validated.response)
    if validated.payload_hash != expected_hash:
        raise ValueError("qa_final payload_hash mismatch")
    expected_id = _stable_qa_id(
        thread_id=validated.thread_id,
        request_id=validated.request_id,
        qa_scope=validated.qa_scope,
        payload_hash=expected_hash,
    )
    if validated.qa_id != expected_id:
        raise ValueError("qa_final qa_id mismatch")
    return validated


def _stable_qa_payload_hash(response: QAResponse) -> str:
    return hashlib.sha256(
        json.dumps(
            response.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _stable_qa_id(
    *,
    thread_id: str,
    request_id: str,
    qa_scope: QAScope,
    payload_hash: str,
) -> str:
    stable_input = "\x1f".join(("qa:v1", thread_id, request_id, qa_scope, payload_hash))
    return "qa:v1:" + hashlib.sha256(stable_input.encode("utf-8")).hexdigest()


def _reject_non_json_qa_sequences(value: object, *, path: str = "root") -> None:
    if isinstance(value, tuple):
        raise TypeError(f"{path} must use a JSON array, not a Python tuple")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_json_qa_sequences(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} object keys must be strings")
            _reject_non_json_qa_sequences(item, path=f"{path}.{key}")


def qa_final_trace_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    response_value = payload.get("response")
    response: dict[str, Any] = (
        response_value if isinstance(response_value, dict) else {}
    )
    return {
        "qa_id": sanitize_workspace_text(payload.get("qa_id"), max_chars=80),
        "payload_hash": sanitize_workspace_text(
            payload.get("payload_hash"),
            max_chars=80,
        ),
        "qa_scope": sanitize_workspace_text(payload.get("qa_scope"), max_chars=40),
        "thread_id": sanitize_workspace_text(payload.get("thread_id"), max_chars=120),
        "request_id": sanitize_workspace_text(payload.get("request_id"), max_chars=120),
        "grounding_status": sanitize_workspace_text(
            response.get("grounding_status"),
            max_chars=60,
        ),
        "answer_chars": len(str(response.get("answer") or "")),
        "uncertainty_note_present": bool(response.get("uncertainty_note")),
        "suggestion_count": len(response.get("suggestions") or []),
    }


def _qa_request_payload(
    *,
    question: str,
    qa_scope: QAScope,
    kept_evidence_count: int,
    requires_live_verification: bool,
) -> str:
    return json.dumps(
        {
            "question": question,
            "qa_scope": qa_scope,
            "kept_evidence_count": kept_evidence_count,
            "requires_live_verification": requires_live_verification,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _last_human_query(state: Mapping[str, Any]) -> str:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
        if isinstance(message, dict) and str(message.get("role") or "") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _render_qa_message(response: QAResponse) -> str:
    answer = response.answer.strip()
    note = response.uncertainty_note.strip()
    return f"{answer}\n\n{note}" if note else answer
