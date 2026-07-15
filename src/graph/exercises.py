"""Collaborative leveled-exercise resource-generation nodes."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.assessment.attempt_contracts import (
    AssessmentLearningGuidanceBindingV1,
    AssessmentQuizSourceItemV1,
    PublicExerciseCardV1,
)
from src.assessment.checkpoint import merge_assessment_checkpoint_resource_v2
from src.assessment.identity import stable_exercise_question_id
from src.config import get_setting, load_prompt
from src.config.evidence_orchestration_contracts import (
    RESOURCE_EVIDENCE_CONTRACT_VERSION,
    ResourceEvidenceAssignment,
)
from src.graph.assessment_quiz import (
    build_assessment_quiz_projection_v1,
    build_public_exercise_cards_v1,
)
from src.graph.llm import invoke_plain_llm_fail_fast
from src.graph.resource_final_v3 import ResourceFinalV3ResourceValidation
from src.graph.resource_validation import validate_renderable_resource_result
from src.graph.state import LearningState
from src.llm.structured_output import (
    StructuredOutputError,
    get_llm_output_mode,
    get_max_raw_chars,
    invoke_structured_llm,
)
from src.observability.a3_trace import emit_a3_trace
from src.tools.document_tool import create_document_artifact
from src.tracing import traced_llm_call, traced_node

logger = logging.getLogger(__name__)

ExerciseLevel = Literal["basic", "intermediate", "application", "self_check"]
ExerciseQuestionType = Literal["free_text", "single_choice"]
REQUIRED_LEVELS = frozenset({"basic", "intermediate", "application", "self_check"})


class ExerciseContractError(ValueError):
    """Raised when generated or checkpointed exercise state violates its contract."""


class ExerciseApprovalError(RuntimeError):
    """Raised when an unapproved exercise attempts to reach output."""


class ExerciseArtifactWriteError(RuntimeError):
    """Raised when the public exercise document cannot be durably written."""


class ExerciseOutputValidationError(RuntimeError):
    """Raised when written exercise artifacts fail final renderability checks."""


def _assessment_learning_guidance_binding(
    state: LearningState,
) -> AssessmentLearningGuidanceBindingV1 | None:
    """Capture an exact candidate assignment; legacy quiz flows remain unbound."""

    raw_assignment = state.get("resource_evidence_assignment")
    raw_contract_version = state.get("resource_evidence_contract_version")
    if raw_assignment in (None, {}) and raw_contract_version in (None, ""):
        return None
    if not isinstance(raw_assignment, Mapping):
        raise ExerciseContractError(
            "quiz guidance binding requires a resource evidence assignment"
        )
    if raw_contract_version != RESOURCE_EVIDENCE_CONTRACT_VERSION:
        raise ExerciseContractError(
            "quiz guidance binding has an invalid assignment contract version"
        )
    try:
        assignment = ResourceEvidenceAssignment.model_validate(raw_assignment)
    except ValidationError as exc:
        raise ExerciseContractError(
            "quiz guidance binding violates ResourceEvidenceAssignment"
        ) from exc
    if (
        assignment.resource_type != "quiz"
        or len(assignment.subjects) != 1
        or len(assignment.topic_ids) != 1
    ):
        raise ExerciseContractError(
            "quiz guidance binding requires one exact subject and topic"
        )
    user_id = state.get("user_id")
    request_id = state.get("request_id")
    subject = state.get("subject")
    if (
        not isinstance(user_id, str)
        or not user_id.strip()
        or user_id != user_id.strip()
    ):
        raise ExerciseContractError(
            "quiz guidance binding requires an explicit normalized user_id"
        )
    if (
        not isinstance(request_id, str)
        or not request_id.strip()
        or request_id != request_id.strip()
    ):
        raise ExerciseContractError(
            "quiz guidance binding requires an explicit generation request_id"
        )
    if subject != assignment.subjects[0]:
        raise ExerciseContractError(
            "quiz guidance binding subject differs from its evidence assignment"
        )
    return AssessmentLearningGuidanceBindingV1(
        schema_version="assessment_learning_guidance_binding_v1",
        user_id=user_id,
        subject=assignment.subjects[0],
        topic_id=assignment.topic_ids[0],
        resource_type="quiz",
        generation_request_id=request_id,
        assignment_contract_version=RESOURCE_EVIDENCE_CONTRACT_VERSION,
        assignment_fingerprint=assignment.assignment_fingerprint,
    )


class ExerciseItem(BaseModel):
    """A single exercise item with answer and teaching feedback."""

    model_config = ConfigDict(extra="forbid", strict=True)

    level: ExerciseLevel
    question_type: ExerciseQuestionType
    question: str = Field(..., min_length=1)
    choices: list[str] = Field(..., max_length=20)
    answer: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)
    pitfall: str = Field(..., min_length=1)
    tags: list[str] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_item(self) -> ExerciseItem:
        text_values = (self.question, self.answer, self.explanation, self.pitfall)
        if any(not value.strip() for value in text_values):
            raise ValueError("exercise item text fields must not be blank")
        if any(not value.strip() for value in (*self.choices, *self.tags)):
            raise ValueError("exercise choices and tags must not contain blanks")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("exercise choices must be unique")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("exercise tags must be unique")
        if self.question_type == "free_text" and self.choices:
            raise ValueError("free_text exercise items must not define choices")
        if self.question_type == "single_choice":
            if len(self.choices) < 2:
                raise ValueError(
                    "single_choice exercise items require at least two choices"
                )
            if self.answer not in self.choices:
                raise ValueError(
                    "single_choice exercise answer must exactly match one choice"
                )
        return self


class ExerciseArtifact(BaseModel):
    """Structured exercise resource produced by exercise_agent."""

    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(..., min_length=1)
    items: list[ExerciseItem] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_title(self) -> ExerciseArtifact:
        if not self.title.strip():
            raise ValueError("exercise title must not be blank")
        return self


class ExerciseReviewVerdict(BaseModel):
    """Structured quality gate output for exercise_reviewer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["approve", "reject"]
    reason: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_reason(self) -> ExerciseReviewVerdict:
        if not self.reason.strip():
            raise ValueError("exercise review reason must not be blank")
        return self


def validate_exercise_artifact(parsed: BaseModel) -> str:
    if not isinstance(parsed, ExerciseArtifact):
        return "root expected ExerciseArtifact"
    if len(parsed.items) < 4:
        return f"items expected at least 4, got {len(parsed.items)}"
    levels = {item.level for item in parsed.items}
    missing_levels = sorted(REQUIRED_LEVELS - levels)
    if missing_levels:
        return f"items missing required levels: {', '.join(missing_levels)}"
    questions = [item.question.strip() for item in parsed.items]
    if len(questions) != len(set(questions)):
        return "duplicate questions detected"
    return ""


def validate_review_verdict(parsed: BaseModel) -> str:
    if not isinstance(parsed, ExerciseReviewVerdict):
        return "root expected ExerciseReviewVerdict"
    if parsed.verdict not in {"approve", "reject"}:
        return "verdict must be approve or reject"
    if not str(parsed.reason or "").strip():
        return "reason must be non-empty"
    return ""


def _last_human_query(state: LearningState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _format_keypoints(state: LearningState) -> str:
    keypoints = state.get("keypoints", [])
    return (
        ", ".join(str(item) for item in keypoints if str(item).strip())
        or "No explicit keypoints."
    )


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No judged evidence is available. Do not invent citations."
    parts: list[str] = []
    for idx, item in enumerate(context[:8], 1):
        source = (
            item.get("source")
            or item.get("title")
            or item.get("url")
            or "learning material"
        )
        content = str(
            item.get("content") or item.get("snippet") or item.get("text") or ""
        )[:800]
        if content:
            parts.append(f"[{idx}] Source: {source}\n{content}")
    return "\n\n".join(parts) or "Judged evidence has no readable body."


def _render_prompt(prompt_name: str, replacements: dict[str, str]) -> str:
    prompt = load_prompt(prompt_name)
    for key, value in replacements.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt


def _is_web_evidence(item: dict) -> bool:
    return item.get("source_type") == "web" or item.get("type") == "web_evidence"


def _web_evidence_items(context: list[dict]) -> list[dict]:
    return [item for item in context if _is_web_evidence(item)]


def _assessment_source_items(
    artifact: ExerciseArtifact,
) -> tuple[AssessmentQuizSourceItemV1, ...]:
    return tuple(
        AssessmentQuizSourceItemV1(
            question_id=stable_exercise_question_id(
                level=item.level,
                question_type=item.question_type,
                question=item.question,
                choices=item.choices,
                tags=item.tags,
            ),
            question_type=item.question_type,
            level=item.level,
            question=item.question,
            choices=tuple(item.choices),
            answer=item.answer,
            explanation=item.explanation,
            pitfall=item.pitfall,
            tags=tuple(item.tags),
        )
        for item in artifact.items
    )


def _validated_state_source_items(
    items: object,
) -> tuple[AssessmentQuizSourceItemV1, ...]:
    if not isinstance(items, list) or not items:
        raise ExerciseContractError("exercise items must be a non-empty list")
    parsed: list[AssessmentQuizSourceItemV1] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ExerciseContractError(f"exercise item {index + 1} must be an object")
        try:
            parsed.append(
                AssessmentQuizSourceItemV1.model_validate_json(
                    json.dumps(
                        dict(item),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    strict=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ExerciseContractError(
                f"exercise item {index + 1} violates AssessmentQuizSourceItemV1"
            ) from exc
    question_ids = [item.question_id for item in parsed]
    if len(question_ids) != len(set(question_ids)):
        raise ExerciseContractError("exercise question_id values must be unique")
    levels = {item.level for item in parsed}
    missing_levels = sorted(REQUIRED_LEVELS - levels)
    if missing_levels:
        raise ExerciseContractError(
            f"exercise items missing required levels: {', '.join(missing_levels)}"
        )
    return tuple(parsed)


def _local_review_failure(
    items: Sequence[AssessmentQuizSourceItemV1], _query: str
) -> str:
    if len(items) < 4:
        return f"exercise item count is too low: {len(items)}"
    levels = {item.level for item in items}
    missing_levels = sorted(REQUIRED_LEVELS - levels)
    if missing_levels:
        return f"exercise items missing required levels: {', '.join(missing_levels)}"
    return ""


def _render_exercise_markdown(
    title: str,
    items: list[dict],
) -> str:
    if not title.strip():
        raise ExerciseContractError("public exercise title must not be blank")
    lines = [f"## {title}", ""]
    for idx, item in enumerate(items, 1):
        try:
            card = PublicExerciseCardV1.model_validate_json(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                strict=True,
            )
        except (TypeError, ValueError) as exc:
            raise ExerciseContractError(
                f"public exercise item {idx} violates PublicExerciseCardV1"
            ) from exc
        tags = ", ".join(card.tags)
        lines.extend(
            [
                f"### {idx}. {card.level}",
                f"**Question:** {card.question}",
                f"**Question type:** {card.question_type}",
            ]
        )
        if card.choices:
            lines.extend(f"- {choice}" for choice in card.choices)
        if tags:
            lines.append(f"**Tags:** {tags}")
        lines.append("")
    return "\n".join(lines).strip()


def _exercise_model_name() -> str:
    configured_model = get_setting("llm.exercise.model", None)
    if not isinstance(configured_model, str) or not configured_model.strip():
        raise ValueError("llm.exercise.model must be explicitly configured")
    return configured_model.strip()


def _exercise_temperature() -> float:
    configured_temperature = get_setting("llm.exercise.temperature", None)
    if isinstance(configured_temperature, bool) or not isinstance(
        configured_temperature, (int, float)
    ):
        raise ValueError("llm.exercise.temperature must be explicitly configured")
    temperature = float(configured_temperature)
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("llm.exercise.temperature must be between 0 and 2")
    return temperature


def _exercise_max_generation_rounds() -> int:
    configured_rounds = get_setting("llm.exercise.max_generation_rounds", None)
    if isinstance(configured_rounds, bool) or not isinstance(configured_rounds, int):
        raise ValueError(
            "llm.exercise.max_generation_rounds must be explicitly configured"
        )
    if configured_rounds < 1:
        raise ValueError("llm.exercise.max_generation_rounds must be positive")
    return configured_rounds


def _exercise_round(state: Mapping[str, object]) -> int:
    value = state.get("exercise_round")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExerciseContractError("exercise_round must be a non-negative integer")
    return value


def _exercise_artifact_refs(artifact: Mapping[str, object]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for key in ("markdown_url", "docx_url"):
        value = artifact.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ExerciseArtifactWriteError(
                f"exercise document artifact missing required {key}"
            )
        refs[key] = value.strip()
    return refs


@traced_node
async def exercise_planner(state: LearningState) -> dict:
    query = _last_human_query(state)
    context = state.get("context", [])
    web_evidence = _web_evidence_items(context)
    emit_a3_trace(
        logger,
        "exercise_planner",
        {
            "context_count": len(context),
            "context_web_count": len(web_evidence),
            "dual_source_mode": bool(state.get("dual_source_mode")),
            "evidence_judge_state": state.get("evidence_judge_state", ""),
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )
    prompt = _render_prompt(
        "exercise_planner",
        {
            "question": query,
            "keypoints": _format_keypoints(state),
            "context": _format_context(context),
        },
    )
    outline = await invoke_plain_llm_fail_fast(
        node_name="exercise_planner",
        llm_node="exercise",
        messages=[
            SystemMessage(
                content="You are a university course exercise planner. Return a concrete exercise outline only."
            ),
            HumanMessage(content=prompt),
        ],
        state=state,
        temperature=_exercise_temperature(),
    )
    if not outline.strip():
        raise ValueError("exercise_planner produced empty outline")
    return {
        "exercise_outline": outline,
        "exercise_items": [],
        "exercise_artifact": {},
        "exercise_review_verdict": "",
        "exercise_review_reason": "",
        "exercise_revision_notes": "",
        "exercise_round": 0,
    }


@traced_node
async def exercise_agent(state: LearningState) -> dict:
    outline = state.get("exercise_outline", "")
    if not outline.strip():
        raise ValueError("exercise outline is empty")
    round_no = _exercise_round(state) + 1
    prompt = _render_prompt(
        "exercise_agent",
        {
            "question": _last_human_query(state),
            "keypoints": _format_keypoints(state),
            "context": _format_context(state.get("context", [])),
            "exercise_outline": outline,
            "revision_notes": state.get("exercise_revision_notes", "") or "None",
        },
    )
    model_name = _exercise_model_name()
    with traced_llm_call(
        model_name=model_name,
        node_name="exercise_agent",
        temperature=_exercise_temperature(),
    ):
        structured_result = await invoke_structured_llm(
            node_name="exercise_agent",
            llm_node="exercise",
            schema=ExerciseArtifact,
            messages=[
                SystemMessage(
                    content="You are a leveled exercise generator. Return only JSON for ExerciseArtifact."
                ),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode("exercise_agent"),
            business_validator=validate_exercise_artifact,
            state=state,
            max_raw_chars=get_max_raw_chars("exercise_agent"),
        )
    if not structured_result.success:
        raise StructuredOutputError(structured_result)
    result = structured_result.parsed
    if not isinstance(result, ExerciseArtifact):
        raise TypeError("exercise_agent parsed result is not ExerciseArtifact")
    source_items = _assessment_source_items(result)
    return {
        "exercise_items": [item.model_dump(mode="json") for item in source_items],
        "exercise_artifact": {"title": result.title.strip()},
        "exercise_round": round_no,
        "exercise_review_verdict": "",
        "exercise_review_reason": "",
    }


@traced_node
async def exercise_reviewer(state: LearningState) -> dict:
    try:
        source_items = _validated_state_source_items(state.get("exercise_items"))
    except ExerciseContractError as exc:
        return {
            "exercise_review_verdict": "reject",
            "exercise_review_reason": str(exc),
            "exercise_revision_notes": f"Please rewrite: {exc}",
        }
    local_failure = _local_review_failure(source_items, _last_human_query(state))
    if local_failure:
        return {
            "exercise_review_verdict": "reject",
            "exercise_review_reason": local_failure,
            "exercise_revision_notes": f"Please rewrite: {local_failure}",
        }
    prompt = _render_prompt(
        "exercise_reviewer",
        {
            "question": _last_human_query(state),
            "exercise_outline": state.get("exercise_outline", ""),
            "exercise_items": json.dumps(
                [item.model_dump(mode="json") for item in source_items],
                ensure_ascii=False,
            ),
        },
    )
    model_name = _exercise_model_name()
    with traced_llm_call(
        model_name=model_name, node_name="exercise_reviewer", temperature=0.0
    ):
        structured_result = await invoke_structured_llm(
            node_name="exercise_reviewer",
            llm_node="exercise",
            schema=ExerciseReviewVerdict,
            messages=[
                SystemMessage(
                    content="You are a course exercise quality reviewer. Return only JSON."
                ),
                HumanMessage(content=prompt),
            ],
            output_mode=get_llm_output_mode("exercise_reviewer"),
            business_validator=validate_review_verdict,
            state=state,
            max_raw_chars=get_max_raw_chars("exercise_reviewer"),
        )
    if not structured_result.success:
        raise StructuredOutputError(structured_result)
    result = structured_result.parsed
    if not isinstance(result, ExerciseReviewVerdict):
        raise TypeError("exercise_reviewer parsed result is not ExerciseReviewVerdict")
    return {
        "exercise_review_verdict": result.verdict,
        "exercise_review_reason": result.reason.strip(),
        "exercise_revision_notes": ""
        if result.verdict == "approve"
        else f"Please rewrite: {result.reason.strip()}",
    }


@traced_node
async def exercise_rewrite(state: LearningState) -> dict:
    reason = state.get("exercise_review_reason", "")
    if not reason.strip():
        raise ValueError("exercise rewrite requested without review reason")
    return {
        "exercise_revision_notes": f"Revise the exercise artifact according to reviewer feedback:\n{reason}",
        "exercise_outline": state.get("exercise_outline", ""),
    }


@traced_node
async def exercise_output(state: LearningState) -> dict:
    verdict = state.get("exercise_review_verdict")
    if verdict != "approve":
        raise ExerciseApprovalError(
            f"exercise output requires approve verdict, got {verdict!r}"
        )
    source_items = _validated_state_source_items(state.get("exercise_items"))
    raw_artifact = state.get("exercise_artifact")
    if not isinstance(raw_artifact, Mapping):
        raise ExerciseContractError("exercise_artifact must be an object")
    raw_title = raw_artifact.get("title")
    if not isinstance(raw_title, str) or not raw_title.strip():
        raise ExerciseContractError("exercise artifact title must not be blank")
    title = raw_title.strip()
    public_cards = build_public_exercise_cards_v1(source_items)
    public_items = [card.model_dump(mode="json") for card in public_cards]
    content = _render_exercise_markdown(title, public_items)
    try:
        artifact = create_document_artifact(
            markdown_text=content,
            title=title,
            artifact_kind="exercises",
        )
    except Exception as exc:
        raise ExerciseArtifactWriteError(
            "exercise document artifact generation failed"
        ) from exc
    artifact_refs = _exercise_artifact_refs(artifact)
    validation_artifact = {
        **artifact,
        "schema_version": "exercise_public_artifact_v1",
        "title": title,
        "items": public_items,
    }
    branch_validation = validate_renderable_resource_result(
        "quiz",
        validation_artifact,
        (),
        {"exercise_items": public_items},
    )
    if not branch_validation.valid or branch_validation.terminal_status != "success":
        raise ExerciseOutputValidationError(
            "exercise document failed strict renderability validation: "
            f"{branch_validation.failure_reason or branch_validation.terminal_status}"
        )
    try:
        v3_validation = ResourceFinalV3ResourceValidation.model_validate(
            branch_validation.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as exc:
        raise ExerciseOutputValidationError(
            "exercise validation cannot satisfy Resource Final V3"
        ) from exc

    thread_id = state.get("thread_id")
    request_id = state.get("request_id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ExerciseContractError("exercise output requires a non-blank thread_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ExerciseContractError("exercise output requires a non-blank request_id")
    projection = build_assessment_quiz_projection_v1(
        thread_id=thread_id,
        request_id=request_id,
        title=title,
        summary=f"{len(public_items)} validated exercise cards.",
        source_items=source_items,
        artifact_refs=artifact_refs,
        validation=v3_validation,
        learning_guidance_binding=_assessment_learning_guidance_binding(state),
    )
    checkpoint = merge_assessment_checkpoint_resource_v2(
        thread_id=thread_id,
        existing=state.get("assessment_checkpoint_resources"),
        resource=projection.checkpoint_resource,
    )

    emit_a3_trace(
        logger,
        "exercise_output",
        {
            "item_count": len(public_items),
            "markdown_chars": len(content),
            "resource_id": projection.public_resource.resource_id,
            "has_document_artifact": True,
        },
        state=state,
        env_flag="LOG_GENERATION_SUMMARY",
    )

    return {
        "exercise_items": public_items,
        "exercise_artifact": {
            **validation_artifact,
            "resource_id": projection.public_resource.resource_id,
            "payload_hash": projection.public_resource.payload_hash,
        },
        "exercise_resource_v3": projection.public_resource.model_dump(mode="json"),
        "assessment_checkpoint_resources": checkpoint.model_dump(mode="json"),
        "messages": [AIMessage(content=content)],
    }


def should_rewrite_exercise(state: Mapping[str, object]) -> str:
    verdict = state.get("exercise_review_verdict")
    if verdict == "approve":
        return "output"
    if verdict != "reject":
        raise ExerciseApprovalError(
            f"exercise review verdict must be approve or reject, got {verdict!r}"
        )
    max_rounds = _exercise_max_generation_rounds()
    current_round = _exercise_round(state)
    if current_round < max_rounds:
        return "rewrite"
    raise RuntimeError(
        f"exercise rejected after max rounds: {state.get('exercise_review_reason', '')}"
    )
