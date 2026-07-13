"""Tests for strict, checkpoint-backed leveled-exercise generation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from src.graph.exercises import (
    ExerciseApprovalError,
    ExerciseArtifact,
    ExerciseArtifactWriteError,
    ExerciseContractError,
    ExerciseItem,
    ExerciseOutputValidationError,
    ExerciseReviewVerdict,
    _exercise_max_generation_rounds,
    _exercise_model_name,
    _exercise_temperature,
    exercise_agent,
    exercise_output,
    exercise_reviewer,
    should_rewrite_exercise,
    stable_exercise_question_id,
    validate_exercise_artifact,
)
from src.llm.structured_output import StructuredLLMResult, StructuredOutputError
from src.graph.resource_validation import ResourceValidationResultV1


LEVELS = ("basic", "intermediate", "application", "self_check")


def _exercise_item(level: str) -> ExerciseItem:
    return ExerciseItem(
        level=level,
        question_type="free_text",
        question=f"{level} question",
        choices=[],
        answer=f"{level} private answer",
        explanation=f"{level} private explanation",
        pitfall=f"{level} private pitfall",
        tags=["machine learning"],
    )


def _artifact() -> ExerciseArtifact:
    return ExerciseArtifact(
        title="Machine Learning Exercises",
        items=[_exercise_item(level) for level in LEVELS],
    )


def _source_items() -> list[dict]:
    items: list[dict] = []
    for item in _artifact().items:
        item_payload = item.model_dump(mode="python")
        item_payload["choices"] = tuple(item.choices)
        item_payload["tags"] = tuple(item.tags)
        items.append(
            {
                "question_id": stable_exercise_question_id(
                    level=item.level,
                    question_type=item.question_type,
                    question=item.question,
                    choices=item.choices,
                    tags=item.tags,
                ),
                **item_payload,
            }
        )
    return items


def _structured_failure(node_name: str, schema_name: str) -> StructuredLLMResult:
    return StructuredLLMResult(
        success=False,
        parsed=None,
        node_name=node_name,
        llm_node="exercise",
        schema_name=schema_name,
        provider="test",
        model="test",
        output_mode="deepseek_tool_call_strict",
        failure_phase="business_validation_error",
        error_type="BusinessValidationError",
        error_message="invalid exercise contract",
    )


def _document_artifact() -> dict[str, str]:
    return {
        "artifact_id": "exercise-1",
        "filename": "exercises.md",
        "docx_filename": "exercises.docx",
        "markdown_url": "/artifacts/exercises/exercise-1/exercises.md",
        "docx_url": "/artifacts/exercises/exercise-1/exercises.docx",
        "title": "Machine Learning Exercises",
    }


def _resource_validation(
    *, terminal_status: Literal["success", "failed"] = "success"
) -> ResourceValidationResultV1:
    failed = terminal_status == "failed"
    return ResourceValidationResultV1(
        schema_version="resource_validation_v1",
        resource_type="quiz",
        valid=not failed,
        terminal_status=terminal_status,
        renderable_count=0 if failed else 3,
        downloadable_count=0 if failed else 2,
        verified_local_count=0 if failed else 2,
        remote_unverified_count=0,
        failure_reason="quiz.no_renderable_artifact" if failed else "",
        warnings=(),
    )


def _output_state(**updates: object) -> dict:
    state: dict[str, object] = {
        "thread_id": "thread-1",
        "request_id": "request-1",
        "exercise_items": _source_items(),
        "exercise_artifact": {"title": "Machine Learning Exercises"},
        "exercise_review_verdict": "approve",
        "exercise_review_reason": "validated",
    }
    state.update(updates)
    return state


def test_exercise_schema_forbids_drift_and_requires_explicit_question_type() -> None:
    payload = _exercise_item("basic").model_dump(mode="python")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ExerciseItem.model_validate(payload, strict=True)

    payload.pop("unexpected")
    payload.pop("question_type")
    with pytest.raises(ValidationError):
        ExerciseItem.model_validate(payload, strict=True)

    with pytest.raises(ValidationError):
        ExerciseItem.model_validate(
            {
                **_exercise_item("basic").model_dump(mode="python"),
                "level": "foundation",
            },
            strict=True,
        )


def test_single_choice_requires_real_choices_and_exact_answer() -> None:
    with pytest.raises(ValidationError, match="at least two choices"):
        ExerciseItem(
            level="basic",
            question_type="single_choice",
            question="Choose one.",
            choices=["A"],
            answer="A",
            explanation="A is correct.",
            pitfall="Do not guess.",
            tags=["choice"],
        )
    with pytest.raises(ValidationError, match="exactly match one choice"):
        ExerciseItem(
            level="basic",
            question_type="single_choice",
            question="Choose one.",
            choices=["A", "B"],
            answer="C",
            explanation="A is correct.",
            pitfall="Do not guess.",
            tags=["choice"],
        )


def test_artifact_business_validation_requires_all_levels() -> None:
    artifact = ExerciseArtifact(
        title="Incomplete",
        items=[_exercise_item("basic") for _ in range(4)],
    )
    assert "missing required levels" in validate_exercise_artifact(artifact)


def test_review_verdict_forbids_extra_fields_and_blank_reason() -> None:
    with pytest.raises(ValidationError):
        ExerciseReviewVerdict.model_validate(
            {"verdict": "approve", "reason": "ok", "extra": True},
            strict=True,
        )
    with pytest.raises(ValidationError, match="must not be blank"):
        ExerciseReviewVerdict(verdict="approve", reason="   ")


@pytest.mark.anyio
async def test_exercise_agent_generates_private_source_items() -> None:
    artifact = _artifact()
    with patch(
        "src.graph.exercises.invoke_structured_llm",
        return_value=SimpleNamespace(success=True, parsed=artifact),
    ):
        result = await exercise_agent(
            {
                "messages": [HumanMessage(content="Create machine learning exercises")],
                "context": [{"content": "Course notes", "source": "ml.md"}],
                "exercise_outline": "basic, intermediate, application, self_check",
                "exercise_round": 0,
            }
        )

    assert result["exercise_artifact"]["title"] == "Machine Learning Exercises"
    assert result["exercise_round"] == 1
    assert {item["level"] for item in result["exercise_items"]} == set(LEVELS)
    assert all(
        item["question_type"] == "free_text" for item in result["exercise_items"]
    )
    assert all(
        item["question_id"].startswith("question:v1:")
        for item in result["exercise_items"]
    )
    assert len({item["question_id"] for item in result["exercise_items"]}) == 4


@pytest.mark.anyio
async def test_exercise_agent_blocks_failed_structured_result() -> None:
    with patch(
        "src.graph.exercises.invoke_structured_llm",
        return_value=_structured_failure("exercise_agent", "ExerciseArtifact"),
    ):
        with pytest.raises(StructuredOutputError):
            await exercise_agent(
                {
                    "messages": [HumanMessage(content="Create exercises")],
                    "context": [],
                    "exercise_outline": "strict outline",
                    "exercise_round": 0,
                }
            )


def test_stable_exercise_question_id_is_order_independent_for_tags() -> None:
    first = stable_exercise_question_id(
        level="basic",
        question_type="free_text",
        question="What is overfitting?",
        choices=[],
        tags=["machine learning", "generalization"],
    )
    second = stable_exercise_question_id(
        level="basic",
        question_type="free_text",
        question="What is overfitting?",
        choices=[],
        tags=["generalization", "machine learning"],
    )
    assert first == second
    assert first.startswith("question:v1:")


def test_stable_exercise_question_id_rejects_incomplete_identity() -> None:
    with pytest.raises(ValueError, match="requires a canonical level"):
        stable_exercise_question_id(
            level="basic",
            question_type="free_text",
            question="",
            choices=[],
            tags=["ml"],
        )


@pytest.mark.anyio
async def test_exercise_agent_empty_outline_raises() -> None:
    with pytest.raises(ValueError, match="outline"):
        await exercise_agent({"exercise_outline": "", "exercise_round": 0})


@pytest.mark.anyio
async def test_exercise_reviewer_rejects_incomplete_items() -> None:
    result = await exercise_reviewer(
        {
            "messages": [HumanMessage(content="Create exercises")],
            "exercise_outline": "all levels",
            "exercise_items": [_source_items()[0]],
        }
    )
    assert result["exercise_review_verdict"] == "reject"
    assert "missing required levels" in result["exercise_review_reason"]


@pytest.mark.anyio
async def test_exercise_reviewer_rejects_missing_required_field() -> None:
    items = _source_items()
    items[2].pop("explanation")
    result = await exercise_reviewer(
        {
            "messages": [HumanMessage(content="Create exercises")],
            "exercise_outline": "all levels",
            "exercise_items": items,
        }
    )
    assert result["exercise_review_verdict"] == "reject"
    assert "violates AssessmentQuizSourceItemV1" in result["exercise_review_reason"]


@pytest.mark.anyio
async def test_exercise_reviewer_blocks_failed_structured_result() -> None:
    with patch(
        "src.graph.exercises.invoke_structured_llm",
        return_value=_structured_failure("exercise_reviewer", "ExerciseReviewVerdict"),
    ):
        with pytest.raises(StructuredOutputError):
            await exercise_reviewer(
                {
                    "messages": [HumanMessage(content="Create exercises")],
                    "exercise_outline": "all levels",
                    "exercise_items": _source_items(),
                }
            )


@pytest.mark.anyio
async def test_exercise_output_is_public_and_checkpoint_keeps_private_answers() -> None:
    with (
        patch(
            "src.graph.exercises.create_document_artifact",
            return_value=_document_artifact(),
        ),
        patch(
            "src.graph.exercises.validate_renderable_resource_result",
            return_value=_resource_validation(),
        ),
    ):
        result = await exercise_output(_output_state())

    assert isinstance(result["messages"][0], AIMessage)
    public_surface = {
        "message": result["messages"][0].content,
        "exercise_items": result["exercise_items"],
        "exercise_artifact": result["exercise_artifact"],
        "exercise_resource_v3": result["exercise_resource_v3"],
    }
    public_json = json.dumps(public_surface, ensure_ascii=False, sort_keys=True)
    assert "private answer" not in public_json
    assert "private explanation" not in public_json
    assert "private pitfall" not in public_json
    assert '"answer"' not in public_json
    assert '"explanation"' not in public_json
    assert '"pitfall"' not in public_json
    assert all(
        set(item)
        == {
            "schema_version",
            "question_id",
            "question_type",
            "level",
            "question",
            "choices",
            "tags",
        }
        for item in result["exercise_items"]
    )

    checkpoint = result["assessment_checkpoint_resources"]
    assert checkpoint["thread_id"] == "thread-1"
    assert len(checkpoint["resources"]) == 1
    assert (
        checkpoint["resources"][0]["resource_id"]
        == result["exercise_artifact"]["resource_id"]
    )
    assert checkpoint["resources"][0]["questions"][0]["answer_key"][
        "accepted_answers"
    ] == ["basic private answer"]


@pytest.mark.anyio
async def test_exercise_output_real_markdown_and_docx_do_not_contain_answers(
    tmp_path,
    monkeypatch,
) -> None:
    from docx import Document

    monkeypatch.setenv("EXERCISE_ARTIFACT_DIR", str(tmp_path))

    result = await exercise_output(_output_state())

    artifact = result["exercise_artifact"]
    artifact_dir = tmp_path / artifact["artifact_id"]
    markdown = (artifact_dir / artifact["filename"]).read_text(encoding="utf-8")
    document = Document(artifact_dir / artifact["docx_filename"])
    docx_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    public_files = f"{markdown}\n{docx_text}"
    assert "private answer" not in public_files
    assert "private explanation" not in public_files
    assert "private pitfall" not in public_files
    assert result["exercise_resource_v3"]["validation"]["renderable_count"] == 3
    assert result["exercise_resource_v3"]["validation"]["verified_local_count"] == 2


@pytest.mark.anyio
async def test_exercise_output_writer_failure_is_typed_and_not_success() -> None:
    with patch(
        "src.graph.exercises.create_document_artifact",
        side_effect=OSError("disk unavailable"),
    ):
        with pytest.raises(ExerciseArtifactWriteError, match="generation failed"):
            await exercise_output(_output_state())


@pytest.mark.anyio
async def test_exercise_output_requires_complete_artifact_refs() -> None:
    artifact = _document_artifact()
    artifact.pop("docx_url")
    with patch(
        "src.graph.exercises.create_document_artifact",
        return_value=artifact,
    ):
        with pytest.raises(ExerciseArtifactWriteError, match="docx_url"):
            await exercise_output(_output_state())


@pytest.mark.anyio
async def test_exercise_output_blocks_failed_renderability_validation() -> None:
    with (
        patch(
            "src.graph.exercises.create_document_artifact",
            return_value=_document_artifact(),
        ),
        patch(
            "src.graph.exercises.validate_renderable_resource_result",
            return_value=_resource_validation(terminal_status="failed"),
        ),
    ):
        with pytest.raises(ExerciseOutputValidationError, match="renderability"):
            await exercise_output(_output_state())


@pytest.mark.anyio
@pytest.mark.parametrize("verdict", ["", "reject", "unknown", None])
async def test_exercise_output_rejects_every_non_approve_verdict(
    verdict: object,
) -> None:
    writer = AsyncMock()
    with patch("src.graph.exercises.create_document_artifact", writer):
        with pytest.raises(ExerciseApprovalError, match="requires approve"):
            await exercise_output(_output_state(exercise_review_verdict=verdict))
    writer.assert_not_awaited()


@pytest.mark.anyio
async def test_exercise_output_empty_artifact_raises() -> None:
    with pytest.raises(ExerciseContractError, match="non-empty list"):
        await exercise_output(_output_state(exercise_items=[]))


def test_should_rewrite_exercise_requires_known_verdict_and_caps_rounds() -> None:
    assert (
        should_rewrite_exercise(
            {"exercise_review_verdict": "approve", "exercise_round": 1}
        )
        == "output"
    )
    assert (
        should_rewrite_exercise(
            {"exercise_review_verdict": "reject", "exercise_round": 1}
        )
        == "rewrite"
    )
    with pytest.raises(RuntimeError, match="max rounds"):
        should_rewrite_exercise(
            {"exercise_review_verdict": "reject", "exercise_round": 3}
        )
    for verdict in ("", "unknown", None):
        with pytest.raises(ExerciseApprovalError, match="approve or reject"):
            should_rewrite_exercise(
                {"exercise_review_verdict": verdict, "exercise_round": 1}
            )


@pytest.mark.parametrize(
    ("helper", "setting_key"),
    [
        (_exercise_model_name, "llm.exercise.model"),
        (_exercise_temperature, "llm.exercise.temperature"),
        (_exercise_max_generation_rounds, "llm.exercise.max_generation_rounds"),
    ],
)
def test_exercise_runtime_config_has_no_missing_value_default(
    helper, setting_key
) -> None:
    with patch("src.graph.exercises.get_setting", return_value=None) as mocked:
        with pytest.raises(ValueError, match="explicitly configured"):
            helper()
    mocked.assert_called_once_with(setting_key, None)
