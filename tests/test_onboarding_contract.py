"""Strict public-to-internal onboarding contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

from pydantic import ValidationError
import pytest

from src.schemas import (
    CompiledOnboardRequestV2,
    OnboardRequest,
    compile_onboard_request_v2,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "onboard_v2",
        "profile": {
            "schema_version": "learning_guidance_profile_write_request_v1",
            "request_id": "onboard-request-1",
            "user_id": "user-1",
            "skills": [
                {
                    "subject": "math",
                    "topic_id": "math.algebra",
                    "level": 0.3,
                    "confidence": 0.8,
                }
            ],
            "goals": [
                {
                    "subject": "math",
                    "topic_id": "math.algebra",
                    "goal": "Learn algebra",
                    "importance": 0.9,
                    "progress": 0.2,
                }
            ],
            "preferences": [
                {
                    "subject": "math",
                    "topic_id": "math.algebra",
                    "dimension": "prefer_visual",
                    "strength": 0.8,
                }
            ],
        },
        "nickname": "learner",
        "grade": "grade-10",
        "dislikes": ["rote repetition"],
    }


@pytest.mark.parametrize(
    "invalid_sequence",
    [
        ("rote repetition",),
        {"rote repetition"},
        (item for item in ["rote repetition"]),
        "rote repetition",
    ],
    ids=["tuple", "set", "generator", "string"],
)
def test_public_onboarding_rejects_non_list_dislikes(
    invalid_sequence: object,
) -> None:
    payload = _payload()
    payload["dislikes"] = invalid_sequence

    with pytest.raises(ValidationError):
        OnboardRequest.model_validate(payload, strict=True)


def test_public_onboarding_enforces_dislike_item_length_boundary() -> None:
    payload = _payload()
    payload["dislikes"] = ["x" * 500]

    request = OnboardRequest.model_validate(payload, strict=True)

    assert request.dislikes == ["x" * 500]

    payload["dislikes"] = ["x" * 501]
    with pytest.raises(ValidationError):
        OnboardRequest.model_validate(payload, strict=True)


def test_json_lists_pass_and_compile_to_deeply_immutable_contract() -> None:
    request = OnboardRequest.model_validate(_payload(), strict=True)
    from_json = OnboardRequest.model_validate_json(
        request.model_dump_json(), strict=True
    )

    compiled = compile_onboard_request_v2(from_json)

    assert isinstance(compiled, CompiledOnboardRequestV2)
    assert compiled.dislikes == ("rote repetition",)
    assert isinstance(compiled.profile.skills, tuple)
    assert isinstance(compiled.profile.goals, tuple)
    assert isinstance(compiled.profile.preferences, tuple)
    with pytest.raises(FrozenInstanceError):
        compiled.grade = "grade-11"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        compiled.profile.skills[0].subject = "physics"  # type: ignore[misc]


def test_compiler_revalidates_a_mutated_existing_onboarding_instance() -> None:
    request = OnboardRequest.model_validate(_payload(), strict=True)
    request.dislikes.append(123)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        compile_onboard_request_v2(request)
