"""Strict contracts shared by video-animation graph nodes and render tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AnimationStepV1 = Literal[
    "fade_in",
    "move",
    "highlight",
    "arrow_draw",
    "code_highlight",
    "fade_out",
]

REQUIRED_ANIMATION_STEPS_V1: frozenset[str] = frozenset(
    {
        "fade_in",
        "move",
        "highlight",
        "arrow_draw",
        "code_highlight",
        "fade_out",
    }
)


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class AnimationBoxElementV1(_StrictContract):
    type: Literal["box"]
    text: str = Field(min_length=1, max_length=240)
    x: float = Field(ge=0, le=1200)
    y: float = Field(ge=0, le=650)
    width: float = Field(ge=40, le=520)
    height: float = Field(ge=30, le=240)


class AnimationTextElementV1(_StrictContract):
    type: Literal["text"]
    text: str = Field(min_length=1, max_length=240)
    x: float = Field(ge=0, le=1200)
    y: float = Field(ge=0, le=650)
    width: float = Field(ge=40, le=520)
    height: float = Field(ge=30, le=240)


class AnimationCircleElementV1(_StrictContract):
    type: Literal["circle"]
    text: str = Field(min_length=1, max_length=240)
    x: float = Field(ge=0, le=1200)
    y: float = Field(ge=0, le=650)
    width: float = Field(ge=40, le=520)
    height: float = Field(ge=30, le=240)


class AnimationArrowElementV1(_StrictContract):
    type: Literal["arrow"]
    source: str = Field(min_length=1, max_length=120)
    target: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=160)


class AnimationElementV1(_StrictContract):
    """DeepSeek-compatible uniform element with explicit type-specific fields."""

    type: Literal["box", "text", "circle", "arrow"]
    text: str = Field(max_length=240)
    x: float = Field(ge=0, le=1200)
    y: float = Field(ge=0, le=650)
    width: float = Field(ge=40, le=520)
    height: float = Field(ge=30, le=240)
    source: str = Field(max_length=120)
    target: str = Field(max_length=120)

    @model_validator(mode="after")
    def validate_type_fields(self) -> AnimationElementV1:
        if not self.text.strip() or self.text != self.text.strip():
            raise ValueError("animation element text must be non-blank and stripped")
        if self.type == "arrow":
            if (
                not self.source.strip()
                or not self.target.strip()
                or self.source != self.source.strip()
                or self.target != self.target.strip()
            ):
                raise ValueError(
                    "arrow source and target must be non-blank and stripped"
                )
            if self.source == self.target:
                raise ValueError("arrow source and target must differ")
        elif self.source or self.target:
            raise ValueError(
                "box, text, and circle elements require empty source and target"
            )
        return self


class AnimationResolutionV1(_StrictContract):
    width: Literal[1280]
    height: Literal[720]


class AnimationStyleV1(_StrictContract):
    theme: str = Field(min_length=1, max_length=80)
    background: str = Field(pattern=r"^(?:#[0-9A-Fa-f]{6}|white|black|transparent)$")
    font: str = Field(min_length=1, max_length=160, pattern=r"^[^{};<>]+$")


class AnimationSceneV1(_StrictContract):
    scene_id: str = Field(min_length=1, max_length=80)
    start: float = Field(ge=0, le=90)
    end: float = Field(gt=0, le=90)
    title: str = Field(min_length=1, max_length=160)
    subtitle: str = Field(min_length=1, max_length=240)
    narration: str = Field(min_length=1, max_length=1200)
    visual_type: str = Field(min_length=1, max_length=80)
    elements: list[AnimationElementV1] = Field(min_length=1, max_length=30)
    animation_steps: list[AnimationStepV1] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def validate_time_range(self) -> AnimationSceneV1:
        if self.end <= self.start:
            raise ValueError("scene end must be greater than start")
        return self


class VideoAnimationSpecV1(_StrictContract):
    schema_version: Literal["video_animation_spec_v1"]
    title: str = Field(min_length=1, max_length=160)
    duration_seconds: float = Field(gt=0, le=90)
    resolution: AnimationResolutionV1
    style: AnimationStyleV1
    scenes: list[AnimationSceneV1] = Field(min_length=5, max_length=8)


class AnimationReviewVerdictV1(_StrictContract):
    verdict: Literal["approve", "reject"]
    reason: str = Field(min_length=1, max_length=600)


def validate_video_animation_spec(parsed: BaseModel) -> str:
    if not isinstance(parsed, VideoAnimationSpecV1):
        return "root expected VideoAnimationSpecV1"

    seen_scene_ids: set[str] = set()
    previous_end = 0.0
    for index, scene in enumerate(parsed.scenes):
        if scene.scene_id in seen_scene_ids:
            return f"scene_id must be unique: {scene.scene_id}"
        seen_scene_ids.add(scene.scene_id)
        if index and scene.start < previous_end:
            return f"scene {index + 1} overlaps the previous scene"
        if scene.end > parsed.duration_seconds:
            return f"scene {index + 1} end exceeds duration_seconds"
        if set(scene.animation_steps) != REQUIRED_ANIMATION_STEPS_V1:
            return (
                f"scene {index + 1} animation_steps must contain every required step "
                "exactly once"
            )
        node_labels = {
            element.text for element in scene.elements if element.type != "arrow"
        }
        for element in scene.elements:
            if element.type != "arrow":
                continue
            if element.source not in node_labels or element.target not in node_labels:
                return (
                    f"scene {index + 1} arrow source and target must reference "
                    "non-arrow element text in the same scene"
                )
        previous_end = scene.end
    return ""


def validate_animation_review_verdict(parsed: BaseModel) -> str:
    if not isinstance(parsed, AnimationReviewVerdictV1):
        return "root expected AnimationReviewVerdictV1"
    if not parsed.reason.strip():
        return "reason must be non-empty"
    return ""
