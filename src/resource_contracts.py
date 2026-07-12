"""Low-level canonical resource-type contracts shared across packages."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias


ResourceType: TypeAlias = Literal[
    "review_doc",
    "mindmap",
    "quiz",
    "code_practice",
    "video_script",
    "video_animation",
    "study_plan",
]

RESOURCE_TYPE_ORDER: tuple[ResourceType, ...] = (
    "review_doc",
    "mindmap",
    "quiz",
    "code_practice",
    "video_script",
    "video_animation",
    "study_plan",
)
SUPPORTED_RESOURCE_TYPES: frozenset[str] = frozenset(RESOURCE_TYPE_ORDER)
RESOURCE_ALIASES: dict[str, ResourceType] = {
    "exercise": "quiz",
    "exercises": "quiz",
    "practice": "quiz",
    "practice_questions": "quiz",
    "review": "review_doc",
    "review_document": "review_doc",
    "doc": "review_doc",
    "document": "review_doc",
    "learning_plan": "study_plan",
    "roadmap": "study_plan",
    "mind_map": "mindmap",
    "xmind": "mindmap",
    "code": "code_practice",
    "coding_practice": "code_practice",
    "video": "video_animation",
    "animation": "video_animation",
    "video_animation": "video_animation",
    "video_script": "video_script",
}


def normalize_resource_type(value: Any) -> str:
    """Normalize a public resource alias to one canonical resource type."""

    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return ""
    normalized = RESOURCE_ALIASES.get(text, text)
    return normalized if normalized in SUPPORTED_RESOURCE_TYPES else ""


def normalize_requested_resource_types(*values: Any) -> list[str]:
    """Return ordered, deduplicated canonical resource types."""

    normalized: list[str] = []

    def add_one(item: Any) -> None:
        resource_type = normalize_resource_type(item)
        if resource_type and resource_type not in normalized:
            normalized.append(resource_type)

    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add_one(item)
        else:
            add_one(value)
    return normalized


__all__ = [
    "RESOURCE_ALIASES",
    "RESOURCE_TYPE_ORDER",
    "SUPPORTED_RESOURCE_TYPES",
    "ResourceType",
    "normalize_requested_resource_types",
    "normalize_resource_type",
]
