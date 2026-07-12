"""Safe, minimal suggestion contract shared by every QA scope."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

from src.resource_contracts import RESOURCE_TYPE_ORDER, SUPPORTED_RESOURCE_TYPES

QA_SUGGESTION_REGISTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QASuggestionAction:
    """One public action a QA response may suggest without runtime disclosure."""

    action_id: str
    label: str
    description: str
    requires_resource_type: bool


_QA_SUGGESTION_ACTIONS: Final[tuple[QASuggestionAction, ...]] = (
    QASuggestionAction(
        action_id="ask_followup",
        label="Ask a follow-up",
        description="Clarify the current question or learning objective.",
        requires_resource_type=False,
    ),
    QASuggestionAction(
        action_id="continue_qa",
        label="Continue the question",
        description="Continue question answering in the current scope.",
        requires_resource_type=False,
    ),
    QASuggestionAction(
        action_id="generate_resource",
        label="Generate a learning resource",
        description="Start one registered learning-resource workflow.",
        requires_resource_type=True,
    ),
)


def get_qa_suggestion_actions() -> tuple[QASuggestionAction, ...]:
    """Return the canonical, ordered QA suggestion action contract."""
    return _QA_SUGGESTION_ACTIONS


def get_qa_suggestion_action(action_id: object) -> QASuggestionAction | None:
    """Resolve an action only when the model emitted its exact public id."""
    if not isinstance(action_id, str):
        return None
    return next(
        (item for item in _QA_SUGGESTION_ACTIONS if item.action_id == action_id),
        None,
    )


def get_qa_suggestion_resource_types() -> tuple[str, ...]:
    """Return safe resource identifiers, without runtime capability metadata."""
    return tuple(
        resource_type
        for resource_type in RESOURCE_TYPE_ORDER
        if resource_type in SUPPORTED_RESOURCE_TYPES
    )


def build_safe_qa_suggestion_registry() -> str:
    """Render the bounded public suggestion contract for provider-bound messages."""
    actions = get_qa_suggestion_actions()
    payload = {
        "schema_version": QA_SUGGESTION_REGISTRY_SCHEMA_VERSION,
        "suggestions_optional": True,
        "actions": [
            {
                "action_id": action.action_id,
                "label": action.label,
                "description": action.description,
                "requires_resource_type": action.requires_resource_type,
            }
            for action in actions
        ],
        "resource_types": list(get_qa_suggestion_resource_types()),
        "resource_type_required_actions": [
            action.action_id for action in actions if action.requires_resource_type
        ],
        "resource_type_empty_actions": [
            action.action_id for action in actions if not action.requires_resource_type
        ],
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"<QA_SUGGESTION_REGISTRY>\n{rendered}\n</QA_SUGGESTION_REGISTRY>"


def qa_suggestion_validation_guidance() -> str:
    """Return bounded correction text that names every legal output identifier."""
    actions = ", ".join(action.action_id for action in get_qa_suggestion_actions())
    resource_types = ", ".join(get_qa_suggestion_resource_types())
    resource_actions = ", ".join(
        action.action_id
        for action in get_qa_suggestion_actions()
        if action.requires_resource_type
    )
    empty_actions = ", ".join(
        action.action_id
        for action in get_qa_suggestion_actions()
        if not action.requires_resource_type
    )
    return (
        f"allowed action values: [{actions}]; "
        f"allowed resource_type values: [{resource_types}]; "
        f"resource_type is required only for: [{resource_actions}]; "
        f"resource_type must be empty for: [{empty_actions}]"
    )
