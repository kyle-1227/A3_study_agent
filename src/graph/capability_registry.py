"""Canonical, safe capability metadata for routing, QA, and observability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from src.context_engineering.workspace import sanitize_workspace_text
from src.graph.resource_contracts import RESOURCE_TYPE_ORDER, SUPPORTED_RESOURCE_TYPES
from src.observability.node_registry import (
    get_registered_node_metadata,
    get_resource_workflow_nodes,
)

CAPABILITY_CONTEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CapabilityAction:
    action_id: str
    label: str
    description: str
    requires_resource_type: bool


_CAPABILITY_ACTIONS = (
    CapabilityAction(
        action_id="ask_followup",
        label="Ask a follow-up",
        description="Clarify the current question or learning objective.",
        requires_resource_type=False,
    ),
    CapabilityAction(
        action_id="continue_qa",
        label="Continue the question",
        description="Continue question answering in the current scope.",
        requires_resource_type=False,
    ),
    CapabilityAction(
        action_id="generate_resource",
        label="Generate a learning resource",
        description="Start one registered learning-resource workflow.",
        requires_resource_type=True,
    ),
)


def get_capability_actions() -> tuple[CapabilityAction, ...]:
    return _CAPABILITY_ACTIONS


def get_registered_capability_action(action_id: object) -> CapabilityAction | None:
    normalized = str(action_id or "").strip()
    return next(
        (item for item in _CAPABILITY_ACTIONS if item.action_id == normalized),
        None,
    )


def get_registered_resource_types() -> tuple[str, ...]:
    """Return the canonical, ordered resource registry."""
    return tuple(
        resource_type
        for resource_type in RESOURCE_TYPE_ORDER
        if resource_type in SUPPORTED_RESOURCE_TYPES
    )


def build_safe_capability_context(
    *,
    context_policy_mode: str,
    runtime_metadata: Mapping[str, object],
) -> str:
    """Render safe runtime capabilities without secrets or provider configuration."""
    checkpointer_enabled = runtime_metadata.get("checkpointer_enabled")
    checkpointer_type = sanitize_workspace_text(
        runtime_metadata.get("checkpointer_type"),
        max_chars=80,
    )
    if not isinstance(checkpointer_enabled, bool):
        raise ValueError("runtime capability metadata requires checkpointer_enabled")
    if not checkpointer_type:
        raise ValueError("runtime capability metadata requires checkpointer_type")
    policy_mode = str(context_policy_mode or "").strip()
    if policy_mode not in {"strict", "broad"}:
        raise ValueError(
            "runtime capability metadata requires a valid context policy mode"
        )

    workflows = get_resource_workflow_nodes()
    resource_types = get_registered_resource_types()
    graph_groups = sorted(
        {
            metadata.group
            for metadata in get_registered_node_metadata()
            if metadata.visible and metadata.group
        }
    )
    payload = {
        "schema_version": CAPABILITY_CONTEXT_SCHEMA_VERSION,
        "actions": [
            {
                "action_id": action.action_id,
                "label": action.label,
                "description": action.description,
                "requires_resource_type": action.requires_resource_type,
            }
            for action in _CAPABILITY_ACTIONS
        ],
        "resource_types": list(resource_types),
        "resource_workflow_stage_counts": {
            resource_type: len(workflows.get(resource_type, ()))
            for resource_type in resource_types
        },
        "graph_groups": graph_groups,
        "context_engineering": {"policy_mode": policy_mode},
        "persistence": {
            "checkpointer_enabled": checkpointer_enabled,
            "checkpointer_type": checkpointer_type,
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"<CAPABILITY_CONTEXT>\n{rendered}\n</CAPABILITY_CONTEXT>"
