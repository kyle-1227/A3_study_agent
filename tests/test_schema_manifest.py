from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from src.llm.schema_manifest import (
    build_canonical_manifest,
    get_structured_output_manifest_config,
    load_drift_guard_config,
    render_manifest_text,
)


class DemoKind(str, Enum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"


class DemoItem(BaseModel):
    name: str = Field(..., max_length=20, description="Canonical item name.")
    kind: DemoKind = DemoKind.PRIMARY
    note: str | None = Field(default=None, max_length=40)


class DemoBatch(BaseModel):
    items: list[DemoItem] = Field(default_factory=list, max_length=3)
    status: Literal["ok", "needs_review"] = "ok"


def _config() -> dict:
    return {
        "manifest": {
            "enabled": True,
            "max_chars": 800,
            "include_descriptions": True,
            "include_constraints": True,
            "include_enum_values": True,
        },
        "drift_guards": {
            "default": {
                "global_forbidden_output_fields": ["metadata", "debug"],
                "global_forbidden_aliases": {"reason": ["rationale"]},
            },
            "demo_node": {
                "forbidden_output_fields": ["task_id"],
                "canonical_aliases": {"name": ["title", "label"]},
            },
        },
    }


def test_manifest_generic_nested_optional_enum_literal_and_constraints():
    manifest = build_canonical_manifest(
        DemoBatch,
        node_name="demo_node",
        output_mode="deepseek_tool_call_strict",
        config=_config(),
    )

    paths = {field.path: field for field in manifest.fields}
    assert "items" in paths
    assert "items[]" in paths
    assert "items[].name" in paths
    assert "items[].kind" in paths
    assert "items[].note" in paths
    assert "status" in paths
    assert paths["items"].constraints["maxItems"] == 3
    assert paths["items[].name"].constraints["maxLength"] == 20
    assert paths["items[].kind"].enum_values == ["primary", "supporting"]
    assert paths["status"].enum_values == ["ok", "needs_review"]
    assert paths["items[].name"].forbidden_aliases == ["title", "label"]

    rendered = render_manifest_text(manifest, max_chars=4000)
    assert "Canonical schema manifest: DemoBatch" in rendered
    assert "items[].name" in rendered
    assert "Forbidden output fields: metadata, debug, task_id" in rendered
    assert "name <- title, label" in rendered


def test_drift_guard_default_and_node_merge_is_validated():
    manifest_cfg = get_structured_output_manifest_config(_config())
    drift_guard = load_drift_guard_config("demo_node", config=_config())

    assert manifest_cfg.enabled is True
    assert drift_guard.drift_guard_source == "default+demo_node"
    assert drift_guard.drift_guard_config_validated is True
    assert drift_guard.forbidden_output_fields == ["metadata", "debug", "task_id"]
    assert drift_guard.canonical_aliases["reason"] == ["rationale"]
    assert drift_guard.canonical_aliases["name"] == ["title", "label"]


def test_invalid_drift_guard_config_fails_fast():
    bad_config = {
        "drift_guards": {
            "default": {
                "global_forbidden_output_fields": "metadata",
            }
        }
    }

    with pytest.raises(Exception):
        load_drift_guard_config("demo_node", config=bad_config)
