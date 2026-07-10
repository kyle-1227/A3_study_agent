"""Graph Manifest contract and compiled-topology tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.graph.builder import get_compiled_graph
from src.observability.contracts import GraphManifest
from src.observability.graph_manifest import (
    GraphManifestBuildError,
    build_graph_manifest,
    graph_manifest_error_payload,
)
from src.observability.node_registry import get_resource_workflow_nodes


def _build_manifest():
    return build_graph_manifest(
        get_compiled_graph(),
        context_policy_mode="strict",
        checkpointer_enabled=True,
        checkpointer_type="postgres",
    )


def test_compiled_graph_manifest_is_deterministic_and_complete():
    first = _build_manifest()
    second = _build_manifest()

    assert first.schema_version == "graph_manifest_v1"
    assert first.graph_version == second.graph_version
    assert first.generated_at
    node_ids = {node.node_id for node in first.nodes}
    physical_ids = {
        str(node_id)
        for node_id in get_compiled_graph().get_graph().nodes
        if not (str(node_id).startswith("__") and str(node_id).endswith("__"))
    }
    assert {node.node_id for node in first.nodes if not node.logical} == physical_ids
    assert all(
        edge.source in node_ids and edge.target in node_ids for edge in first.edges
    )


def test_manifest_logical_nodes_come_from_workflow_registry():
    manifest = _build_manifest()
    workflows = get_resource_workflow_nodes()
    logical_ids = {node.node_id for node in manifest.nodes if node.logical}

    assert logical_ids == {
        node_id for node_ids in workflows.values() for node_id in node_ids
    }
    for node in manifest.nodes:
        if node.logical:
            assert node.parent == "resource_worker"
            assert node.workflow in workflows


def test_manifest_does_not_expose_runtime_connection_metadata():
    manifest = _build_manifest()
    serialized = manifest.model_dump_json()

    assert "postgresql://" not in serialized
    assert "api_key" not in serialized
    assert manifest.capability_metadata["checkpointer_type"] == "postgres"


def test_missing_node_metadata_fails_instead_of_guessing():
    fake_graph = SimpleNamespace(
        get_graph=lambda: SimpleNamespace(
            nodes={"__start__": object(), "unregistered_node": object()},
            edges=[],
        )
    )

    with pytest.raises(GraphManifestBuildError, match="node metadata missing"):
        build_graph_manifest(
            fake_graph,
            context_policy_mode="strict",
            checkpointer_enabled=False,
            checkpointer_type="disabled",
        )


def test_manifest_unavailable_contract_is_safe_and_typed():
    payload = graph_manifest_error_payload(
        GraphManifestBuildError("topology unavailable", error_type="TopologyError")
    )

    assert payload.model_dump() == {
        "schema_version": "graph_manifest_error_v1",
        "error": "graph_manifest_unavailable",
        "reason": "topology unavailable",
        "error_type": "TopologyError",
    }


def test_graph_manifest_forbids_schema_drift():
    valid = _build_manifest().model_dump(mode="json")
    valid["unexpected"] = True

    with pytest.raises(ValidationError):
        GraphManifest.model_validate(valid)
