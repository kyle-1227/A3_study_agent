"""Deterministic Graph Manifest construction from compiled LangGraph topology."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from src.context_engineering.itemizer import sanitize_metadata
from src.context_engineering.workspace import sanitize_workspace_text, utc_now_iso
from src.observability.contracts import (
    GRAPH_MANIFEST_SCHEMA_VERSION,
    GraphManifest,
    GraphManifestEdge,
    GraphManifestNode,
    GraphManifestUnavailable,
)
from src.observability.node_registry import (
    get_node_runtime_metadata,
    get_resource_workflow_nodes,
)


class GraphManifestBuildError(RuntimeError):
    """Raised when executable topology cannot be represented without guessing."""

    def __init__(self, reason: str, *, error_type: str = "GraphManifestBuildError"):
        self.reason = sanitize_workspace_text(reason, max_chars=160, fallback="")
        self.error_type = sanitize_workspace_text(
            error_type,
            max_chars=120,
            fallback="GraphManifestBuildError",
        )
        super().__init__(self.reason)


def build_graph_manifest(
    compiled_graph: Any,
    *,
    context_policy_mode: str,
    checkpointer_enabled: bool,
    checkpointer_type: str,
) -> GraphManifest:
    """Build and validate the backend graph source of truth."""
    drawable = _compiled_drawable(compiled_graph)
    raw_nodes = getattr(drawable, "nodes", None)
    raw_edges = getattr(drawable, "edges", None)
    if not isinstance(raw_nodes, Mapping) or not isinstance(raw_edges, list):
        raise GraphManifestBuildError("compiled graph topology shape is invalid")

    physical_ids = list(get_physical_graph_node_ids(compiled_graph))
    physical_nodes = [
        _manifest_node(node_id, logical=False) for node_id in physical_ids
    ]

    workflow_nodes = get_resource_workflow_nodes()
    logical_nodes: list[GraphManifestNode] = []
    for workflow in sorted(workflow_nodes):
        for node_id in workflow_nodes[workflow]:
            logical_nodes.append(_manifest_node(node_id, logical=True))

    nodes_by_id = {node.node_id: node for node in [*physical_nodes, *logical_nodes]}
    if len(nodes_by_id) != len(physical_nodes) + len(logical_nodes):
        raise GraphManifestBuildError("physical and logical node ids overlap")

    edges = _physical_edges(raw_edges, valid_ids=set(physical_ids))
    edges.extend(_logical_edges(workflow_nodes, valid_ids=set(nodes_by_id)))
    edges = sorted(edges, key=lambda item: (item.kind, item.workflow, item.edge_id))
    nodes = sorted(
        nodes_by_id.values(),
        key=lambda item: (
            item.logical,
            item.stage_rank,
            item.group,
            item.order,
            item.node_id,
        ),
    )
    capability_metadata = sanitize_metadata(
        {
            "resource_types": sorted(workflow_nodes),
            "resource_workflows": {
                workflow: list(workflow_nodes[workflow])
                for workflow in sorted(workflow_nodes)
            },
            "context_policy_mode": _required_text(
                context_policy_mode,
                "context_policy_mode",
            ),
            "checkpointer_enabled": bool(checkpointer_enabled),
            "checkpointer_type": _required_text(
                checkpointer_type,
                "checkpointer_type",
            ),
            "physical_node_count": len(physical_nodes),
            "logical_node_count": len(logical_nodes),
        }
    )
    identity = {
        "schema_version": GRAPH_MANIFEST_SCHEMA_VERSION,
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "capability_metadata": capability_metadata,
    }
    graph_version = f"graph:v1:{_stable_digest(identity)}"
    return GraphManifest(
        schema_version=GRAPH_MANIFEST_SCHEMA_VERSION,
        graph_version=graph_version,
        generated_at=utc_now_iso(),
        nodes=nodes,
        edges=edges,
        capability_metadata=capability_metadata,
    )


def graph_manifest_error_payload(exc: BaseException) -> GraphManifestUnavailable:
    """Build a typed, content-free 503 detail payload."""
    reason = (
        exc.reason
        if isinstance(exc, GraphManifestBuildError)
        else "graph manifest construction failed"
    )
    error_type = (
        exc.error_type
        if isinstance(exc, GraphManifestBuildError)
        else type(exc).__name__
    )
    return GraphManifestUnavailable(
        schema_version="graph_manifest_error_v1",
        error="graph_manifest_unavailable",
        reason=reason,
        error_type=error_type,
    )


def graph_manifest_ref_payload(graph_version: str) -> dict[str, str]:
    """Return the lightweight SSE reference to the cached manifest."""
    return {
        "type": "graph_manifest_ref",
        "schema_version": "graph_manifest_ref_v1",
        "graph_version": _required_text(graph_version, "graph_version"),
        "endpoint": "/graph/manifest",
    }


def graph_manifest_status_payload(manifest: GraphManifest) -> dict[str, int | str]:
    """Return safe topology counts for manifest-delivery observability."""
    return {
        "schema_version": GRAPH_MANIFEST_SCHEMA_VERSION,
        "graph_version": manifest.graph_version,
        "node_count": len(manifest.nodes),
        "visible_node_count": sum(1 for node in manifest.nodes if node.visible),
        "edge_count": len(manifest.edges),
    }


def get_physical_graph_node_ids(compiled_graph: Any) -> tuple[str, ...]:
    """Return executable node ids from the compiled graph without registry guessing."""

    drawable = _compiled_drawable(compiled_graph)
    raw_nodes = getattr(drawable, "nodes", None)
    if not isinstance(raw_nodes, Mapping):
        raise GraphManifestBuildError("compiled graph nodes shape is invalid")
    return tuple(
        sorted(
            node_id
            for node_id in (str(item or "").strip() for item in raw_nodes)
            if node_id and not _is_internal_node(node_id)
        )
    )


def _compiled_drawable(compiled_graph: Any) -> Any:
    try:
        return compiled_graph.get_graph()
    except Exception as exc:
        raise GraphManifestBuildError(
            "compiled graph introspection failed",
            error_type=type(exc).__name__,
        ) from exc


def _manifest_node(node_id: str, *, logical: bool) -> GraphManifestNode:
    metadata = get_node_runtime_metadata(node_id)
    if metadata is None:
        raise GraphManifestBuildError(f"node metadata missing for {node_id}")
    return GraphManifestNode(
        node_id=metadata.node_id,
        label=metadata.label,
        description=metadata.description,
        kind=metadata.role,
        group=metadata.group,
        parent=metadata.parent,
        workflow=metadata.workflow,
        order=metadata.order,
        stage_rank=metadata.stage_rank,
        visible=metadata.visible,
        logical=logical,
        activity_running=metadata.activity_running,
        activity_completed=metadata.activity_completed,
    )


def _physical_edges(
    raw_edges: list[Any], *, valid_ids: set[str]
) -> list[GraphManifestEdge]:
    result: list[GraphManifestEdge] = []
    for raw in raw_edges:
        source = str(getattr(raw, "source", "") or "").strip()
        target = str(getattr(raw, "target", "") or "").strip()
        if source not in valid_ids or target not in valid_ids:
            continue
        conditional = bool(getattr(raw, "conditional", False))
        label = sanitize_workspace_text(
            getattr(raw, "data", ""),
            max_chars=160,
            fallback="",
        )
        result.append(
            GraphManifestEdge(
                edge_id=_stable_edge_id(
                    source=source,
                    target=target,
                    kind="graph",
                    workflow="",
                    label=label,
                    conditional=conditional,
                ),
                source=source,
                target=target,
                kind="graph",
                conditional=conditional,
                label=label,
            )
        )
    return result


def _logical_edges(
    workflows: Mapping[str, tuple[str, ...]],
    *,
    valid_ids: set[str],
) -> list[GraphManifestEdge]:
    result: list[GraphManifestEdge] = []
    for workflow in sorted(workflows):
        node_ids = workflows[workflow]
        for source, target in zip(node_ids, node_ids[1:]):
            if source not in valid_ids or target not in valid_ids:
                raise GraphManifestBuildError(
                    f"logical workflow {workflow} references unknown metadata"
                )
            result.append(
                GraphManifestEdge(
                    edge_id=_stable_edge_id(
                        source=source,
                        target=target,
                        kind="logical",
                        workflow=workflow,
                        label="",
                        conditional=False,
                    ),
                    source=source,
                    target=target,
                    kind="logical",
                    conditional=False,
                    workflow=workflow,
                )
            )
    return result


def _stable_edge_id(
    *,
    source: str,
    target: str,
    kind: str,
    workflow: str,
    label: str,
    conditional: bool,
) -> str:
    digest = _stable_digest(
        {
            "source": source,
            "target": target,
            "kind": kind,
            "workflow": workflow,
            "label": label,
            "conditional": conditional,
        }
    )
    return f"edge:v1:{digest}"


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def _is_internal_node(node_id: str) -> bool:
    return node_id.startswith("__") and node_id.endswith("__")


def _required_text(value: object, field: str) -> str:
    text = sanitize_workspace_text(value, max_chars=180, fallback="")
    if not text:
        raise GraphManifestBuildError(f"{field} is required")
    return text
