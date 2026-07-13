from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from src.checkpoint_migration.cli import run_checkpoint_migration_cli
from src.checkpoint_migration.contracts import (
    CheckpointMigrationSchemaContractV1,
    CheckpointMigrationSnapshotV1,
    CheckpointMigrationSpecificationV1,
)
from src.checkpoint_migration.service import (
    CheckpointMigrationCommandV1,
    CheckpointMigrationValidators,
    migrate_checkpoints,
    project_node_references,
)


class _LegacyResource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int
    type: str
    legacy_id: str


class _ResourceV3(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str
    type: str
    resource_final_id: str


class FakeCheckpointer:
    def __init__(self, snapshots: list[CheckpointMigrationSnapshotV1]) -> None:
        self.snapshots = snapshots

    async def iter_latest_migration_snapshots(
        self,
    ) -> AsyncIterator[CheckpointMigrationSnapshotV1]:
        for snapshot in self.snapshots:
            yield snapshot


class FakeGraph:
    def __init__(self) -> None:
        self.commands: list[CheckpointMigrationCommandV1] = []

    async def apply_checkpoint_migration(
        self,
        command: CheckpointMigrationCommandV1,
    ) -> None:
        self.commands.append(command)


class FakeProjector:
    def project_target_values(
        self,
        snapshot: CheckpointMigrationSnapshotV1,
        specification: CheckpointMigrationSpecificationV1,
    ) -> Mapping[str, Any]:
        projected = project_node_references(
            snapshot.values,
            specification.node_id_map,
        )
        projected["graph_version"] = specification.target.graph_version
        projected["schema_version"] = specification.target.run_control_schema_version
        resource = projected.get("last_resource_final_payload")
        if resource:
            projected["last_resource_final_payload"] = {
                "schema_version": specification.target.resource_final_schema_version,
                "type": "resource_final",
                "resource_final_id": f"v3:{resource['legacy_id']}",
            }
        return projected


class InvalidTargetResourceProjector(FakeProjector):
    def project_target_values(
        self,
        snapshot: CheckpointMigrationSnapshotV1,
        specification: CheckpointMigrationSpecificationV1,
    ) -> Mapping[str, Any]:
        projected = dict(super().project_target_values(snapshot, specification))
        projected["last_resource_final_payload"] = {
            "schema_version": "resource_final_v3",
            "type": "resource_final",
            "legacy_id": "not-a-v3-resource",
        }
        return projected


def _validate_run_control(values: Mapping[str, Any]) -> None:
    if values.get("run_status") not in {"completed", "stopped"}:
        raise ValueError("invalid run status")
    if not isinstance(values.get("resume_available"), bool):
        raise TypeError("resume_available must be boolean")


def _validators() -> CheckpointMigrationValidators:
    return CheckpointMigrationValidators(
        run_control={
            "run_control_v1": _validate_run_control,
            "run_control_v2": _validate_run_control,
        },
        resource_final={
            2: lambda value: _LegacyResource.model_validate(value, strict=True),
            "resource_final_v3": lambda value: _ResourceV3.model_validate(
                value,
                strict=True,
            ),
        },
    )


def _specification() -> CheckpointMigrationSpecificationV1:
    return CheckpointMigrationSpecificationV1(
        schema_version="checkpoint_migration_spec_v1",
        migration_id="migration:test-v1-to-v3",
        source=CheckpointMigrationSchemaContractV1(
            schema_version="checkpoint_schema_contract_v1",
            graph_version="graph-old",
            run_control_schema_version="run_control_v1",
            resource_final_schema_version=2,
        ),
        target=CheckpointMigrationSchemaContractV1(
            schema_version="checkpoint_schema_contract_v1",
            graph_version="graph-new",
            run_control_schema_version="run_control_v2",
            resource_final_schema_version="resource_final_v3",
        ),
        node_id_map={
            "rag_retrieve": "parent_child_retrieve",
            "web_search": "web_research",
            "supervisor": "supervisor",
        },
    )


def _snapshot(
    *,
    thread_id: str,
    disposition: str,
    pending_node_ids: tuple[str, ...],
    with_resource: bool,
) -> CheckpointMigrationSnapshotV1:
    values: dict[str, Any] = {
        "graph_version": "graph-old",
        "schema_version": "run_control_v1",
        "run_status": "stopped" if disposition == "pending" else "completed",
        "resume_available": disposition == "pending",
        "current_node": "rag_retrieve",
        "last_completed_node": "web_search",
        "activity_timeline": [{"node": "rag_retrieve", "parent": "supervisor"}],
    }
    if with_resource:
        values["last_resource_final_payload"] = {
            "schema_version": 2,
            "type": "resource_final",
            "legacy_id": f"legacy:{thread_id}",
        }
    return CheckpointMigrationSnapshotV1.model_validate(
        {
            "schema_version": "checkpoint_migration_snapshot_v1",
            "thread_id": thread_id,
            "checkpoint_id": f"checkpoint:{thread_id}",
            "disposition": disposition,
            "pending_node_ids": pending_node_ids,
            "values": values,
        },
        strict=True,
    )


@pytest.mark.anyio
async def test_dry_run_plans_terminal_and_pending_without_writes() -> None:
    snapshots = [
        _snapshot(
            thread_id="terminal-1",
            disposition="terminal",
            pending_node_ids=(),
            with_resource=True,
        ),
        _snapshot(
            thread_id="pending-1",
            disposition="pending",
            pending_node_ids=("rag_retrieve", "web_search"),
            with_resource=False,
        ),
    ]
    graph = FakeGraph()

    result = await migrate_checkpoints(
        specification=_specification(),
        checkpointer=FakeCheckpointer(snapshots),
        graph=graph,
        projector=FakeProjector(),
        validators=_validators(),
        mode="dry_run",
    )

    assert result.status == "planned"
    assert result.scanned_count == 2
    assert result.applied_count == 0
    assert result.pending_results[0].target_pending_node_ids == (
        "parent_child_retrieve",
        "web_research",
    )
    assert graph.commands == []


@pytest.mark.anyio
async def test_apply_updates_terminal_and_pending_through_graph_boundary() -> None:
    snapshots = [
        _snapshot(
            thread_id="terminal-1",
            disposition="terminal",
            pending_node_ids=(),
            with_resource=True,
        ),
        _snapshot(
            thread_id="pending-1",
            disposition="pending",
            pending_node_ids=("rag_retrieve",),
            with_resource=False,
        ),
    ]
    original_values = deepcopy(snapshots[0].values)
    graph = FakeGraph()

    result = await migrate_checkpoints(
        specification=_specification(),
        checkpointer=FakeCheckpointer(snapshots),
        graph=graph,
        projector=FakeProjector(),
        validators=_validators(),
        mode="apply",
    )

    assert result.status == "applied"
    assert result.applied_count == 2
    assert [command.disposition for command in graph.commands] == [
        "terminal",
        "pending",
    ]
    terminal_command = graph.commands[0]
    assert terminal_command.expected_checkpoint_id == "checkpoint:terminal-1"
    assert terminal_command.target_values["graph_version"] == "graph-new"
    assert terminal_command.target_values["schema_version"] == "run_control_v2"
    assert terminal_command.target_values["current_node"] == ("parent_child_retrieve")
    assert terminal_command.target_values["activity_timeline"][0]["node"] == (
        "parent_child_retrieve"
    )
    assert snapshots[0].values == original_values


@pytest.mark.anyio
async def test_unknown_pending_node_blocks_entire_batch() -> None:
    terminal = _snapshot(
        thread_id="terminal-1",
        disposition="terminal",
        pending_node_ids=(),
        with_resource=False,
    )
    unknown = _snapshot(
        thread_id="pending-unknown",
        disposition="pending",
        pending_node_ids=("unmapped_old_node",),
        with_resource=False,
    )
    graph = FakeGraph()

    result = await migrate_checkpoints(
        specification=_specification(),
        checkpointer=FakeCheckpointer([terminal, unknown]),
        graph=graph,
        projector=FakeProjector(),
        validators=_validators(),
        mode="apply",
    )

    assert result.status == "blocked"
    assert result.blocked_count == 1
    assert result.applied_count == 0
    assert result.terminal_results[0].outcome == "not_applied"
    assert result.pending_results[0].outcome == "blocked"
    assert result.pending_results[0].blocked_reasons == (
        "unknown_pending_node_ids:unmapped_old_node",
    )
    assert graph.commands == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_reason"),
    [
        ("graph_version", "wrong-graph", "source_graph_version_mismatch"),
        (
            "schema_version",
            "wrong-run-control",
            "source_run_control_schema_version_mismatch",
        ),
    ],
)
async def test_source_graph_and_run_control_mismatch_block(
    field_name: str,
    invalid_value: str,
    expected_reason: str,
) -> None:
    snapshot = _snapshot(
        thread_id="invalid-source",
        disposition="terminal",
        pending_node_ids=(),
        with_resource=False,
    )
    snapshot.values[field_name] = invalid_value
    graph = FakeGraph()

    result = await migrate_checkpoints(
        specification=_specification(),
        checkpointer=FakeCheckpointer([snapshot]),
        graph=graph,
        projector=FakeProjector(),
        validators=_validators(),
        mode="apply",
    )

    assert result.status == "blocked"
    assert result.terminal_results[0].blocked_reasons == (expected_reason,)
    assert graph.commands == []


@pytest.mark.anyio
async def test_resource_schema_mismatch_blocks_before_projection() -> None:
    snapshot = _snapshot(
        thread_id="wrong-resource",
        disposition="terminal",
        pending_node_ids=(),
        with_resource=True,
    )
    snapshot.values["last_resource_final_payload"]["schema_version"] = 1
    graph = FakeGraph()

    result = await migrate_checkpoints(
        specification=_specification(),
        checkpointer=FakeCheckpointer([snapshot]),
        graph=graph,
        projector=FakeProjector(),
        validators=_validators(),
        mode="apply",
    )

    assert result.status == "blocked"
    assert result.terminal_results[0].blocked_reasons == (
        "source_resource_final_schema_version_mismatch",
    )
    assert graph.commands == []


@pytest.mark.anyio
async def test_target_resource_business_schema_validation_blocks_apply() -> None:
    snapshot = _snapshot(
        thread_id="invalid-target-resource",
        disposition="terminal",
        pending_node_ids=(),
        with_resource=True,
    )
    graph = FakeGraph()

    result = await migrate_checkpoints(
        specification=_specification(),
        checkpointer=FakeCheckpointer([snapshot]),
        graph=graph,
        projector=InvalidTargetResourceProjector(),
        validators=_validators(),
        mode="apply",
    )

    assert result.status == "blocked"
    assert result.terminal_results[0].blocked_reasons == (
        "target_resource_final_validation_failed:ValidationError",
    )
    assert graph.commands == []


def test_cli_is_dry_run_by_default_and_requires_apply_for_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    specification_path = tmp_path / "migration.json"
    specification_path.write_text(
        json.dumps(_specification().model_dump(mode="json")),
        encoding="utf-8",
    )
    snapshot = _snapshot(
        thread_id="cli-terminal",
        disposition="terminal",
        pending_node_ids=(),
        with_resource=False,
    )
    dry_graph = FakeGraph()

    dry_result = run_checkpoint_migration_cli(
        argv=["--specification", str(specification_path)],
        checkpointer=FakeCheckpointer([snapshot]),
        graph=dry_graph,
        projector=FakeProjector(),
        validators=_validators(),
    )

    assert dry_result.mode == "dry_run"
    assert dry_graph.commands == []
    assert json.loads(capsys.readouterr().out)["status"] == "planned"

    apply_graph = FakeGraph()
    apply_result = run_checkpoint_migration_cli(
        argv=["--specification", str(specification_path), "--apply"],
        checkpointer=FakeCheckpointer([snapshot]),
        graph=apply_graph,
        projector=FakeProjector(),
        validators=_validators(),
    )

    assert apply_result.mode == "apply"
    assert len(apply_graph.commands) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "applied"


def test_contracts_forbid_extra_fields() -> None:
    payload = _specification().model_dump(mode="python")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        CheckpointMigrationSpecificationV1.model_validate(payload, strict=True)
