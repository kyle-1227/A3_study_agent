"""Fail-closed checkpoint migration planning and application service."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.checkpoint_migration.contracts import (
    CheckpointMigrationBatchResultV1,
    CheckpointMigrationPlanRecordV1,
    CheckpointMigrationSchemaContractV1,
    CheckpointMigrationSnapshotV1,
    CheckpointMigrationSpecificationV1,
    MigrationMode,
    PendingCheckpointMigrationPlanV1,
    PendingCheckpointMigrationResultV1,
    SchemaVersion,
    TerminalCheckpointMigrationPlanV1,
    TerminalCheckpointMigrationResultV1,
)

StateValidator = Callable[[Mapping[str, Any]], object]
ResourceValidator = Callable[[Mapping[str, Any]], object]


class CheckpointMigrationCheckpointer(Protocol):
    """Read boundary implemented by an approved saver adapter, never raw SQL."""

    def iter_latest_migration_snapshots(
        self,
    ) -> AsyncIterator[CheckpointMigrationSnapshotV1]: ...


class CheckpointMigrationProjector(Protocol):
    """Version-specific state transformer injected by the migration release."""

    def project_target_values(
        self,
        snapshot: CheckpointMigrationSnapshotV1,
        specification: CheckpointMigrationSpecificationV1,
    ) -> Mapping[str, Any]: ...


class CheckpointMigrationGraph(Protocol):
    """Compare-and-set graph update boundary implemented without raw SQL."""

    async def apply_checkpoint_migration(
        self,
        command: CheckpointMigrationCommandV1,
    ) -> None: ...


@dataclass(frozen=True)
class CheckpointMigrationValidators:
    """Explicit validators registered for every source and target schema."""

    run_control: Mapping[str, StateValidator]
    resource_final: Mapping[SchemaVersion, ResourceValidator]

    def require_run_control(self, schema_version: str) -> StateValidator:
        validator = self.run_control.get(schema_version)
        if validator is None:
            raise ValueError("run_control_validator_not_registered")
        return validator

    def require_resource_final(
        self,
        schema_version: SchemaVersion,
    ) -> ResourceValidator:
        validator = self.resource_final.get(schema_version)
        if validator is None:
            raise ValueError("resource_final_validator_not_registered")
        return validator


class CheckpointMigrationCommandV1(BaseModel):
    """Internal compare-and-set command passed to the graph update boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["checkpoint_migration_command_v1"]
    migration_id: str = Field(min_length=1, max_length=160)
    thread_id: str = Field(min_length=1, max_length=200)
    expected_checkpoint_id: str = Field(min_length=1, max_length=240)
    disposition: Literal["terminal", "pending"]
    target_pending_node_ids: tuple[str, ...] = Field(max_length=100)
    target_values: dict[str, Any]


@dataclass(frozen=True)
class _PreparedMigration:
    plan: CheckpointMigrationPlanRecordV1
    command: CheckpointMigrationCommandV1 | None


class CheckpointMigrationApplyError(RuntimeError):
    """Typed failure that reports partial progress without hiding write failure."""

    def __init__(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        applied_count: int,
    ) -> None:
        self.thread_id = thread_id
        self.checkpoint_id = checkpoint_id
        self.applied_count = applied_count
        super().__init__(
            "checkpoint migration apply failed "
            f"for thread={thread_id} checkpoint={checkpoint_id} "
            f"after applied_count={applied_count}"
        )


async def migrate_checkpoints(
    *,
    specification: CheckpointMigrationSpecificationV1,
    checkpointer: CheckpointMigrationCheckpointer,
    graph: CheckpointMigrationGraph,
    projector: CheckpointMigrationProjector,
    validators: CheckpointMigrationValidators,
    mode: MigrationMode,
) -> CheckpointMigrationBatchResultV1:
    """Plan every latest checkpoint, then apply only an entirely valid batch."""
    _require_validator_registry(specification, validators)
    prepared: list[_PreparedMigration] = []
    seen_threads: set[str] = set()
    async for snapshot in checkpointer.iter_latest_migration_snapshots():
        if snapshot.thread_id in seen_threads:
            prepared.append(
                _blocked_prepared(
                    specification,
                    snapshot,
                    ("duplicate_latest_checkpoint_for_thread",),
                )
            )
            continue
        seen_threads.add(snapshot.thread_id)
        prepared.append(
            _prepare_migration(
                specification=specification,
                snapshot=snapshot,
                projector=projector,
                validators=validators,
            )
        )

    blocked = any(item.plan.blocked_reasons for item in prepared)
    if blocked or mode == "dry_run":
        return _build_result(
            specification=specification,
            mode=mode,
            prepared=prepared,
            applied_count=0,
            blocked=blocked,
        )

    applied_count = 0
    for item in prepared:
        command = item.command
        if command is None:
            raise AssertionError("validated migration is missing its apply command")
        try:
            await graph.apply_checkpoint_migration(command)
        except Exception as exc:
            raise CheckpointMigrationApplyError(
                thread_id=command.thread_id,
                checkpoint_id=command.expected_checkpoint_id,
                applied_count=applied_count,
            ) from exc
        applied_count += 1
    return _build_result(
        specification=specification,
        mode=mode,
        prepared=prepared,
        applied_count=applied_count,
        blocked=False,
    )


def project_node_references(
    values: Mapping[str, Any],
    node_id_map: Mapping[str, str],
) -> dict[str, Any]:
    """Apply only the operator-approved mapping to known durable node fields."""
    projected = deepcopy(dict(values))
    for field_name in ("current_node", "last_completed_node"):
        value = projected.get(field_name)
        if isinstance(value, str) and value in node_id_map:
            projected[field_name] = node_id_map[value]
    timeline = projected.get("activity_timeline")
    if isinstance(timeline, list):
        for raw_item in timeline:
            if not isinstance(raw_item, dict):
                continue
            for field_name in ("node", "parent"):
                value = raw_item.get(field_name)
                if isinstance(value, str) and value in node_id_map:
                    raw_item[field_name] = node_id_map[value]
    return projected


def _require_validator_registry(
    specification: CheckpointMigrationSpecificationV1,
    validators: CheckpointMigrationValidators,
) -> None:
    validators.require_run_control(specification.source.run_control_schema_version)
    validators.require_run_control(specification.target.run_control_schema_version)
    validators.require_resource_final(
        specification.source.resource_final_schema_version
    )
    validators.require_resource_final(
        specification.target.resource_final_schema_version
    )


def _prepare_migration(
    *,
    specification: CheckpointMigrationSpecificationV1,
    snapshot: CheckpointMigrationSnapshotV1,
    projector: CheckpointMigrationProjector,
    validators: CheckpointMigrationValidators,
) -> _PreparedMigration:
    blocked_reasons: list[str] = []
    source_resource_present = False
    mapped_pending: tuple[str, ...] = ()
    unknown_pending = tuple(
        node_id
        for node_id in snapshot.pending_node_ids
        if node_id not in specification.node_id_map
    )
    if unknown_pending:
        blocked_reasons.append(
            "unknown_pending_node_ids:" + ",".join(sorted(unknown_pending))
        )
    else:
        mapped_pending = tuple(
            specification.node_id_map[node_id] for node_id in snapshot.pending_node_ids
        )

    try:
        source_resource_present = _validate_state_contract(
            values=snapshot.values,
            contract=specification.source,
            validators=validators,
            stage="source",
        )
    except ValueError as exc:
        blocked_reasons.append(str(exc))

    target_values: dict[str, Any] | None = None
    if not blocked_reasons:
        try:
            target_values = deepcopy(
                dict(projector.project_target_values(snapshot, specification))
            )
        except Exception as exc:
            blocked_reasons.append(f"target_projection_failed:{type(exc).__name__}")

    if target_values is not None and not blocked_reasons:
        try:
            target_resource_present = _validate_state_contract(
                values=target_values,
                contract=specification.target,
                validators=validators,
                stage="target",
            )
            if target_resource_present != source_resource_present:
                raise ValueError("target_resource_final_presence_mismatch")
            _validate_exact_node_projection(
                source=snapshot.values,
                target=target_values,
                node_id_map=specification.node_id_map,
            )
        except ValueError as exc:
            blocked_reasons.append(str(exc))

    if blocked_reasons or target_values is None:
        return _blocked_prepared(
            specification,
            snapshot,
            tuple(blocked_reasons),
            source_resource_present=source_resource_present,
            target_pending_node_ids=mapped_pending,
        )

    plan = _build_plan(
        specification=specification,
        snapshot=snapshot,
        resource_final_present=source_resource_present,
        blocked_reasons=(),
        target_pending_node_ids=mapped_pending,
    )
    command = CheckpointMigrationCommandV1(
        schema_version="checkpoint_migration_command_v1",
        migration_id=specification.migration_id,
        thread_id=snapshot.thread_id,
        expected_checkpoint_id=snapshot.checkpoint_id,
        disposition=snapshot.disposition,
        target_pending_node_ids=mapped_pending,
        target_values=target_values,
    )
    return _PreparedMigration(plan=plan, command=command)


def _validate_state_contract(
    *,
    values: Mapping[str, Any],
    contract: CheckpointMigrationSchemaContractV1,
    validators: CheckpointMigrationValidators,
    stage: Literal["source", "target"],
) -> bool:
    if values.get("graph_version") != contract.graph_version:
        raise ValueError(f"{stage}_graph_version_mismatch")
    if values.get("schema_version") != contract.run_control_schema_version:
        raise ValueError(f"{stage}_run_control_schema_version_mismatch")
    try:
        validators.require_run_control(contract.run_control_schema_version)(values)
    except Exception as exc:
        raise ValueError(
            f"{stage}_run_control_validation_failed:{type(exc).__name__}"
        ) from exc

    resource = values.get("last_resource_final_payload", _MISSING)
    if resource is _MISSING or resource == {}:
        return False
    if not isinstance(resource, Mapping):
        raise ValueError(f"{stage}_resource_final_not_mapping")
    if resource.get("schema_version") != contract.resource_final_schema_version:
        raise ValueError(f"{stage}_resource_final_schema_version_mismatch")
    try:
        validators.require_resource_final(contract.resource_final_schema_version)(
            resource
        )
    except Exception as exc:
        raise ValueError(
            f"{stage}_resource_final_validation_failed:{type(exc).__name__}"
        ) from exc
    return True


def _validate_exact_node_projection(
    *,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    node_id_map: Mapping[str, str],
) -> None:
    for field_name in ("current_node", "last_completed_node"):
        source_value = source.get(field_name)
        target_value = target.get(field_name)
        expected = (
            node_id_map[source_value]
            if isinstance(source_value, str) and source_value in node_id_map
            else source_value
        )
        if target_value != expected:
            raise ValueError(f"target_{field_name}_mapping_mismatch")

    source_timeline = source.get("activity_timeline", _MISSING)
    target_timeline = target.get("activity_timeline", _MISSING)
    if source_timeline is _MISSING and target_timeline is _MISSING:
        return
    if not isinstance(source_timeline, list) or not isinstance(target_timeline, list):
        raise ValueError("target_activity_timeline_shape_mismatch")
    if len(source_timeline) != len(target_timeline):
        raise ValueError("target_activity_timeline_length_mismatch")
    for index, (source_item, target_item) in enumerate(
        zip(source_timeline, target_timeline, strict=True)
    ):
        if not isinstance(source_item, Mapping) or not isinstance(target_item, Mapping):
            raise ValueError(f"target_activity_timeline_item_invalid:{index}")
        for field_name in ("node", "parent"):
            source_value = source_item.get(field_name)
            expected = (
                node_id_map[source_value]
                if isinstance(source_value, str) and source_value in node_id_map
                else source_value
            )
            if target_item.get(field_name) != expected:
                raise ValueError(
                    f"target_activity_{field_name}_mapping_mismatch:{index}"
                )


def _blocked_prepared(
    specification: CheckpointMigrationSpecificationV1,
    snapshot: CheckpointMigrationSnapshotV1,
    blocked_reasons: tuple[str, ...],
    *,
    source_resource_present: bool = False,
    target_pending_node_ids: tuple[str, ...] = (),
) -> _PreparedMigration:
    plan = _build_plan(
        specification=specification,
        snapshot=snapshot,
        resource_final_present=source_resource_present,
        blocked_reasons=blocked_reasons,
        target_pending_node_ids=target_pending_node_ids,
    )
    return _PreparedMigration(plan=plan, command=None)


def _build_plan(
    *,
    specification: CheckpointMigrationSpecificationV1,
    snapshot: CheckpointMigrationSnapshotV1,
    resource_final_present: bool,
    blocked_reasons: tuple[str, ...],
    target_pending_node_ids: tuple[str, ...],
) -> CheckpointMigrationPlanRecordV1:
    common: dict[str, Any] = {
        "schema_version": "checkpoint_migration_plan_v1",
        "migration_id": specification.migration_id,
        "thread_id": snapshot.thread_id,
        "checkpoint_id": snapshot.checkpoint_id,
        "source_graph_version": specification.source.graph_version,
        "target_graph_version": specification.target.graph_version,
        "resource_final_present": resource_final_present,
        "blocked_reasons": blocked_reasons,
    }
    if snapshot.disposition == "terminal":
        return TerminalCheckpointMigrationPlanV1(
            disposition="terminal",
            **common,
        )
    return PendingCheckpointMigrationPlanV1(
        disposition="pending",
        source_pending_node_ids=snapshot.pending_node_ids,
        target_pending_node_ids=target_pending_node_ids,
        **common,
    )


def _build_result(
    *,
    specification: CheckpointMigrationSpecificationV1,
    mode: MigrationMode,
    prepared: list[_PreparedMigration],
    applied_count: int,
    blocked: bool,
) -> CheckpointMigrationBatchResultV1:
    terminal: list[TerminalCheckpointMigrationResultV1] = []
    pending: list[PendingCheckpointMigrationResultV1] = []
    for item in prepared:
        plan = item.plan
        if plan.blocked_reasons:
            outcome: Literal["not_applied", "applied", "blocked"] = "blocked"
            reasons = plan.blocked_reasons
        elif mode == "apply" and not blocked:
            outcome = "applied"
            reasons = ()
        else:
            outcome = "not_applied"
            reasons = ()
        common: dict[str, Any] = {
            "schema_version": "checkpoint_migration_result_v1",
            "migration_id": plan.migration_id,
            "thread_id": plan.thread_id,
            "checkpoint_id": plan.checkpoint_id,
            "outcome": outcome,
            "blocked_reasons": reasons,
        }
        if isinstance(plan, TerminalCheckpointMigrationPlanV1):
            terminal.append(
                TerminalCheckpointMigrationResultV1(
                    disposition="terminal",
                    **common,
                )
            )
        else:
            pending.append(
                PendingCheckpointMigrationResultV1(
                    disposition="pending",
                    source_pending_node_ids=plan.source_pending_node_ids,
                    target_pending_node_ids=plan.target_pending_node_ids,
                    **common,
                )
            )
    status: Literal["planned", "applied", "blocked"] = (
        "blocked" if blocked else "applied" if mode == "apply" else "planned"
    )
    return CheckpointMigrationBatchResultV1(
        schema_version="checkpoint_migration_result_v1",
        migration_id=specification.migration_id,
        mode=mode,
        status=status,
        scanned_count=len(prepared),
        terminal_count=len(terminal),
        pending_count=len(pending),
        blocked_count=sum(bool(item.plan.blocked_reasons) for item in prepared),
        applied_count=applied_count,
        terminal_results=tuple(terminal),
        pending_results=tuple(pending),
    )


_MISSING = object()


__all__ = [
    "CheckpointMigrationApplyError",
    "CheckpointMigrationCheckpointer",
    "CheckpointMigrationCommandV1",
    "CheckpointMigrationGraph",
    "CheckpointMigrationProjector",
    "CheckpointMigrationValidators",
    "migrate_checkpoints",
    "project_node_references",
]
