"""Content-safe contracts for one-time LangGraph checkpoint migrations."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

CHECKPOINT_MIGRATION_SPEC_SCHEMA_VERSION = "checkpoint_migration_spec_v1"
CHECKPOINT_MIGRATION_SCHEMA_CONTRACT_VERSION = "checkpoint_schema_contract_v1"
CHECKPOINT_MIGRATION_SNAPSHOT_SCHEMA_VERSION = "checkpoint_migration_snapshot_v1"
CHECKPOINT_MIGRATION_PLAN_SCHEMA_VERSION = "checkpoint_migration_plan_v1"
CHECKPOINT_MIGRATION_RESULT_SCHEMA_VERSION = "checkpoint_migration_result_v1"

SchemaVersion: TypeAlias = str | int
MigrationMode: TypeAlias = Literal["dry_run", "apply"]
MigrationBatchStatus: TypeAlias = Literal["planned", "applied", "blocked"]
MigrationRecordOutcome: TypeAlias = Literal["not_applied", "applied", "blocked"]


def _require_exact_text(value: str, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty without outer whitespace")
    return value


def _validate_schema_version(value: SchemaVersion, field_name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must not be boolean")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{field_name} integer must be positive")
        return
    _require_exact_text(value, field_name)


class CheckpointMigrationSchemaContractV1(BaseModel):
    """Exact source or target schema identities for one migration."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["checkpoint_schema_contract_v1"]
    graph_version: str = Field(min_length=1, max_length=200)
    run_control_schema_version: str = Field(min_length=1, max_length=80)
    resource_final_schema_version: SchemaVersion

    @model_validator(mode="after")
    def validate_exact_versions(self) -> CheckpointMigrationSchemaContractV1:
        _require_exact_text(self.graph_version, "graph_version")
        _require_exact_text(
            self.run_control_schema_version,
            "run_control_schema_version",
        )
        _validate_schema_version(
            self.resource_final_schema_version,
            "resource_final_schema_version",
        )
        return self


class CheckpointMigrationSpecificationV1(BaseModel):
    """Operator-approved immutable migration specification."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["checkpoint_migration_spec_v1"]
    migration_id: str = Field(min_length=1, max_length=160)
    source: CheckpointMigrationSchemaContractV1
    target: CheckpointMigrationSchemaContractV1
    node_id_map: dict[str, str] = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_migration_contract(self) -> CheckpointMigrationSpecificationV1:
        _require_exact_text(self.migration_id, "migration_id")
        if self.source.graph_version == self.target.graph_version:
            raise ValueError("source and target graph versions must differ")
        target_ids: set[str] = set()
        for source_id, target_id in self.node_id_map.items():
            _require_exact_text(source_id, "node_id_map source")
            _require_exact_text(target_id, "node_id_map target")
            if target_id in target_ids:
                raise ValueError("node_id_map targets must be unique")
            target_ids.add(target_id)
        return self


class CheckpointMigrationSnapshotV1(BaseModel):
    """Normalized latest checkpoint supplied by an approved saver adapter."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["checkpoint_migration_snapshot_v1"]
    thread_id: str = Field(min_length=1, max_length=200)
    checkpoint_id: str = Field(min_length=1, max_length=240)
    disposition: Literal["terminal", "pending"]
    pending_node_ids: tuple[str, ...] = Field(max_length=100)
    values: dict[str, Any]

    @model_validator(mode="after")
    def validate_snapshot_identity(self) -> CheckpointMigrationSnapshotV1:
        _require_exact_text(self.thread_id, "thread_id")
        _require_exact_text(self.checkpoint_id, "checkpoint_id")
        seen: set[str] = set()
        for node_id in self.pending_node_ids:
            _require_exact_text(node_id, "pending_node_id")
            if node_id in seen:
                raise ValueError("pending_node_ids must be unique")
            seen.add(node_id)
        if self.disposition == "terminal" and self.pending_node_ids:
            raise ValueError("terminal checkpoint cannot contain pending nodes")
        if self.disposition == "pending" and not self.pending_node_ids:
            raise ValueError("pending checkpoint requires at least one pending node")
        return self


class _CheckpointMigrationPlanBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["checkpoint_migration_plan_v1"]
    migration_id: str = Field(min_length=1, max_length=160)
    thread_id: str = Field(min_length=1, max_length=200)
    checkpoint_id: str = Field(min_length=1, max_length=240)
    source_graph_version: str = Field(min_length=1, max_length=200)
    target_graph_version: str = Field(min_length=1, max_length=200)
    resource_final_present: bool
    blocked_reasons: tuple[str, ...] = Field(max_length=50)


class TerminalCheckpointMigrationPlanV1(_CheckpointMigrationPlanBase):
    """Content-free plan for a terminal checkpoint."""

    disposition: Literal["terminal"]


class PendingCheckpointMigrationPlanV1(_CheckpointMigrationPlanBase):
    """Content-free plan for a resumable checkpoint and its exact node mapping."""

    disposition: Literal["pending"]
    source_pending_node_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    target_pending_node_ids: tuple[str, ...] = Field(max_length=100)

    @model_validator(mode="after")
    def validate_pending_mapping(self) -> PendingCheckpointMigrationPlanV1:
        if not self.blocked_reasons and len(self.source_pending_node_ids) != len(
            self.target_pending_node_ids
        ):
            raise ValueError("ready pending plan requires a one-to-one node mapping")
        return self


CheckpointMigrationPlanRecordV1: TypeAlias = Annotated[
    TerminalCheckpointMigrationPlanV1 | PendingCheckpointMigrationPlanV1,
    Field(discriminator="disposition"),
]


class _CheckpointMigrationResultBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["checkpoint_migration_result_v1"]
    migration_id: str = Field(min_length=1, max_length=160)
    thread_id: str = Field(min_length=1, max_length=200)
    checkpoint_id: str = Field(min_length=1, max_length=240)
    outcome: MigrationRecordOutcome
    blocked_reasons: tuple[str, ...] = Field(max_length=50)

    @model_validator(mode="after")
    def validate_outcome(self) -> _CheckpointMigrationResultBase:
        if self.outcome == "blocked" and not self.blocked_reasons:
            raise ValueError("blocked result requires a reason")
        if self.outcome != "blocked" and self.blocked_reasons:
            raise ValueError("non-blocked result cannot contain blocked reasons")
        return self


class TerminalCheckpointMigrationResultV1(_CheckpointMigrationResultBase):
    """Result for one terminal checkpoint without exposing checkpoint state."""

    disposition: Literal["terminal"]


class PendingCheckpointMigrationResultV1(_CheckpointMigrationResultBase):
    """Result for one pending checkpoint without exposing checkpoint state."""

    disposition: Literal["pending"]
    source_pending_node_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    target_pending_node_ids: tuple[str, ...] = Field(max_length=100)


class CheckpointMigrationBatchResultV1(BaseModel):
    """Content-free batch result suitable for CLI output and audit logs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["checkpoint_migration_result_v1"]
    migration_id: str = Field(min_length=1, max_length=160)
    mode: MigrationMode
    status: MigrationBatchStatus
    scanned_count: int = Field(ge=0)
    terminal_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    applied_count: int = Field(ge=0)
    terminal_results: tuple[TerminalCheckpointMigrationResultV1, ...]
    pending_results: tuple[PendingCheckpointMigrationResultV1, ...]

    @model_validator(mode="after")
    def validate_counts_and_status(self) -> CheckpointMigrationBatchResultV1:
        all_results = (*self.terminal_results, *self.pending_results)
        if self.scanned_count != len(all_results):
            raise ValueError("scanned_count does not match result count")
        if self.terminal_count != len(self.terminal_results):
            raise ValueError("terminal_count does not match terminal results")
        if self.pending_count != len(self.pending_results):
            raise ValueError("pending_count does not match pending results")
        if self.blocked_count != sum(item.outcome == "blocked" for item in all_results):
            raise ValueError("blocked_count does not match blocked results")
        if self.applied_count != sum(item.outcome == "applied" for item in all_results):
            raise ValueError("applied_count does not match applied results")
        if self.status == "blocked":
            if self.blocked_count < 1 or self.applied_count:
                raise ValueError("blocked batch must have blockers and no writes")
        elif self.status == "planned":
            if self.mode != "dry_run" or self.blocked_count or self.applied_count:
                raise ValueError("planned batch must be an unblocked dry run")
        elif self.mode != "apply" or self.blocked_count:
            raise ValueError("applied batch must be an unblocked apply run")
        return self


__all__ = [
    "CHECKPOINT_MIGRATION_PLAN_SCHEMA_VERSION",
    "CHECKPOINT_MIGRATION_RESULT_SCHEMA_VERSION",
    "CHECKPOINT_MIGRATION_SCHEMA_CONTRACT_VERSION",
    "CHECKPOINT_MIGRATION_SNAPSHOT_SCHEMA_VERSION",
    "CHECKPOINT_MIGRATION_SPEC_SCHEMA_VERSION",
    "CheckpointMigrationBatchResultV1",
    "CheckpointMigrationPlanRecordV1",
    "CheckpointMigrationSchemaContractV1",
    "CheckpointMigrationSnapshotV1",
    "CheckpointMigrationSpecificationV1",
    "MigrationMode",
    "PendingCheckpointMigrationPlanV1",
    "PendingCheckpointMigrationResultV1",
    "SchemaVersion",
    "TerminalCheckpointMigrationPlanV1",
    "TerminalCheckpointMigrationResultV1",
]
