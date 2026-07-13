"""Strict, adapter-driven checkpoint migration contracts and service."""

from src.checkpoint_migration.contracts import (
    CHECKPOINT_MIGRATION_PLAN_SCHEMA_VERSION,
    CHECKPOINT_MIGRATION_RESULT_SCHEMA_VERSION,
    CHECKPOINT_MIGRATION_SNAPSHOT_SCHEMA_VERSION,
    CheckpointMigrationBatchResultV1,
    CheckpointMigrationSchemaContractV1,
    CheckpointMigrationSnapshotV1,
    CheckpointMigrationSpecificationV1,
    PendingCheckpointMigrationPlanV1,
    PendingCheckpointMigrationResultV1,
    TerminalCheckpointMigrationPlanV1,
    TerminalCheckpointMigrationResultV1,
)
from src.checkpoint_migration.service import (
    CheckpointMigrationApplyError,
    CheckpointMigrationCheckpointer,
    CheckpointMigrationCommandV1,
    CheckpointMigrationGraph,
    CheckpointMigrationProjector,
    CheckpointMigrationValidators,
    migrate_checkpoints,
    project_node_references,
)

__all__ = [
    "CHECKPOINT_MIGRATION_PLAN_SCHEMA_VERSION",
    "CHECKPOINT_MIGRATION_RESULT_SCHEMA_VERSION",
    "CHECKPOINT_MIGRATION_SNAPSHOT_SCHEMA_VERSION",
    "CheckpointMigrationApplyError",
    "CheckpointMigrationBatchResultV1",
    "CheckpointMigrationCheckpointer",
    "CheckpointMigrationCommandV1",
    "CheckpointMigrationGraph",
    "CheckpointMigrationProjector",
    "CheckpointMigrationSchemaContractV1",
    "CheckpointMigrationSnapshotV1",
    "CheckpointMigrationSpecificationV1",
    "CheckpointMigrationValidators",
    "PendingCheckpointMigrationPlanV1",
    "PendingCheckpointMigrationResultV1",
    "TerminalCheckpointMigrationPlanV1",
    "TerminalCheckpointMigrationResultV1",
    "migrate_checkpoints",
    "project_node_references",
]
