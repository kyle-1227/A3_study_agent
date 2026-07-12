"""Transparent LangGraph checkpointer proxy with nested database/checkpoint spans."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from src.context_engineering.workspace import sanitize_workspace_text
from src.observability.performance_runtime import performance_span


class ObservableCheckpointer(BaseCheckpointSaver):
    """Delegate every saver operation while recording content-free timings."""

    def __init__(self, delegate: Any) -> None:
        if delegate is None:
            raise ValueError("observable checkpointer requires a delegate")
        super().__init__(serde=getattr(delegate, "serde", None))
        self._delegate = delegate
        self._backend_type = _backend_type(delegate)

    @property
    def config_specs(self):
        return self._delegate.config_specs

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        with self._observe("get_tuple"):
            return self._delegate.get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        with self._observe("list"):
            yield from self._delegate.list(
                config,
                filter=filter,
                before=before,
                limit=limit,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self._observe("put"):
            return self._delegate.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self._observe("put_writes"):
            return self._delegate.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        with self._observe("delete_thread"):
            return self._delegate.delete_thread(thread_id)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        with self._observe("aget_tuple"):
            return await self._delegate.aget_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        with self._observe("alist"):
            async for item in self._delegate.alist(
                config,
                filter=filter,
                before=before,
                limit=limit,
            ):
                yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self._observe("aput"):
            return await self._delegate.aput(
                config,
                checkpoint,
                metadata,
                new_versions,
            )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self._observe("aput_writes"):
            return await self._delegate.aput_writes(
                config,
                writes,
                task_id,
                task_path,
            )

    async def adelete_thread(self, thread_id: str) -> None:
        with self._observe("adelete_thread"):
            return await self._delegate.adelete_thread(thread_id)

    def get_next_version(self, current: Any, channel: Any) -> Any:
        return self._delegate.get_next_version(current, channel)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @contextmanager
    def _observe(self, method: str):
        attributes = {"backend_type": self._backend_type}
        with performance_span(
            "database",
            "database.checkpointer",
            attributes=attributes,
        ):
            with performance_span(
                "checkpoint",
                f"checkpoint.{method}",
                attributes=attributes,
            ):
                yield


def observe_checkpointer(checkpointer: Any) -> Any:
    if checkpointer is None or isinstance(checkpointer, ObservableCheckpointer):
        return checkpointer
    return ObservableCheckpointer(checkpointer)


def _backend_type(delegate: Any) -> str:
    value = sanitize_workspace_text(
        type(delegate).__name__,
        max_chars=120,
        fallback="unknown",
    )
    return value or "unknown"


__all__ = ["ObservableCheckpointer", "observe_checkpointer"]
