"""Bounded, ordered execution for strict document-embedding batches.

The helper deliberately owns only execution scheduling.  Provider protocol
validation, retries, and vector validation remain at their existing explicit
boundaries.  A failed batch never substitutes another provider or result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Generic, TypeVar

from src.rag.parent_child.exceptions import ParentChildError


_ResultT = TypeVar("_ResultT")


class EmbeddingBatchExecutionError(ParentChildError):
    """Raised when one bounded document-embedding batch fails."""


@dataclass(frozen=True, slots=True)
class EmbeddedDocumentBatch(Generic[_ResultT]):
    """One completed batch, emitted in original input order."""

    batch_start: int
    batch_size: int
    result: _ResultT


def _validate_inputs(
    *,
    texts: Sequence[str],
    batch_size: int,
    max_in_flight_batches: int,
) -> tuple[str, ...]:
    if type(batch_size) is not int or batch_size <= 0:
        raise EmbeddingBatchExecutionError("embedding batch_size must be positive")
    if type(max_in_flight_batches) is not int or max_in_flight_batches <= 0:
        raise EmbeddingBatchExecutionError(
            "embedding max_in_flight_batches must be positive"
        )
    frozen_texts = tuple(texts)
    if not frozen_texts or any(
        not isinstance(text, str) or not text for text in frozen_texts
    ):
        raise EmbeddingBatchExecutionError(
            "embedding batch inputs must be non-empty strings"
        )
    return frozen_texts


def _cancel_outstanding(futures: Sequence[Future[object]]) -> None:
    for future in futures:
        future.cancel()


def iter_bounded_document_embedding_batches(
    *,
    texts: Sequence[str],
    batch_size: int,
    max_in_flight_batches: int,
    embed_documents: Callable[[list[str]], _ResultT],
) -> Iterator[EmbeddedDocumentBatch[_ResultT]]:
    """Execute document batches with an explicit in-flight ceiling.

    The iterator submits no more than ``max_in_flight_batches`` requests at a
    time and emits completed results by their original batch position.  If any
    request fails, outstanding futures are cancelled and an explicit typed
    error is raised; it never retries, reorders inputs, or supplies a fallback
    result.
    """

    ordered_texts = _validate_inputs(
        texts=texts,
        batch_size=batch_size,
        max_in_flight_batches=max_in_flight_batches,
    )
    batch_starts = tuple(range(0, len(ordered_texts), batch_size))
    executor = ThreadPoolExecutor(max_workers=max_in_flight_batches)
    in_flight: dict[Future[_ResultT], int] = {}
    completed: dict[int, _ResultT] = {}
    next_to_submit = 0
    next_to_emit = 0

    def submit_available() -> None:
        nonlocal next_to_submit
        while len(in_flight) < max_in_flight_batches and next_to_submit < len(
            batch_starts
        ):
            batch_start = batch_starts[next_to_submit]
            batch = list(ordered_texts[batch_start : batch_start + batch_size])
            future = executor.submit(embed_documents, batch)
            in_flight[future] = batch_start
            next_to_submit += 1

    try:
        submit_available()
        while next_to_emit < len(batch_starts):
            expected_batch_start = batch_starts[next_to_emit]
            while expected_batch_start not in completed:
                done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: in_flight[item]):
                    completed_batch_start = in_flight[future]
                    try:
                        result = future.result()
                    except Exception:
                        _cancel_outstanding(tuple(in_flight))
                        raise EmbeddingBatchExecutionError(
                            "document embedding failed for "
                            f"batch_start={completed_batch_start}, "
                            "batch_size="
                            f"{min(batch_size, len(ordered_texts) - completed_batch_start)}"
                        ) from None
                    del in_flight[future]
                    completed[completed_batch_start] = result
                submit_available()

            result = completed.pop(expected_batch_start)
            yield EmbeddedDocumentBatch(
                batch_start=expected_batch_start,
                batch_size=min(
                    batch_size,
                    len(ordered_texts) - expected_batch_start,
                ),
                result=result,
            )
            next_to_emit += 1
    except BaseException:
        _cancel_outstanding(tuple(in_flight))
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


__all__ = [
    "EmbeddedDocumentBatch",
    "EmbeddingBatchExecutionError",
    "iter_bounded_document_embedding_batches",
]
