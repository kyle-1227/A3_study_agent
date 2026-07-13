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

    def __init__(
        self,
        message: str,
        *,
        batch_start: int | None,
        batch_size: int | None,
        cause_type: str | None,
        provider_code: int | None,
        retryable: bool | None,
        attempts_exhausted: bool | None,
    ) -> None:
        if (batch_start is None) != (batch_size is None):
            raise ValueError("embedding failure batch coordinates must be paired")
        if batch_start is not None and (
            type(batch_start) is not int or batch_start < 0
        ):
            raise ValueError("embedding failure batch_start must be non-negative")
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError("embedding failure batch_size must be positive")
        if cause_type is not None and (
            not cause_type.isascii()
            or not cause_type.isidentifier()
            or len(cause_type) > 128
        ):
            raise ValueError("embedding failure cause_type must be a safe identifier")
        provider_attributes = (provider_code, retryable, attempts_exhausted)
        if any(value is not None for value in provider_attributes) and not all(
            value is not None for value in provider_attributes
        ):
            raise ValueError("embedding provider failure attributes must be complete")
        if provider_code is not None and type(provider_code) is not int:
            raise ValueError("embedding provider code must be an integer")
        if retryable is not None and type(retryable) is not bool:
            raise ValueError("embedding provider retryable must be a boolean")
        if attempts_exhausted is not None and type(attempts_exhausted) is not bool:
            raise ValueError("embedding provider attempts_exhausted must be a boolean")
        self.batch_start = batch_start
        self.batch_size = batch_size
        self.cause_type = cause_type
        self.provider_code = provider_code
        self.retryable = retryable
        self.attempts_exhausted = attempts_exhausted
        super().__init__(message)


def _safe_exception_type(exc: BaseException) -> str:
    name = type(exc).__name__
    if name.isascii() and name.isidentifier() and len(name) <= 128:
        return name
    return "UnknownFailure"


def _safe_provider_attributes(
    exc: BaseException,
) -> tuple[int | None, bool | None, bool | None]:
    """Copy only the exact primitive provider diagnostics used by strict clients."""

    attributes = vars(exc)
    code = attributes.get("code")
    retryable = attributes.get("retryable")
    attempts_exhausted = attributes.get("attempts_exhausted")
    if (
        type(code) is int
        and type(retryable) is bool
        and type(attempts_exhausted) is bool
    ):
        return code, retryable, attempts_exhausted
    return None, None, None


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
        raise EmbeddingBatchExecutionError(
            "embedding batch_size must be positive",
            batch_start=None,
            batch_size=None,
            cause_type=None,
            provider_code=None,
            retryable=None,
            attempts_exhausted=None,
        )
    if type(max_in_flight_batches) is not int or max_in_flight_batches <= 0:
        raise EmbeddingBatchExecutionError(
            "embedding max_in_flight_batches must be positive",
            batch_start=None,
            batch_size=None,
            cause_type=None,
            provider_code=None,
            retryable=None,
            attempts_exhausted=None,
        )
    frozen_texts = tuple(texts)
    if not frozen_texts or any(
        not isinstance(text, str) or not text for text in frozen_texts
    ):
        raise EmbeddingBatchExecutionError(
            "embedding batch inputs must be non-empty strings",
            batch_start=None,
            batch_size=None,
            cause_type=None,
            provider_code=None,
            retryable=None,
            attempts_exhausted=None,
        )
    return frozen_texts


def _cancel_outstanding(futures: Sequence[Future[_ResultT]]) -> None:
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
                    batch_failure: EmbeddingBatchExecutionError | None = None
                    try:
                        result = future.result()
                    except Exception as exc:
                        _cancel_outstanding(tuple(in_flight))
                        provider_code, retryable, attempts_exhausted = (
                            _safe_provider_attributes(exc)
                        )
                        batch_failure = EmbeddingBatchExecutionError(
                            "document embedding batch execution failed",
                            batch_start=completed_batch_start,
                            batch_size=min(
                                batch_size,
                                len(ordered_texts) - completed_batch_start,
                            ),
                            cause_type=_safe_exception_type(exc),
                            provider_code=provider_code,
                            retryable=retryable,
                            attempts_exhausted=attempts_exhausted,
                        )
                    if batch_failure is not None:
                        # Raise outside the provider ``except`` block so Python does
                        # not retain the raw provider exception as ``__context__``.
                        raise batch_failure
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
