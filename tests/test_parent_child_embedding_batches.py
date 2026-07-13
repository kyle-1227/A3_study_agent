from __future__ import annotations

from threading import Event, Lock, Thread

import pytest

from src.rag.parent_child.embedding_batches import (
    EmbeddingBatchExecutionError,
    iter_bounded_document_embedding_batches,
)
from src.rag.parent_child.provider_clients import ProviderReportedError


class _BlockingEmbeddingProvider:
    def __init__(self) -> None:
        self.release = Event()
        self.at_limit = Event()
        self._lock = Lock()
        self.active = 0
        self.peak_active = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            if self.active >= 2:
                self.at_limit.set()
        assert self.release.wait(timeout=3.0)
        with self._lock:
            self.active -= 1
        return [[float(int(text))] for text in texts]


def test_embedding_batches_cap_concurrency_and_emit_original_order() -> None:
    provider = _BlockingEmbeddingProvider()
    received: list[object] = []
    failures: list[BaseException] = []

    def consume() -> None:
        try:
            received.extend(
                iter_bounded_document_embedding_batches(
                    texts=("0", "1", "2", "3", "4"),
                    batch_size=2,
                    max_in_flight_batches=2,
                    embed_documents=provider.embed_documents,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    consumer = Thread(target=consume)
    consumer.start()
    assert provider.at_limit.wait(timeout=3.0)
    assert provider.peak_active == 2
    provider.release.set()
    consumer.join(timeout=5.0)

    assert not consumer.is_alive()
    assert failures == []
    assert [
        (batch.batch_start, batch.batch_size, batch.result) for batch in received
    ] == [
        (0, 2, [[0.0], [1.0]]),
        (2, 2, [[2.0], [3.0]]),
        (4, 1, [[4.0]]),
    ]
    assert provider.peak_active <= 2


class _FailingEmbeddingProvider:
    def __init__(self) -> None:
        self.failure_started = Event()
        self.release_hold = Event()
        self._lock = Lock()
        self.calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 1
        value = texts[0]
        with self._lock:
            self.calls.append(value)
        if value == "bad":
            self.failure_started.set()
            raise RuntimeError("provider failure")
        if value == "hold":
            assert self.release_hold.wait(timeout=3.0)
            return [[1.0]]
        raise AssertionError("a failed bounded execution must not submit later batches")


def test_embedding_batch_failure_cancels_remaining_work_and_raises_typed_error() -> (
    None
):
    provider = _FailingEmbeddingProvider()
    failures: list[BaseException] = []

    def consume() -> None:
        try:
            list(
                iter_bounded_document_embedding_batches(
                    texts=("hold", "bad", "must-not-run"),
                    batch_size=1,
                    max_in_flight_batches=2,
                    embed_documents=provider.embed_documents,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    consumer = Thread(target=consume)
    consumer.start()
    assert provider.failure_started.wait(timeout=3.0)
    provider.release_hold.set()
    consumer.join(timeout=5.0)

    assert not consumer.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], EmbeddingBatchExecutionError)
    assert failures[0].batch_start == 1
    assert failures[0].batch_size == 1
    assert failures[0].cause_type == "RuntimeError"
    assert failures[0].provider_code is None
    assert failures[0].retryable is None
    assert failures[0].attempts_exhausted is None
    assert failures[0].__cause__ is None
    assert failures[0].__context__ is None
    assert set(provider.calls) == {"hold", "bad"}


def test_embedding_batch_failure_copies_only_safe_provider_diagnostics() -> None:
    secret = "sensitive-token-value"
    provider_error = ProviderReportedError(
        code=429,
        retryable=True,
        attempts_exhausted=True,
    )
    provider_error.response_body = {"authorization": secret}
    provider_error.request_url = f"https://provider.invalid/?key={secret}"

    def fail(_texts: list[str]) -> list[list[float]]:
        raise provider_error

    with pytest.raises(EmbeddingBatchExecutionError) as captured:
        list(
            iter_bounded_document_embedding_batches(
                texts=("one", "two"),
                batch_size=2,
                max_in_flight_batches=1,
                embed_documents=fail,
            )
        )

    failure = captured.value
    assert failure.batch_start == 0
    assert failure.batch_size == 2
    assert failure.cause_type == "ProviderReportedError"
    assert failure.provider_code == 429
    assert failure.retryable is True
    assert failure.attempts_exhausted is True
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert secret not in str(failure)
    assert secret not in repr(vars(failure))


@pytest.mark.parametrize("max_in_flight_batches", (0, -1, True))
def test_embedding_batches_require_a_positive_explicit_concurrency_limit(
    max_in_flight_batches: object,
) -> None:
    with pytest.raises(EmbeddingBatchExecutionError, match="max_in_flight_batches"):
        list(
            iter_bounded_document_embedding_batches(
                texts=("one",),
                batch_size=1,
                max_in_flight_batches=max_in_flight_batches,  # type: ignore[arg-type]
                embed_documents=lambda texts: [[float(len(texts))]],
            )
        )
