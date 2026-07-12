"""Bounded performance report registry and authenticated browser ingestion."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from src.observability.a3_trace import emit_a3_trace
from src.observability.performance_config import (
    PerformanceObservabilityConfig,
    load_performance_observability_config,
    resolve_frontend_performance_secret,
)
from src.observability.performance_contracts import (
    PERFORMANCE_EVENT_SCHEMA_VERSION,
    FrontendSampleStatus,
    FrontendPerformanceBatchV1,
    PerformanceFrontendBatchEventV1,
    PerformanceRequestReportV1,
)
from src.observability.performance_runtime import (
    PerformanceRecorder,
    build_performance_report,
    performance_report_trace_payload,
    performance_request_recorder,
    stable_performance_id,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

_CURRENT_CAPABILITY: ContextVar[dict[str, Any] | None] = ContextVar(
    "a3_frontend_performance_capability",
    default=None,
)
_SERVICE_LOCK = threading.RLock()
_SERVICE: PerformanceService | None = None


class FrontendPerformanceRejected(RuntimeError):
    def __init__(self, code: str, *, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


@dataclass(slots=True)
class _RequestBinding:
    request_id: str
    thread_id: str
    trace_id: str
    root_span_id: str
    user_fingerprint: str
    token_digest: str
    expires_epoch: int
    expires_at: str
    capability_exposed: bool = False
    consumed: bool = False
    milestone_count: int = 0
    frontend_batch: FrontendPerformanceBatchV1 | None = None
    report: PerformanceRequestReportV1 | None = None


class PerformanceService:
    """Process-local bounded registry; never writes LearningState or checkpoints."""

    def __init__(
        self,
        config: PerformanceObservabilityConfig,
        *,
        secret: bytes | None,
    ) -> None:
        self.config = config
        self._secret = secret
        self._bindings: OrderedDict[str, _RequestBinding] = OrderedDict()
        self._reports: OrderedDict[str, PerformanceRequestReportV1] = OrderedDict()
        self._lock = threading.RLock()

    def register_request(
        self,
        *,
        recorder: PerformanceRecorder,
        user_id: str,
    ) -> dict[str, Any]:
        frontend = self.config.frontend_ingestion
        if not frontend.enabled:
            return {}
        if self._secret is None:
            raise RuntimeError("frontend performance ingestion secret is unavailable")
        now_epoch = int(time.time())
        expires_epoch = now_epoch + frontend.token_ttl_seconds
        expires_at = datetime.fromtimestamp(expires_epoch, tz=timezone.utc).isoformat()
        user_fingerprint = hashlib.sha256(
            str(user_id or "anonymous").encode("utf-8")
        ).hexdigest()
        claims = {
            "version": 1,
            "request_id": recorder.request_id,
            "thread_id": recorder.thread_id,
            "trace_id": recorder.trace_id,
            "user_fingerprint": user_fingerprint,
            "expires_epoch": expires_epoch,
        }
        token = self._sign_claims(claims)
        token_digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        binding = _RequestBinding(
            request_id=recorder.request_id,
            thread_id=recorder.thread_id,
            trace_id=recorder.trace_id,
            root_span_id=recorder.root_span_id,
            user_fingerprint=user_fingerprint,
            token_digest=token_digest,
            expires_epoch=expires_epoch,
            expires_at=expires_at,
        )
        with self._lock:
            self._purge_locked(now_epoch)
            self._bindings[token_digest] = binding
            self._trim_locked()
        return {
            "schema_version": "frontend_performance_capability_v1",
            "enabled": True,
            "endpoint": frontend.endpoint_path,
            "trace_id": recorder.trace_id,
            "token": token,
            "expires_at": expires_at,
        }

    def mark_capability_exposed(self, token: str) -> None:
        token_digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        with self._lock:
            binding = self._bindings.get(token_digest)
            if binding is not None:
                binding.capability_exposed = True

    def store_report(
        self,
        recorder: PerformanceRecorder,
        *,
        token: str = "",
    ) -> PerformanceRequestReportV1:
        frontend_status: FrontendSampleStatus
        token_digest = (
            hashlib.sha256(token.encode("ascii")).hexdigest() if token else ""
        )
        with self._lock:
            self._purge_locked(int(time.time()))
            binding = self._bindings.get(token_digest) if token_digest else None
            if not self.config.frontend_ingestion.enabled:
                frontend_status = "not_requested"
                milestone_count = 0
            elif binding is None or not binding.capability_exposed:
                frontend_status = "incomplete"
                milestone_count = 0
            elif binding.consumed:
                frontend_status = "accepted"
                milestone_count = binding.milestone_count
            else:
                frontend_status = "pending"
                milestone_count = 0
        report = build_performance_report(
            recorder,
            frontend_sample_status=frontend_status,
            frontend_milestone_count=milestone_count,
        )
        with self._lock:
            self._reports[recorder.request_id] = report
            self._reports.move_to_end(recorder.request_id)
            if binding is not None:
                binding.report = report
            self._trim_locked()
        return report

    def accept_frontend_batch(
        self,
        *,
        authorization: str,
        origin: str,
        raw_size: int,
        payload: FrontendPerformanceBatchV1,
    ) -> PerformanceFrontendBatchEventV1:
        frontend = self.config.frontend_ingestion
        if not frontend.enabled:
            raise FrontendPerformanceRejected(
                "frontend_performance_disabled",
                status_code=503,
            )
        normalized_origin = str(origin or "").strip().rstrip("/")
        if normalized_origin not in frontend.allowed_origins:
            raise FrontendPerformanceRejected(
                "frontend_performance_origin_rejected",
                status_code=403,
            )
        if raw_size > frontend.max_payload_bytes:
            raise FrontendPerformanceRejected(
                "frontend_performance_payload_too_large",
                status_code=413,
            )
        if len(payload.milestones) > frontend.max_milestones_per_request:
            raise FrontendPerformanceRejected(
                "frontend_performance_milestone_limit_exceeded",
                status_code=413,
            )
        token = _bearer_token(authorization)
        claims = self._verify_token(token)
        token_digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        now_epoch = int(time.time())
        with self._lock:
            self._purge_locked(now_epoch)
            binding = self._bindings.get(token_digest)
            if binding is None:
                raise FrontendPerformanceRejected(
                    "frontend_performance_request_binding_missing",
                    status_code=401,
                )
            if binding.consumed:
                raise FrontendPerformanceRejected(
                    "frontend_performance_replay_rejected",
                    status_code=409,
                )
            if not _claims_match_binding(claims, binding) or (
                payload.request_id != binding.request_id
                or payload.thread_id != binding.thread_id
                or payload.trace_id != binding.trace_id
            ):
                raise FrontendPerformanceRejected(
                    "frontend_performance_binding_mismatch",
                    status_code=403,
                )
            binding.consumed = True
            binding.milestone_count = len(payload.milestones)
            binding.frontend_batch = payload
            if binding.report is not None:
                binding.report = _validated_report_update(
                    binding.report,
                    frontend_sample_status="accepted",
                    frontend_milestone_count=binding.milestone_count,
                )
                self._reports[binding.request_id] = binding.report
            event = PerformanceFrontendBatchEventV1(
                schema_version=PERFORMANCE_EVENT_SCHEMA_VERSION,
                stage="performance.frontend.batch.accepted",
                trace_id=binding.trace_id,
                span_id=stable_performance_id(
                    "span",
                    {"trace_id": binding.trace_id, "stage": "frontend.milestones"},
                ),
                parent_span_id=binding.root_span_id,
                operation_id=stable_performance_id(
                    "operation",
                    {"trace_id": binding.trace_id, "operation": "frontend.milestones"},
                ),
                request_id=binding.request_id,
                thread_id=binding.thread_id,
                operation_type="request",
                operation_name="frontend.milestones",
                status="completed",
                occurred_at=utc_now_iso(),
                monotonic_clock_source="browser.performance_now",
                milestone_count=binding.milestone_count,
            )
        emit_a3_trace(
            logger,
            event.stage,
            {
                "schema_version": event.schema_version,
                "trace_id": event.trace_id,
                "span_id": event.span_id,
                "parent_span_id": event.parent_span_id,
                "operation_id": event.operation_id,
                "milestone_count": event.milestone_count,
                "monotonic_clock_source": event.monotonic_clock_source,
            },
            state={"request_id": event.request_id, "thread_id": event.thread_id},
            env_flag="LOG_PERFORMANCE_TRACE",
            level="info",
        )
        return event

    def get_report(self, request_id: str) -> PerformanceRequestReportV1 | None:
        with self._lock:
            self._purge_locked(int(time.time()))
            report = self._reports.get(request_id)
            if report is not None:
                self._reports.move_to_end(request_id)
            return report

    def get_frontend_batch(self, request_id: str) -> FrontendPerformanceBatchV1 | None:
        with self._lock:
            self._purge_locked(int(time.time()))
            for binding in self._bindings.values():
                if binding.request_id == request_id:
                    return binding.frontend_batch
        return None

    def _sign_claims(self, claims: dict[str, Any]) -> str:
        if self._secret is None:
            raise RuntimeError("frontend performance signing is not configured")
        encoded = _b64encode(
            json.dumps(
                claims,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        signature = _b64encode(
            hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        return f"{encoded}.{signature}"

    def _verify_token(self, token: str) -> dict[str, Any]:
        if self._secret is None:
            raise FrontendPerformanceRejected(
                "frontend_performance_signing_unavailable",
                status_code=503,
            )
        try:
            encoded, signature = token.split(".", 1)
            expected = _b64encode(
                hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(signature, expected):
                raise ValueError("signature mismatch")
            claims = json.loads(_b64decode(encoded).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FrontendPerformanceRejected(
                "frontend_performance_token_invalid",
                status_code=401,
            ) from exc
        required = {
            "version",
            "request_id",
            "thread_id",
            "trace_id",
            "user_fingerprint",
            "expires_epoch",
        }
        if not isinstance(claims, dict) or set(claims) != required:
            raise FrontendPerformanceRejected(
                "frontend_performance_token_claims_invalid",
                status_code=401,
            )
        if claims.get("version") != 1 or not isinstance(
            claims.get("expires_epoch"), int
        ):
            raise FrontendPerformanceRejected(
                "frontend_performance_token_claims_invalid",
                status_code=401,
            )
        if int(claims["expires_epoch"]) < int(time.time()):
            raise FrontendPerformanceRejected(
                "frontend_performance_token_expired",
                status_code=401,
            )
        return claims

    def _purge_locked(self, now_epoch: int) -> None:
        expired = [
            digest
            for digest, binding in self._bindings.items()
            if binding.expires_epoch <= now_epoch
        ]
        for digest in expired:
            binding = self._bindings.pop(digest)
            if (
                binding.report is not None
                and binding.report.frontend_sample_status == "pending"
            ):
                report = _validated_report_update(
                    binding.report,
                    frontend_sample_status="incomplete",
                    frontend_milestone_count=binding.report.frontend_milestone_count,
                )
                self._reports[binding.request_id] = report

    def _trim_locked(self) -> None:
        limit = self.config.report_retention_count
        while len(self._bindings) > limit:
            self._bindings.popitem(last=False)
        while len(self._reports) > limit:
            self._reports.popitem(last=False)


def configure_performance_service(
    config: PerformanceObservabilityConfig | None = None,
) -> PerformanceService:
    global _SERVICE
    resolved = config or load_performance_observability_config()
    secret = resolve_frontend_performance_secret(resolved)
    service = PerformanceService(resolved, secret=secret)
    with _SERVICE_LOCK:
        _SERVICE = service
    return service


def get_performance_service() -> PerformanceService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = configure_performance_service()
        return _SERVICE


def reset_performance_service_for_tests() -> None:
    global _SERVICE
    with _SERVICE_LOCK:
        _SERVICE = None


@contextmanager
def observe_request_performance(
    *,
    request_id: str,
    thread_id: str,
    user_id: str = "",
) -> Iterator[PerformanceRecorder | None]:
    service = get_performance_service()
    if not service.config.enabled:
        yield None
        return
    recorder: PerformanceRecorder | None = None
    capability: dict[str, Any] = {}
    capability_token = ""
    token = None
    try:
        with performance_request_recorder(
            request_id=request_id,
            thread_id=thread_id,
            max_spans=service.config.max_spans_per_request,
        ) as recorder:
            capability = service.register_request(recorder=recorder, user_id=user_id)
            capability_token = str(capability.get("token") or "")
            token = _CURRENT_CAPABILITY.set(capability or None)
            yield recorder
    finally:
        if token is not None:
            _CURRENT_CAPABILITY.reset(token)
        if recorder is not None and recorder.events:
            try:
                report = service.store_report(recorder, token=capability_token)
                emit_a3_trace(
                    logger,
                    "performance.request.reported",
                    performance_report_trace_payload(report),
                    state={"request_id": request_id, "thread_id": thread_id},
                    env_flag="LOG_PERFORMANCE_TRACE",
                    level="info",
                )
            except Exception as exc:
                emit_a3_trace(
                    logger,
                    "performance.request.report_failed",
                    {"error_type": type(exc).__name__},
                    state={"request_id": request_id, "thread_id": thread_id},
                    env_flag="LOG_PERFORMANCE_TRACE",
                    level="info",
                )


def current_frontend_performance_capability() -> dict[str, Any]:
    capability = _CURRENT_CAPABILITY.get()
    if not capability:
        return {}
    token = str(capability.get("token") or "")
    if token:
        get_performance_service().mark_capability_exposed(token)
    return dict(capability)


def _bearer_token(value: str) -> str:
    scheme, separator, token = str(value or "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        raise FrontendPerformanceRejected(
            "frontend_performance_authorization_required",
            status_code=401,
        )
    return token.strip()


def _claims_match_binding(claims: dict[str, Any], binding: _RequestBinding) -> bool:
    return (
        claims.get("request_id") == binding.request_id
        and claims.get("thread_id") == binding.thread_id
        and claims.get("trace_id") == binding.trace_id
        and claims.get("user_fingerprint") == binding.user_fingerprint
        and claims.get("expires_epoch") == binding.expires_epoch
    )


def _validated_report_update(
    report: PerformanceRequestReportV1,
    *,
    frontend_sample_status: FrontendSampleStatus,
    frontend_milestone_count: int,
) -> PerformanceRequestReportV1:
    payload = report.model_dump(mode="python")
    payload.update(
        {
            "frontend_sample_status": frontend_sample_status,
            "frontend_milestone_count": frontend_milestone_count,
        }
    )
    return PerformanceRequestReportV1.model_validate(payload)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


__all__ = [
    "FrontendPerformanceRejected",
    "PerformanceService",
    "configure_performance_service",
    "current_frontend_performance_capability",
    "get_performance_service",
    "observe_request_performance",
    "reset_performance_service_for_tests",
]
