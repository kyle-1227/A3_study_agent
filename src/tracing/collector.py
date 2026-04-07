"""OpenTelemetry collector -- TracerProvider setup with OTLP + SQLite fallback.

Timeout strategy
~~~~~~~~~~~~~~~~
When the OTLP collector (Jaeger) is unreachable the gRPC exporter enters an
internal retry loop with exponential back-off.  With the SDK defaults
(timeout=10 s, up to 3 retries) a single ``export()`` call can block for ~33 s.
Because ``BatchSpanProcessor`` runs that call on a daemon thread that holds the
GIL, the asyncio event-loop is starved and SSE responses stall.

Fix: set a short per-RPC ``timeout`` on the exporter **and** a short
``export_timeout_millis`` on the processor so traces are silently dropped when
the collector is down, rather than blocking the application.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None

# ── Timeout knobs (seconds / milliseconds) ──────────────────────────────
_OTLP_TIMEOUT_SEC = 5          # gRPC deadline *and* SDK total retry ceiling
_BATCH_EXPORT_TIMEOUT_MS = 8000  # max wait for a single batch export
_BATCH_SCHEDULE_DELAY_MS = 5000  # how often the processor flushes
_SHUTDOWN_TIMEOUT_MS = 5000      # max wait during provider.shutdown()


def setup_tracing() -> TracerProvider | None:
    """Initialize the OpenTelemetry TracerProvider with configured exporters.

    Reads configuration from environment variables:
        OTEL_TRACING_ENABLED  -- "true"/"false" kill switch (default "true")
        OTEL_SERVICE_NAME     -- resource service name (default "gaokao-tutor")
        OTEL_TRACES_EXPORTER  -- "otlp", "sqlite", or "none" (default "otlp")
        OTEL_EXPORTER_OTLP_ENDPOINT -- gRPC endpoint (default "localhost:4317")
        OTEL_SQLITE_FALLBACK_PATH   -- SQLite DB path (default "logs/traces.db")

    Returns:
        The configured TracerProvider, or None if tracing is disabled.
    """
    global _tracer_provider

    enabled = os.getenv("OTEL_TRACING_ENABLED", "true").lower()
    if enabled != "true":
        logger.info("OpenTelemetry tracing is disabled (OTEL_TRACING_ENABLED=%s)", enabled)
        return None

    service_name = os.getenv("OTEL_SERVICE_NAME", "gaokao-tutor")
    exporter_type = os.getenv("OTEL_TRACES_EXPORTER", "otlp").lower()

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    # Primary exporter: OTLP to Jaeger
    if exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
            otlp_exporter = OTLPSpanExporter(
                endpoint=endpoint,
                insecure=True,
                timeout=_OTLP_TIMEOUT_SEC,
            )
            provider.add_span_processor(
                BatchSpanProcessor(
                    otlp_exporter,
                    export_timeout_millis=_BATCH_EXPORT_TIMEOUT_MS,
                    schedule_delay_millis=_BATCH_SCHEDULE_DELAY_MS,
                )
            )
            logger.info("OTLP exporter configured -> %s (timeout=%ss)", endpoint, _OTLP_TIMEOUT_SEC)
        except Exception:
            logger.exception("Failed to configure OTLP exporter, continuing with fallback only")

    # SQLite fallback (always added unless exporter is "none")
    if exporter_type != "none":
        try:
            from src.tracing.sqlite_exporter import SQLiteSpanExporter

            db_path = os.getenv("OTEL_SQLITE_FALLBACK_PATH", "logs/traces.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            sqlite_exporter = SQLiteSpanExporter(db_path)
            provider.add_span_processor(BatchSpanProcessor(sqlite_exporter))
            logger.info("SQLite fallback exporter configured -> %s", db_path)
        except Exception:
            logger.exception("Failed to configure SQLite fallback exporter")

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    logger.info(
        "OpenTelemetry tracing initialized (service=%s, exporter=%s)",
        service_name,
        exporter_type,
    )
    return provider


def get_tracer(name: str = "gaokao_tutor") -> trace.Tracer:
    """Return a Tracer instance. Safe to call even if tracing is not initialized."""
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the TracerProvider.

    Uses a bounded timeout so the application never blocks at exit waiting for
    an unreachable collector.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.force_flush(timeout_millis=_SHUTDOWN_TIMEOUT_MS)
        except Exception:
            logger.warning("Timeout flushing remaining traces — dropping them")
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry tracing shut down")
        _tracer_provider = None
