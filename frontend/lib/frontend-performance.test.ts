import { describe, expect, it, vi } from "vitest"

import { FrontendPerformanceTracker } from "@/lib/frontend-performance"

function capability() {
  return {
    schemaVersion: "frontend_performance_capability_v1" as const,
    endpoint: "/observability/frontend-performance",
    traceId: `trace:v1:${"a".repeat(64)}`,
    token: "signed.capability",
    expiresAt: "2026-07-12T00:00:00+00:00",
  }
}

describe("FrontendPerformanceTracker", () => {
  it("records browser-relative milestones and delivers exactly one terminal batch", async () => {
    let now = 100
    const tracker = new FrontendPerformanceTracker({ now: () => now })
    tracker.recordEvent("thread_id")
    now = 104.5
    tracker.recordEvent("stream_context")
    tracker.bind(capability(), { requestId: "request-1", threadId: "thread-1" })
    now = 120
    tracker.recordEvent("resource_final")
    now = 145.25
    tracker.recordEvent("done")
    tracker.recordEvent("error")

    const fetchImpl = vi.fn().mockResolvedValue({ status: 204 } as Response)
    const result = await tracker.deliver(fetchImpl as unknown as typeof fetch, "http://127.0.0.1:8000")

    expect(result).toEqual({ status: "delivered" })
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("http://127.0.0.1:8000/observability/frontend-performance")
    expect(init?.headers).toMatchObject({ Authorization: "Bearer signed.capability" })
    const body = JSON.parse(String(init?.body))
    expect(body).toMatchObject({
      schema_version: "frontend_performance_v1",
      request_id: "request-1",
      thread_id: "thread-1",
      trace_id: capability().traceId,
    })
    expect(body.milestones).toEqual([
      { name: "submit_to_first_event", duration_ms: 0 },
      { name: "submit_to_stream_context", duration_ms: 4.5 },
      { name: "submit_to_resource_final", duration_ms: 20 },
      { name: "submit_to_done", duration_ms: 45.25, status: "completed" },
    ])
    expect(await tracker.deliver(fetchImpl as unknown as typeof fetch, "http://127.0.0.1:8000")).toEqual({ status: "skipped" })
  })

  it("does not send without a capability and reports non-204 delivery as incomplete", async () => {
    let now = 0
    const withoutCapability = new FrontendPerformanceTracker({ now: () => now })
    now = 5
    withoutCapability.recordEvent("done")
    const fetchImpl = vi.fn()
    expect(await withoutCapability.deliver(fetchImpl as typeof fetch, "http://api")).toEqual({ status: "skipped" })
    expect(fetchImpl).not.toHaveBeenCalled()

    const withCapability = new FrontendPerformanceTracker({ now: () => now })
    withCapability.bind(capability(), { requestId: "request-1", threadId: "thread-1" })
    now = 7
    withCapability.recordEvent("interrupt")
    const failedFetch = vi.fn().mockResolvedValue({ status: 503 } as Response)
    expect(await withCapability.deliver(failedFetch as unknown as typeof fetch, "http://api")).toEqual({
      status: "incomplete",
      reason: "frontend_performance_http_503",
    })

    const networkFailure = new FrontendPerformanceTracker({ now: () => now })
    networkFailure.bind(capability(), { requestId: "request-1", threadId: "thread-1" })
    networkFailure.recordEvent("error")
    const rejectedFetch = vi.fn().mockRejectedValue(new TypeError("network unavailable"))
    expect(await networkFailure.deliver(rejectedFetch as unknown as typeof fetch, "http://api")).toEqual({
      status: "incomplete",
      reason: "frontend_performance_delivery_failed",
    })
  })

  it("rejects conflicting capability bindings", () => {
    const tracker = new FrontendPerformanceTracker({ now: () => 0 })
    tracker.bind(capability(), { requestId: "request-1", threadId: "thread-1" })

    expect(() =>
      tracker.bind(capability(), { requestId: "request-2", threadId: "thread-1" }),
    ).toThrow("frontend_performance_capability_binding_mismatch")
  })
})
