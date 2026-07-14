import { describe, expect, it, vi } from "vitest"

import { consumeAgentStreamV2 } from "@/lib/agent-stream-client"

const encoder = new TextEncoder()
const requestId = "00000000-0000-4000-8000-000000000001"

function event(sequence: number, type: string): string {
  const payload = {
    schema_version: "agent_stream_v2",
    type,
    stream_id: "stream-1",
    event_id: `stream-1:${sequence}`,
    sequence,
    request_id: requestId,
    thread_id: "thread-1",
    created_at: "2026-07-13T00:00:00Z",
    data: {},
  }
  return `event: ${type}\nid: stream-1:${sequence}\n${sequence === 1 ? "retry: 1\n" : ""}data: ${JSON.stringify(payload)}\n\n`
}

function body(value: string): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(value))
      controller.close()
    },
  })
}

describe("consumeAgentStreamV2", () => {
  it("validates and delivers a complete stream", async () => {
    const received: string[] = []
    await consumeAgentStreamV2({
      initialBody: body(event(1, "stream_start") + event(2, "stream_error") + event(3, "stream_done")),
      onEvent: (item) => received.push(item.type),
      reconnect: vi.fn(),
    })
    expect(received).toEqual(["stream_start", "stream_error", "stream_done"])
  })

  it("replays from the last complete event after EOF", async () => {
    vi.useFakeTimers()
    const received: string[] = []
    const reconnect = vi.fn(async () => body(event(2, "stream_error") + event(3, "stream_done")))
    const promise = consumeAgentStreamV2({
      initialBody: body(event(1, "stream_start")),
      onEvent: (item) => received.push(item.type),
      reconnect,
    })
    await vi.runAllTimersAsync()
    await promise
    expect(reconnect).toHaveBeenCalledWith("stream-1", "stream-1:1", undefined)
    expect(received).toEqual(["stream_start", "stream_error", "stream_done"])
    vi.useRealTimers()
  })

  it("rejects a transport id mismatch", async () => {
    const mismatched = event(1, "stream_start").replace("id: stream-1:1", "id: stream-1:2")
    await expect(
      consumeAgentStreamV2({
        initialBody: body(mismatched),
        onEvent: vi.fn(),
        reconnect: vi.fn(),
      }),
    ).rejects.toThrow("SSE id does not match")
  })

  it("rejects identity drift and sequence gaps before dispatch", async () => {
    const identityDrift = event(2, "stream_error").replace(
      '"thread_id":"thread-1"',
      '"thread_id":"thread-2"',
    )
    await expect(
      consumeAgentStreamV2({
        initialBody: body(event(1, "stream_start") + identityDrift),
        onEvent: vi.fn(),
        reconnect: vi.fn(),
      }),
    ).rejects.toThrow("stream event identity changed")

    await expect(
      consumeAgentStreamV2({
        initialBody: body(event(1, "stream_start") + event(3, "stream_error")),
        onEvent: vi.fn(),
        reconnect: vi.fn(),
      }),
    ).rejects.toThrow("stream sequence gap")
  })

  it("ignores exact replay duplicates and rejects conflicting replay", async () => {
    vi.useFakeTimers()
    const received: string[] = []
    const reconnect = vi.fn(async () =>
      body(event(1, "stream_start") + event(2, "stream_error") + event(3, "stream_done")),
    )
    const replay = consumeAgentStreamV2({
      initialBody: body(event(1, "stream_start")),
      onEvent: (item) => received.push(item.eventId),
      reconnect,
    })
    await vi.runAllTimersAsync()
    await replay
    expect(received).toEqual(["stream-1:1", "stream-1:2", "stream-1:3"])

    const conflicting = event(1, "stream_start").replace('"data":{}', '"data":{"drift":true}')
    const conflictReconnect = vi.fn(async () => body(conflicting))
    const conflict = consumeAgentStreamV2({
      initialBody: body(event(1, "stream_start")),
      onEvent: vi.fn(),
      reconnect: conflictReconnect,
    })
    const rejection = expect(conflict).rejects.toThrow(
      "replayed stream sequence conflicts",
    )
    await vi.runAllTimersAsync()
    await rejection
    vi.useRealTimers()
  })
})
