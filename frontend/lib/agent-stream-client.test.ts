import { describe, expect, it, vi } from "vitest"

import {
  AgentStreamReplayRecoveryError,
  classifyAgentStreamThreadStatusRecovery,
  consumeAgentStreamV2,
  type AgentStreamRecoveryIdentity,
  type AgentStreamStatusRecoveryResult,
  validateAgentStreamThreadStatusIdentity,
} from "@/lib/agent-stream-client"

const encoder = new TextEncoder()
const requestId = "00000000-0000-4000-8000-000000000001"
const userId = "u_stream_recovery"
const recoveryIdentity: AgentStreamRecoveryIdentity = {
  userId,
  streamId: "stream-1",
  requestId,
  threadId: "thread-1",
  lastEventId: "stream-1:1",
}

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

function statusPayload(runStatus: unknown, threadId = "thread-1") {
  return {
    schema_version: "run_control_v1",
    run_status: runStatus,
    thread_id: threadId,
  }
}

describe("authoritative thread status recovery contracts", () => {
  it.each(["completed", "failed", "stopped"] as const)(
    "accepts only the exact %s terminal and preserves the full identity",
    (status) => {
      expect(
        classifyAgentStreamThreadStatusRecovery(
          statusPayload(status),
          recoveryIdentity,
          userId,
          requestId,
        ),
      ).toEqual({ ...recoveryIdentity, status })
    },
  )

  it.each(["running", "stopping", "continuing", "idle", "not_resumable", "unknown", "error"])(
    "rejects non-terminal or legacy terminal spelling %s",
    (status) => {
      expect(() =>
        classifyAgentStreamThreadStatusRecovery(
          statusPayload(status),
          recoveryIdentity,
          userId,
          requestId,
        ),
      ).toThrow("thread status recovery is not terminal")
    },
  )

  it("rejects legacy, malformed, and user/thread/request identity drift", () => {
    expect(() =>
      classifyAgentStreamThreadStatusRecovery(
        { ...statusPayload("completed"), schema_version: "legacy" },
        recoveryIdentity,
        userId,
        requestId,
      ),
    ).toThrow("thread status recovery schema is invalid")
    expect(() =>
      classifyAgentStreamThreadStatusRecovery(
        null,
        recoveryIdentity,
        userId,
        requestId,
      ),
    ).toThrow("thread status recovery contract is invalid")
    expect(() =>
      classifyAgentStreamThreadStatusRecovery(
        statusPayload("completed"),
        recoveryIdentity,
        "u_other",
        requestId,
      ),
    ).toThrow("thread status recovery identity mismatch")
    expect(() =>
      classifyAgentStreamThreadStatusRecovery(
        statusPayload("completed", "thread-2"),
        recoveryIdentity,
        userId,
        requestId,
      ),
    ).toThrow("thread status identity mismatch")
    expect(() =>
      classifyAgentStreamThreadStatusRecovery(
        statusPayload("completed"),
        recoveryIdentity,
        userId,
        "00000000-0000-4000-8000-000000000099",
      ),
    ).toThrow("thread status recovery identity mismatch")
  })

  it("does not treat an absent optional recovery identity as ordinary refresh drift", () => {
    expect(() =>
      validateAgentStreamThreadStatusIdentity("thread-1", "thread-1"),
    ).not.toThrow()
    expect(() =>
      validateAgentStreamThreadStatusIdentity("thread-2", "thread-1"),
    ).toThrow("thread status identity mismatch")
  })
})

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

  it.each(["transport", "expired"] as const)(
    "recovers exactly once from authoritative status after %s replay failure",
    async (reason) => {
      vi.useFakeTimers()
      try {
        const reconnect = vi.fn(async () => {
          throw new AgentStreamReplayRecoveryError(reason)
        })
        const recoverStatus = vi.fn(async () => ({
          ...recoveryIdentity,
          status: "completed" as const,
        }))
        const promise = consumeAgentStreamV2({
          initialBody: body(event(1, "stream_start")),
          onEvent: vi.fn(),
          reconnect,
          recoverStatus,
          recoveryUserId: userId,
        })

        await vi.runAllTimersAsync()
        await promise

        expect(reconnect).toHaveBeenCalledTimes(1)
        expect(recoverStatus).toHaveBeenCalledTimes(1)
        expect(recoverStatus).toHaveBeenCalledWith(recoveryIdentity, undefined)
      } finally {
        vi.useRealTimers()
      }
    },
  )

  it("rejects non-terminal or mismatched recovery results without another replay", async () => {
    vi.useFakeTimers()
    try {
      const reconnect = vi.fn(async () => {
        throw new AgentStreamReplayRecoveryError("transport")
      })
      const nonTerminalRecovery = vi.fn(async () => ({
        ...recoveryIdentity,
        status: "running",
      }) as unknown as AgentStreamStatusRecoveryResult)
      const nonTerminal = consumeAgentStreamV2({
        initialBody: body(event(1, "stream_start")),
        onEvent: vi.fn(),
        reconnect,
        recoverStatus: nonTerminalRecovery,
        recoveryUserId: userId,
      })
      const nonTerminalRejection = expect(nonTerminal).rejects.toThrow(
        "thread status recovery contract is invalid",
      )
      await vi.runAllTimersAsync()
      await nonTerminalRejection
      expect(reconnect).toHaveBeenCalledTimes(1)
      expect(nonTerminalRecovery).toHaveBeenCalledTimes(1)

      const mismatchedRecovery = vi.fn(async () => ({
        ...recoveryIdentity,
        status: "completed" as const,
        requestId: "00000000-0000-4000-8000-000000000099",
      }))
      const mismatched = consumeAgentStreamV2({
        initialBody: body(event(1, "stream_start")),
        onEvent: vi.fn(),
        reconnect,
        recoverStatus: mismatchedRecovery,
        recoveryUserId: userId,
      })
      const mismatchRejection = expect(mismatched).rejects.toThrow(
        "thread status recovery identity mismatch",
      )
      await vi.runAllTimersAsync()
      await mismatchRejection
      expect(reconnect).toHaveBeenCalledTimes(2)
      expect(mismatchedRecovery).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not call status recovery without the original user identity", async () => {
    vi.useFakeTimers()
    try {
      const recoverStatus = vi.fn()
      const promise = consumeAgentStreamV2({
        initialBody: body(event(1, "stream_start")),
        onEvent: vi.fn(),
        reconnect: vi.fn(async () => {
          throw new AgentStreamReplayRecoveryError("transport")
        }),
        recoverStatus,
      })
      const rejection = expect(promise).rejects.toThrow(
        "agent stream status recovery user identity is unavailable",
      )
      await vi.runAllTimersAsync()
      await rejection
      expect(recoverStatus).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })

  it.each([401, 404, 409, 500])(
    "does not use status recovery for replay HTTP %s",
    async (status) => {
      vi.useFakeTimers()
      try {
        const recoverStatus = vi.fn()
        const promise = consumeAgentStreamV2({
          initialBody: body(event(1, "stream_start")),
          onEvent: vi.fn(),
          reconnect: vi.fn(async () => {
            throw new Error(`Stream replay failed: HTTP ${status}`)
          }),
          recoverStatus,
          recoveryUserId: userId,
        })
        const rejection = expect(promise).rejects.toThrow(`HTTP ${status}`)
        await vi.runAllTimersAsync()
        await rejection
        expect(recoverStatus).not.toHaveBeenCalled()
      } finally {
        vi.useRealTimers()
      }
    },
  )

  it("does not use status recovery for replay contract errors", async () => {
    vi.useFakeTimers()
    try {
      const identityDrift = event(2, "stream_error").replace(
        '"thread_id":"thread-1"',
        '"thread_id":"thread-2"',
      )
      const recoverStatus = vi.fn()
      const replay = consumeAgentStreamV2({
        initialBody: body(event(1, "stream_start")),
        onEvent: vi.fn(),
        reconnect: vi.fn(async () => body(identityDrift)),
        recoverStatus,
      })
      const rejection = expect(replay).rejects.toThrow("stream event identity changed")
      await vi.runAllTimersAsync()
      await rejection
      expect(recoverStatus).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
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
