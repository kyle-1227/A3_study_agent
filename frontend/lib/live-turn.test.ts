import { describe, expect, it } from "vitest"

import type { AgentStreamEventType, AgentStreamEventV2 } from "@/lib/agent-stream-contracts"
import { LiveTurnSequenceError, reduceLiveTurn } from "@/lib/live-turn"

function event(
  sequence: number,
  type: AgentStreamEventType,
  data: Record<string, unknown> = {},
): AgentStreamEventV2 {
  return {
    schemaVersion: "agent_stream_v2",
    type,
    streamId: "stream-1",
    eventId: `stream-1:${sequence}`,
    sequence,
    requestId: "request-1",
    threadId: "thread-1",
    createdAt: "2026-07-13T00:00:00Z",
    data,
  }
}

describe("reduceLiveTurn", () => {
  it("builds provisional content without mutating committed messages", () => {
    let state = reduceLiveTurn(null, event(1, "stream_start"))
    state = reduceLiveTurn(
      state,
      event(2, "content_block_start", {
        block_id: "answer",
        block_index: 0,
        block_type: "markdown",
        provisional: true,
      }),
    )
    state = reduceLiveTurn(
      state,
      event(3, "content_block_delta", { block_id: "answer", delta: "你好" }),
    )
    expect(state.provisionalAnswer).toBe("你好")
    expect(state.committed).toBe(false)
  })

  it("deduplicates replayed events and rejects gaps", () => {
    const state = reduceLiveTurn(null, event(1, "stream_start"))
    expect(reduceLiveTurn(state, event(1, "stream_start"))).toBe(state)
    expect(() => reduceLiveTurn(state, event(3, "activity_update"))).toThrow(
      LiveTurnSequenceError,
    )
  })

  it("ignores an old request instead of overwriting the active turn", () => {
    const state = reduceLiveTurn(null, event(1, "stream_start"))
    const stale = { ...event(2, "stream_error", { message: "old" }), requestId: "old" }
    expect(reduceLiveTurn(state, stale)).toBe(state)
  })

  it("marks an assessment final as committed", () => {
    const running = reduceLiveTurn(null, event(1, "stream_start"))
    const committed = reduceLiveTurn(running, event(2, "assessment_final"))
    expect(committed.committed).toBe(true)
  })
})
