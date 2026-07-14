import { describe, expect, it } from "vitest"

import type { AgentStreamEventType, AgentStreamEventV2 } from "@/lib/agent-stream-contracts"
import { LiveTurnSequenceError, reduceLiveTurn } from "@/lib/live-turn"

const REQUEST_ID = "00000000-0000-4000-8000-000000000001"

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
    requestId: REQUEST_ID,
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

  it("rejects identity drift instead of silently ignoring it", () => {
    const state = reduceLiveTurn(null, event(1, "stream_start"))
    const stale = { ...event(2, "stream_error", { message: "old" }), requestId: "old" }
    expect(() => reduceLiveTurn(state, stale)).toThrow(LiveTurnSequenceError)
  })

  it("marks an assessment final as committed", () => {
    const running = reduceLiveTurn(null, event(1, "stream_start"))
    const committed = reduceLiveTurn(running, event(2, "assessment_final"))
    expect(committed.committed).toBe(true)
  })

  it("aborts active evidence progress on stopped before stream_done", () => {
    let state = reduceLiveTurn(null, event(1, "stream_start"))
    state = reduceLiveTurn(
      state,
      event(2, "evidence_progress", {
        schema_version: "evidence_progress_v1",
        progress_id: `evidence-progress:v1:${"a".repeat(64)}`,
        request_id: REQUEST_ID,
        thread_id: "thread-1",
        lifecycle_key: "plan",
        phase_status: "completed",
        details: {
          stage: "evidence_orchestration.plan.accepted",
          requirement_count: 1,
          resource_count: 1,
          subject_count: 1,
          budget_max_rounds: 2,
          budget_max_tasks: 4,
        },
      }),
    )
    state = reduceLiveTurn(
      state,
      event(3, "evidence_progress", {
        schema_version: "evidence_progress_v1",
        progress_id: `evidence-progress:v1:${"b".repeat(64)}`,
        request_id: REQUEST_ID,
        thread_id: "thread-1",
        lifecycle_key: "round:0",
        phase_status: "running",
        details: {
          stage: "evidence_orchestration.round.started",
          round_index: 0,
          task_count: 1,
          local_task_count: 1,
          web_task_count: 0,
          budget_used_tasks: 1,
          budget_remaining_tasks: 3,
        },
      }),
    )
    expect(() => reduceLiveTurn(state, event(4, "stream_done"))).toThrow(
      "before evidence progress terminated",
    )

    state = reduceLiveTurn(state, event(4, "stopped"))
    state = reduceLiveTurn(state, event(5, "stream_done"))
    expect(state.evidenceProgress.aborted).toBe(true)
    expect(state.lifecycle).toBe("waiting")
  })
})
