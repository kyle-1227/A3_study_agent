import { describe, expect, it } from "vitest"

import { AgentStreamContractError, parseAgentStreamEvent } from "@/lib/agent-stream-contracts"

const payload = {
  schema_version: "agent_stream_v2",
  type: "stream_start",
  stream_id: "stream-1",
  event_id: "stream-1:1",
  sequence: 1,
  request_id: "request-1",
  thread_id: "thread-1",
  created_at: "2026-07-13T00:00:00Z",
  data: {},
}

describe("parseAgentStreamEvent", () => {
  it("parses the strict envelope", () => {
    expect(parseAgentStreamEvent(payload).eventId).toBe("stream-1:1")
  })

  it("accepts assessment_final as a public agent_stream_v2 event", () => {
    const event = parseAgentStreamEvent({
      ...payload,
      type: "assessment_final",
      event_id: "stream-1:2",
      sequence: 2,
    })
    expect(event.type).toBe("assessment_final")
  })

  it("rejects unknown fields and mismatched event ids", () => {
    expect(() => parseAgentStreamEvent({ ...payload, extra: true })).toThrow(
      AgentStreamContractError,
    )
    expect(() => parseAgentStreamEvent({ ...payload, event_id: "wrong" })).toThrow(
      AgentStreamContractError,
    )
  })

  it("validates top-level evidence progress and its envelope identity", () => {
    const requestId = "00000000-0000-4000-8000-000000000001"
    const evidence = {
      schema_version: "evidence_progress_v1",
      progress_id: `evidence-progress:v1:${"a".repeat(64)}`,
      request_id: requestId,
      thread_id: "thread-1",
      lifecycle_key: "plan",
      phase_status: "completed",
      details: {
        stage: "evidence_orchestration.plan.accepted",
        requirement_count: 1,
        resource_count: 1,
        subject_count: 1,
        budget_max_rounds: 2,
        budget_max_tasks: 10,
      },
    }
    const event = parseAgentStreamEvent({
      ...payload,
      type: "evidence_progress",
      request_id: requestId,
      data: evidence,
    })
    expect(event.type).toBe("evidence_progress")

    expect(() =>
      parseAgentStreamEvent({
        ...payload,
        type: "evidence_progress",
        request_id: requestId,
        data: { ...evidence, thread_id: "thread-2" },
      }),
    ).toThrow("identity does not match")
  })
})
