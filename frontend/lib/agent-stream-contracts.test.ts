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
})
