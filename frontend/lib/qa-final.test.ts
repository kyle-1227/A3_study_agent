import { describe, expect, it } from "vitest"

import {
  attachQAFinalToMessages,
  parseQAFinalEvent,
  qaFinalDedupeKey,
  type QAFinalEventV1,
  type QAFinalMessage,
} from "@/lib/qa-final"

function wireEvent(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    type: "qa_final",
    schema_version: 1,
    qa_id: `qa:v1:${"a".repeat(64)}`,
    payload_hash: "b".repeat(64),
    qa_scope: "a3_agent",
    response: {
      answer: "The assistant supports grounded learning workflows.",
      uncertainty_note: "",
      grounding_status: "capability_registry",
      suggestions: [{ label: "Create a plan", action: "resource", resource_type: "study_plan" }],
    },
    thread_id: "thread-1",
    request_id: "request-1",
    created_at: "2026-07-12T00:00:00+00:00",
    ...overrides,
  }
}

function event(overrides: Record<string, unknown> = {}): QAFinalEventV1 {
  return parseQAFinalEvent(wireEvent(overrides))
}

describe("qa final contract", () => {
  it("strictly parses the backend wire payload", () => {
    const parsed = event()
    expect(parsed.schemaVersion).toBe(1)
    expect(parsed.response.answer).toContain("grounded learning")
    expect(parsed.response.suggestions[0].resourceType).toBe("study_plan")
    expect(qaFinalDedupeKey(parsed)).toBe(`qa_id:qa:v1:${"a".repeat(64)}`)
  })

  it("rejects malformed or extended payloads", () => {
    expect(() => event({ unexpected: true })).toThrow(/unexpected field/)
    expect(() => event({ qa_id: "qa:v1:not-a-hash" })).toThrow(/qa_id/)
    expect(() => event({ created_at: "2026-07-12T08:00:00" })).toThrow(/UTC/)
  })

  it("binds the answer to the matching assistant request without resource state", () => {
    const messages: QAFinalMessage[] = [
      { id: "assistant-1", role: "assistant", content: "", requestId: "request-1", threadId: "thread-1" },
    ]
    const result = attachQAFinalToMessages(messages, event(), "assistant-1")
    expect(result.attached).toBe(true)
    expect(result.messages).toHaveLength(1)
    expect(result.messages[0].content).toContain("grounded learning")
    expect(result.messages[0].qaFinal?.qaScope).toBe("a3_agent")
    expect("resourceStatus" in result.messages[0]).toBe(false)
  })

  it("is idempotent for repeated realtime or restored events", () => {
    const first = attachQAFinalToMessages<QAFinalMessage>([], event())
    const second = attachQAFinalToMessages(first.messages, event())
    expect(second.attached).toBe(false)
    expect(second.messages).toEqual(first.messages)
  })

  it("does not bind a prior request result to the current placeholder", () => {
    const messages: QAFinalMessage[] = [
      { id: "assistant-current", role: "assistant", content: "", requestId: "request-2", threadId: "thread-1" },
    ]
    const result = attachQAFinalToMessages(messages, event(), "assistant-current")
    expect(result.messages).toHaveLength(2)
    expect(result.messages[0].content).toBe("")
    expect(result.messages[1].requestId).toBe("request-1")
  })

  it("rejects a conflicting final for the same request", () => {
    const first = attachQAFinalToMessages<QAFinalMessage>([], event())
    const conflict = event({
      qa_id: `qa:v1:${"c".repeat(64)}`,
      payload_hash: "d".repeat(64),
    })
    expect(() => attachQAFinalToMessages(first.messages, conflict)).toThrow(/different QA final/)
  })
})
