import { describe, expect, it } from "vitest"

import type { Message } from "@/components/chat-area"
import {
  isCompletedWithoutResourceDiagnostic,
  mergeResourceFinalIntoMessage,
  parseResourceFinalEvent,
  resourceFinalDedupeKey,
  resourceFinalOutcome,
  type ResourceFinalEvent,
} from "@/lib/resource-final"

const RESOURCE_ID = `resource:v1:${"a".repeat(64)}`
const PAYLOAD_HASH = `payload:v1:${"b".repeat(64)}`

function rawEvent(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    type: "resource_final",
    schema_version: 2,
    resource_id: RESOURCE_ID,
    payload_hash: PAYLOAD_HASH,
    resource_type: "mindmap",
    thread_id: "thread-1",
    request_id: "request-1",
    terminal_status: "success",
    validation: {
      success_count: 1,
      partial_success_count: 0,
      failed_count: 0,
      blocked_count: 0,
      renderable_resource_count: 1,
      renderable_count: 1,
      downloadable_count: 1,
    },
    resource: {
      kind: "mindmap",
      title: "Machine learning map",
      summary: "Mindmap ready",
      payload: {
        mindmap: {
          title: "Machine learning map",
          tree: {
            title: "Machine learning",
            children: [{ title: "Supervised learning" }],
          },
          xmind_url: "/artifacts/map.xmind",
        },
      },
      artifact_refs: { xmind_url: "/artifacts/map.xmind" },
      render_hints: { primary_card: "mindmap" },
    },
    ...overrides,
  }
}

describe("resource final contract helpers", () => {
  it("strictly parses the normalized v2 event", () => {
    const event = parseResourceFinalEvent(rawEvent())
    expect(event.terminal_status).toBe("success")
    expect(event.validation.renderableCount).toBe(1)
    expect(event.resource.kind).toBe("mindmap")
  })

  it("rejects malformed terminal truth and stable ids", () => {
    expect(() =>
      parseResourceFinalEvent(rawEvent({ terminal_status: "completed" })),
    ).toThrow(/terminal_status/)
    expect(() =>
      parseResourceFinalEvent(rawEvent({ resource_id: "resource:v1:short" })),
    ).toThrow(/resource_id/)
  })

  it("keeps legacy v1 payload renderable without promoting it to success", () => {
    const value = rawEvent({ schema_version: 1 })
    delete value.terminal_status
    delete value.validation
    const event = parseResourceFinalEvent(value)
    expect(event.terminal_status).toBe("unknown")
    expect(resourceFinalOutcome(event)).toBeNull()
  })

  it("deduplicates by resource id and keeps same-type requests distinct", () => {
    const event = parseResourceFinalEvent(rawEvent())
    expect(resourceFinalDedupeKey(event)).toBe(`resource_id:${RESOURCE_ID}`)
    expect(
      resourceFinalDedupeKey({
        thread_id: "thread-1",
        request_id: "request-2",
        resource_type: "mindmap",
        payload_hash: `payload:v1:${"c".repeat(64)}`,
      }),
    ).not.toBe(
      resourceFinalDedupeKey({
        thread_id: "thread-1",
        request_id: "request-1",
        resource_type: "mindmap",
        payload_hash: PAYLOAD_HASH,
      }),
    )
  })

  it("restores a renderable resource on the matching assistant message", () => {
    const message: Message = {
      id: "assistant-1",
      role: "assistant",
      content: "",
      threadId: "thread-1",
      requestId: "request-1",
    }
    const restored = mergeResourceFinalIntoMessage(
      message,
      parseResourceFinalEvent(rawEvent()),
      "http://api.test",
    )
    expect(restored.content).toBe("Mindmap ready")
    expect(restored.mindmap?.xmindUrl).toBe("http://api.test/artifacts/map.xmind")
    expect(restored.mindmap?.tree.children?.[0].title).toBe("Supervised learning")
  })

  it("does not invent a mindmap tree when the render payload is malformed", () => {
    const event = parseResourceFinalEvent(
      rawEvent({
        resource: {
          kind: "mindmap",
          title: "Machine learning map",
          summary: "Stored summary",
          payload: { mindmap: { title: "Machine learning map", tree: {} } },
          artifact_refs: {},
          render_hints: {},
        },
      }),
    )
    const result = mergeResourceFinalIntoMessage(
      { id: "assistant-1", role: "assistant", content: "" },
      event,
      "http://api.test",
    )
    expect(result.mindmap).toBeUndefined()
  })

  it.each([
    ["success", "completed_with_resource"],
    ["partial_success", "partial_success"],
    ["failed", "failed"],
    ["controlled_stop", "controlled_stop"],
  ] as const)("maps %s terminal truth to %s", (terminalStatus, state) => {
    const event = parseResourceFinalEvent(
      rawEvent({ terminal_status: terminalStatus }),
    ) as ResourceFinalEvent
    expect(resourceFinalOutcome(event)?.state).toBe(state)
  })

  it("recognizes completed_without_resource only from its diagnostic contract", () => {
    expect(
      isCompletedWithoutResourceDiagnostic({
        type: "resource_final_diagnostic",
        status: "completed_without_resource",
      }),
    ).toBe(true)
    expect(
      isCompletedWithoutResourceDiagnostic({
        type: "done",
        status: "completed_without_resource",
      }),
    ).toBe(false)
  })
})
