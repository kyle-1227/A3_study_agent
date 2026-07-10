import { describe, expect, it } from "vitest"

import type { Message } from "@/components/chat-area"
import {
  isCompletedWithoutResourceDiagnostic,
  mergeResourceFinalIntoMessage,
  resourceFinalDedupeKey,
  type ResourceFinalEvent,
} from "@/lib/resource-final"

describe("resource final contract helpers", () => {
  it("deduplicates by stable resource id when present", () => {
    const event: ResourceFinalEvent = {
      type: "resource_final",
      resource_id: "resource:v1:stable",
      thread_id: "thread-1",
      request_id: "request-1",
      resource_type: "mindmap",
      payload_hash: "hash-1",
    }
    expect(resourceFinalDedupeKey(event)).toBe("resource_id:resource:v1:stable")
    expect(resourceFinalDedupeKey({ ...event, request_id: "request-2" })).toBe(
      resourceFinalDedupeKey(event),
    )
  })

  it("keeps multiple resources of the same type in one thread", () => {
    const first: ResourceFinalEvent = {
      type: "resource_final",
      thread_id: "thread-1",
      request_id: "request-1",
      resource_type: "mindmap",
      payload_hash: "hash-1",
    }
    const second = { ...first, request_id: "request-2", payload_hash: "hash-2" }
    expect(resourceFinalDedupeKey(second)).not.toBe(resourceFinalDedupeKey(first))
  })

  it("restores a renderable normalized resource payload onto a message", () => {
    const message: Message = { id: "assistant-1", role: "assistant", content: "" }
    const event: ResourceFinalEvent = {
      type: "resource_final",
      resource_id: "resource:v1:mindmap",
      resource_type: "mindmap",
      thread_id: "thread-1",
      request_id: "request-1",
      payload_hash: "hash-1",
      resource: {
        kind: "mindmap",
        title: "机器学习导图",
        summary: "可渲染摘要",
        payload: {
          mindmap: {
            title: "机器学习导图",
            tree: { title: "机器学习", children: [{ title: "监督学习" }] },
            xmind_url: "/artifacts/map.xmind",
          },
        },
      },
    }
    const restored = mergeResourceFinalIntoMessage(message, event, "http://api.test")
    expect(restored.content).toBe("可渲染摘要")
    expect(restored.mindmap).toEqual({
      title: "机器学习导图",
      tree: { title: "机器学习", note: undefined, children: [{ title: "监督学习", note: undefined, children: undefined }] },
      xmindUrl: "http://api.test/artifacts/map.xmind",
    })
  })

  it("recognizes completed_without_resource only from the resource diagnostic contract", () => {
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
