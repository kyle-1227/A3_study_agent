import { describe, expect, it } from "vitest"

import type { Message } from "@/components/chat-area"
import {
  attachActivityToAssistantMessage,
  restoreActivitiesToMessages,
} from "@/lib/message-activity"
import { parseActivityEvent } from "@/lib/observability-contracts"
import { activityPayload } from "@/test/observability-fixtures"

describe("assistant message activity", () => {
  it("attaches and deduplicates activity on the active assistant message", () => {
    const messages: Message[] = [
      { id: "user-1", role: "user", content: "question" },
      { id: "assistant-1", role: "assistant", content: "" },
    ]
    const event = parseActivityEvent(activityPayload())
    const once = attachActivityToAssistantMessage(messages, event, "assistant-1")
    const twice = attachActivityToAssistantMessage(once, event, "assistant-1")
    expect(twice[1].requestId).toBe("request-fixture")
    expect(twice[1].activities).toHaveLength(1)
  })

  it("restores unmatched persisted activity as an assistant activity message", () => {
    const event = parseActivityEvent(activityPayload())
    const restored = restoreActivitiesToMessages(
      [{ id: "user-1", role: "user", content: "question" }],
      [event],
      "thread-fixture",
    )
    expect(restored).toHaveLength(2)
    expect(restored[1]).toMatchObject({
      role: "assistant",
      requestId: "request-fixture",
      threadId: "thread-fixture",
    })
  })
})
