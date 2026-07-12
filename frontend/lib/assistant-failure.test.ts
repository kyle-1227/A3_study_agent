import { describe, expect, it } from "vitest"

import {
  SAFE_ASSISTANT_FAILURE_CONTENT,
  mergeSafeFailureContent,
} from "@/lib/assistant-failure"

describe("assistant failure content", () => {
  it("fills an empty assistant placeholder with a safe terminal message", () => {
    const result = mergeSafeFailureContent({ role: "assistant" as const, content: "" })

    expect(result.content).toBe(SAFE_ASSISTANT_FAILURE_CONTENT)
  })

  it("preserves an existing assistant response and user content", () => {
    expect(
      mergeSafeFailureContent({ role: "assistant" as const, content: "Partial answer" }),
    ).toEqual({ role: "assistant", content: "Partial answer" })
    expect(mergeSafeFailureContent({ role: "user" as const, content: "Question" })).toEqual({
      role: "user",
      content: "Question",
    })
  })
})
