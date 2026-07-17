import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"

import { describe, expect, it } from "vitest"

import {
  finishThreadContextWindowV3Update,
  markThreadContextWindowV3Updating,
  parseThreadContextWindowV3,
  threadContextWindowV3ForSelection,
} from "@/lib/thread-context-window-v3"

function payload() {
  const stats = {
    retained_tokens: 0,
    lifetime_injected_tokens: 0,
    lifetime_unique_tokens: 0,
    injection_count: 0,
    repeat_injection_count: 0,
    active_item_count: 0,
  }
  return {
    schema_version: 3,
    thread_id: "thread-1",
    updated_at: "2026-07-13T00:00:00Z",
    updating: false,
    window_model: "deepseek-v4-pro",
    context_window_limit_tokens: 1_000_000,
    retained_memory_tokens: 10_000,
    retained_ratio: 0.01,
    lifetime_injected_tokens: 20_000,
    lifetime_unique_tokens: 10_000,
    request_count: 2,
    injection_count: 2,
    repeat_injection_count: 1,
    injection_types: {
      profile: stats,
      memory: {
        retained_tokens: 10_000,
        lifetime_injected_tokens: 20_000,
        lifetime_unique_tokens: 10_000,
        injection_count: 2,
        repeat_injection_count: 1,
        active_item_count: 1,
      },
      evidence: stats,
      artifact: stats,
      rules: stats,
      curriculum: stats,
      trajectory: stats,
      pipeline: stats,
    },
    measurement: {
      last_tokenizer_mode: "estimated_mixed_v1",
      last_estimated: true,
      estimated_injection_count: 2,
    },
    memory_summary: { active_item_count: 1, active_unique_content_count: 1 },
    compaction: {
      status: "never",
      boundary_id: "",
      compacted_at: null,
      before_tokens: 0,
      after_tokens: 0,
    },
  }
}

describe("parseThreadContextWindowV3", () => {
  it("parses retained and lifetime accounting independently", () => {
    const parsed = parseThreadContextWindowV3(payload())
    expect(parsed.retainedRatio).toBe(0.01)
    expect(parsed.lifetimeInjectedTokens).toBe(20_000)
    expect(parsed.repeatInjectionCount).toBe(1)
  })

  it("rejects prediction fields and inconsistent totals", () => {
    expect(() => parseThreadContextWindowV3({ ...payload(), headroom: 4 })).toThrow("unexpected field")
    expect(() => parseThreadContextWindowV3({ ...payload(), retained_ratio: 0.5 })).toThrow(
      "retained_ratio",
    )
  })
})

describe("Thread Context Window V3 page state", () => {
  it("preserves the last valid snapshot when a pre-stream failure ends an update", () => {
    const current = parseThreadContextWindowV3(payload())

    const updating = markThreadContextWindowV3Updating(current)
    const restored = finishThreadContextWindowV3Update(updating)

    expect(updating).toEqual({ ...current, updating: true })
    expect(restored).toEqual(current)
    expect(restored?.retainedMemoryTokens).toBe(10_000)
    expect(restored?.requestCount).toBe(2)
    expect(finishThreadContextWindowV3Update(current)).toBe(current)
  })

  it("retains only a matching snapshot when the active thread is selected again", () => {
    const current = parseThreadContextWindowV3(payload())

    expect(threadContextWindowV3ForSelection(current, "thread-1", "thread-1")).toBe(current)
    expect(threadContextWindowV3ForSelection(current, "thread-1", "thread-2")).toBeNull()
    expect(
      threadContextWindowV3ForSelection(
        { ...current, threadId: "thread-2" },
        "thread-1",
        "thread-1",
      ),
    ).toBeNull()
  })

  it("wires pre-stream cleanup and same-thread selection through the page", () => {
    const page = readFileSync(
      fileURLToPath(new URL("../app/page.tsx", import.meta.url)),
      "utf8",
    )
    const fetchStart = page.indexOf("const fetchWithErrorHandling")
    const fetchEnd = page.indexOf("const refreshThreadStatus", fetchStart)
    const fetchSource = page.slice(fetchStart, fetchEnd)
    const selectionStart = page.indexOf("const handleSelectChat")
    const selectionEnd = page.indexOf("const handleClearChatHistory", selectionStart)
    const selectionSource = page.slice(selectionStart, selectionEnd)

    expect(page).toContain("setThreadContextWindowV3(markThreadContextWindowV3Updating)")
    expect(fetchSource).toContain("let streamEstablished = false")
    expect(fetchSource).toContain("finally")
    expect(fetchSource).toContain(
      "setThreadContextWindowV3(finishThreadContextWindowV3Update)",
    )
    expect(selectionSource).toContain("threadContextWindowV3ForSelection")
    expect(selectionSource).not.toContain("setThreadContextWindowV3(null)")
  })
})
