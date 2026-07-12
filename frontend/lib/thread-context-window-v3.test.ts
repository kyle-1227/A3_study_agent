import { describe, expect, it } from "vitest"

import { parseThreadContextWindowV3 } from "@/lib/thread-context-window-v3"

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
