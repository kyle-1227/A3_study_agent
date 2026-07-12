import { describe, expect, it } from "vitest"

import {
  parseBackgroundContextWindow,
  parseThreadContextWindowV2,
} from "@/lib/observability-contracts"
import {
  backgroundContextPayload,
  threadContextWindowV2Payload,
} from "@/test/observability-fixtures"

describe("ThreadContextWindowV2 contract", () => {
  it("keeps next-call estimate, last call, and inventory separate", () => {
    const window = parseThreadContextWindowV2(threadContextWindowV2Payload())

    expect(window.nextCallContextEstimate.usedRatio).toBe(0.1)
    expect(window.lastLlmCallUsage.usedRatio).toBe(0.12)
    expect(window.backgroundInventory.workspaceArtifactCount).toBe(2)
    expect(window.nextCallContextEstimate.unknownSections).toContain("ce_block")
  })

  it("rejects invalid estimate fingerprints and confidence", () => {
    const invalidFingerprint = threadContextWindowV2Payload()
    ;(invalidFingerprint.next_call_context_estimate as Record<string, unknown>)[
      "state_fingerprint"
    ] = "tokens-from-a-hash"
    expect(() => parseThreadContextWindowV2(invalidFingerprint)).toThrow(
      /state_fingerprint/,
    )

    const invalidConfidence = threadContextWindowV2Payload()
    ;(invalidConfidence.next_call_context_estimate as Record<string, unknown>)[
      "confidence"
    ] = "certain"
    expect(() => parseThreadContextWindowV2(invalidConfidence)).toThrow(
      /confidence/,
    )
  })

  it("preserves the legacy v1 parser as an additive compatibility surface", () => {
    const legacy = parseBackgroundContextWindow(backgroundContextPayload())
    const current = parseThreadContextWindowV2(threadContextWindowV2Payload())

    expect(legacy.schemaVersion).toBe(1)
    expect(current.schemaVersion).toBe(2)
  })
})
