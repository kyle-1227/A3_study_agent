import { readFileSync } from "node:fs"

import { describe, expect, it } from "vitest"

const chatAreaSource = readFileSync(new URL("./chat-area.tsx", import.meta.url), "utf8")
const rightPanelSource = readFileSync(new URL("./right-panel.tsx", import.meta.url), "utf8")

describe("thread context window placement", () => {
  it("keeps the context capsule in the composer and out of the right panel", () => {
    expect(chatAreaSource).toContain("<ThreadContextCapsule")
    expect(chatAreaSource).not.toMatch(/\bMic\b/)
    expect(rightPanelSource).not.toContain("ContextUsagePanel")
  })
})
