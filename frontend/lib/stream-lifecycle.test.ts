import { describe, expect, it } from "vitest"

import {
  beginStreamLifecycle,
  reduceStreamLifecycle,
  streamPhaseIsTerminal,
} from "@/lib/stream-lifecycle"

describe("stream lifecycle", () => {
  it("does not finalize on run_status completed", () => {
    const state = reduceStreamLifecycle(beginStreamLifecycle(), {
      type: "run_status",
      run_status: "completed",
    })
    expect(state).toEqual({ phase: "running", terminalEvent: "" })
    expect(streamPhaseIsTerminal(state)).toBe(false)
  })

  it.each([
    ["done", "completed"],
    ["error", "failed"],
    ["interrupt", "waiting"],
  ] as const)("finalizes %s as %s", (eventType, phase) => {
    const state = reduceStreamLifecycle(beginStreamLifecycle(), { type: eventType })
    expect(state.phase).toBe(phase)
    expect(state.terminalEvent).toBe(eventType)
    expect(streamPhaseIsTerminal(state)).toBe(true)
  })
})
