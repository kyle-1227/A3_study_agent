import { describe, expect, it } from "vitest"

import {
  ACTIVITY_TIMELINE_ITEM_LIMIT,
  mergeActivityTimeline,
} from "@/lib/activity-reducer"
import { parseActivityEvent } from "@/lib/observability-contracts"
import { activityPayload, LATER, NOW } from "@/test/observability-fixtures"

describe("activity timeline reducer", () => {
  it("is idempotent and keeps the newest stable-id update", () => {
    const running = parseActivityEvent(activityPayload())
    const completed = parseActivityEvent(
      activityPayload({
        status: "completed",
        updated_at: LATER,
        completed_at: LATER,
        duration_ms: 1000,
      }),
    )
    const once = mergeActivityTimeline([], [running, completed])
    const twice = mergeActivityTimeline(once, [running, completed])
    expect(twice).toEqual(once)
    expect(twice).toHaveLength(1)
    expect(twice[0].status).toBe("completed")
    expect(twice[0].startedAt).toBe(NOW)
  })

  it("bounds a large timeline deterministically", () => {
    const events = Array.from({ length: ACTIVITY_TIMELINE_ITEM_LIMIT + 25 }, (_, index) =>
      parseActivityEvent(
        activityPayload({
          activity_id: `activity:v1:${String(index).padStart(4, "0")}`,
          sequence: index + 1,
        }),
      ),
    )
    const forward = mergeActivityTimeline([], events)
    const reversed = mergeActivityTimeline([], [...events].reverse())
    expect(forward).toHaveLength(ACTIVITY_TIMELINE_ITEM_LIMIT)
    expect(reversed).toEqual(forward)
    expect(forward[0].sequence).toBe(26)
  })
})
