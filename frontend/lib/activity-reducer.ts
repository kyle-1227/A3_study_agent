import type { ActivityEvent, ActivityStatus } from "@/lib/observability-contracts"

export const ACTIVITY_TIMELINE_ITEM_LIMIT = 200
export const ACTIVITY_TIMELINE_CHAR_LIMIT = 96_000

const STATUS_RANK: Record<ActivityStatus, number> = {
  queued: 0,
  running: 1,
  retrying: 2,
  waiting: 3,
  skipped: 4,
  completed: 5,
  interrupted: 6,
  failed: 7,
}

export function mergeActivityTimeline(
  existing: readonly ActivityEvent[],
  updates: readonly ActivityEvent[],
): ActivityEvent[] {
  const merged = new Map<string, ActivityEvent>()
  for (const candidate of [...existing, ...updates]) {
    const prior = merged.get(candidate.activityId)
    if (!prior) {
      merged.set(candidate.activityId, candidate)
      continue
    }
    if (compareActivity(candidate, prior) < 0) continue
    merged.set(candidate.activityId, {
      ...prior,
      ...candidate,
      sequence: Math.max(prior.sequence, candidate.sequence),
      startedAt: prior.startedAt < candidate.startedAt ? prior.startedAt : candidate.startedAt,
    })
  }

  const ordered = [...merged.values()]
    .sort((left, right) => left.sequence - right.sequence || left.activityId.localeCompare(right.activityId))
    .slice(-ACTIVITY_TIMELINE_ITEM_LIMIT)

  const bounded: ActivityEvent[] = []
  let totalChars = 2
  for (const event of [...ordered].reverse()) {
    const itemChars = JSON.stringify(event).length
    if (itemChars > ACTIVITY_TIMELINE_CHAR_LIMIT) continue
    if (bounded.length > 0 && totalChars + itemChars > ACTIVITY_TIMELINE_CHAR_LIMIT) continue
    bounded.push(event)
    totalChars += itemChars
  }
  return bounded.reverse()
}

export function activitiesForRequest(
  timeline: readonly ActivityEvent[],
  requestId: string,
): ActivityEvent[] {
  if (!requestId) return []
  return timeline.filter((event) => event.requestId === requestId)
}

export function latestActivityByNode(
  timeline: readonly ActivityEvent[],
): Map<string, ActivityEvent> {
  const result = new Map<string, ActivityEvent>()
  for (const event of timeline) {
    if (!event.node) continue
    const prior = result.get(event.node)
    if (!prior || compareActivity(event, prior) >= 0) result.set(event.node, event)
  }
  return result
}

function compareActivity(left: ActivityEvent, right: ActivityEvent): number {
  const sequenceOrder = left.sequence - right.sequence
  if (sequenceOrder !== 0) return sequenceOrder
  const timestampOrder = left.updatedAt.localeCompare(right.updatedAt)
  if (timestampOrder !== 0) return timestampOrder
  return STATUS_RANK[left.status] - STATUS_RANK[right.status]
}
