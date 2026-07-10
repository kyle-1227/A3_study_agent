import type { Message } from "@/components/chat-area"
import { mergeActivityTimeline } from "@/lib/activity-reducer"
import type { ActivityEvent } from "@/lib/observability-contracts"

export function attachActivityToAssistantMessage(
  messages: readonly Message[],
  event: ActivityEvent,
  activeAssistantId: string,
): Message[] {
  let matched = false
  const next = messages.map((message) => {
    const isRequestMatch =
      message.role === "assistant" && message.requestId === event.requestId
    const isActiveMatch =
      message.role === "assistant" && activeAssistantId && message.id === activeAssistantId
    if (!isRequestMatch && !isActiveMatch) return message
    matched = true
    return {
      ...message,
      requestId: event.requestId,
      threadId: event.threadId,
      activities: mergeActivityTimeline(message.activities ?? [], [event]),
    }
  })
  if (matched) return next
  return [
    ...next,
    {
      id: activityMessageId(event.requestId),
      role: "assistant",
      content: "",
      requestId: event.requestId,
      threadId: event.threadId,
      activities: [event],
    },
  ]
}

export function restoreActivitiesToMessages(
  messages: readonly Message[],
  timeline: readonly ActivityEvent[],
  threadId: string,
): Message[] {
  const byRequest = new Map<string, ActivityEvent[]>()
  for (const event of timeline) {
    if (event.threadId !== threadId) continue
    byRequest.set(
      event.requestId,
      mergeActivityTimeline(byRequest.get(event.requestId) ?? [], [event]),
    )
  }

  const assigned = new Set<string>()
  const restored = messages.map((message) => {
    if (message.role !== "assistant" || !message.requestId) return message
    const activities = byRequest.get(message.requestId)
    if (!activities) return message
    assigned.add(message.requestId)
    return { ...message, activities, threadId }
  })

  for (const [requestId, activities] of byRequest) {
    if (assigned.has(requestId)) continue
    restored.push({
      id: activityMessageId(requestId),
      role: "assistant",
      content: "",
      requestId,
      threadId,
      activities,
    })
  }
  return restored
}

function activityMessageId(requestId: string): string {
  const safeId = requestId.replace(/[^a-zA-Z0-9_-]+/g, "-").slice(0, 96)
  return `assistant-activity-${safeId}`
}
