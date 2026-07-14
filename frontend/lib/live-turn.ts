import type { AgentStreamEventV2 } from "@/lib/agent-stream-contracts"
import {
  abortEvidenceProgress,
  emptyEvidenceProgressTimeline,
  parseEvidenceProgressEvent,
  reduceEvidenceProgress,
  type EvidenceProgressTimeline,
} from "@/lib/evidence-progress"

export type LiveTurnLifecycle = "running" | "waiting" | "completed" | "failed"

export interface LiveContentBlock {
  id: string
  index: number
  type: "markdown" | "text" | "tool"
  provisional: boolean
  content: string
  stopped: boolean
}

export interface LiveTurnState {
  streamId: string
  requestId: string
  threadId: string
  lifecycle: LiveTurnLifecycle
  blockOrder: string[]
  blocks: Record<string, LiveContentBlock>
  activities: Record<string, unknown>[]
  tools: Record<string, unknown>[]
  evidenceProgress: EvidenceProgressTimeline
  provisionalAnswer: string
  lastSequence: number
  committed: boolean
  error: string
}

export class LiveTurnSequenceError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "LiveTurnSequenceError"
  }
}

export function reduceLiveTurn(
  state: LiveTurnState | null,
  event: AgentStreamEventV2,
): LiveTurnState {
  if (state === null) {
    if (event.type !== "stream_start" || event.sequence !== 1) {
      throw new LiveTurnSequenceError("first event must be stream_start sequence 1")
    }
    return emptyLiveTurn(event)
  }
  if (
    event.streamId !== state.streamId ||
    event.requestId !== state.requestId ||
    event.threadId !== state.threadId
  ) {
    throw new LiveTurnSequenceError("stream identity changed within one live turn")
  }
  if (event.sequence <= state.lastSequence) return state
  if (event.sequence !== state.lastSequence + 1) {
    throw new LiveTurnSequenceError(
      `sequence gap: expected ${state.lastSequence + 1}, received ${event.sequence}`,
    )
  }

  let next: LiveTurnState = { ...state, lastSequence: event.sequence }
  if (event.type === "content_block_start") {
    const block = parseBlock(event, "")
    if (next.blocks[block.id]) throw new LiveTurnSequenceError("block already exists")
    next = {
      ...next,
      blockOrder: [...next.blockOrder, block.id],
      blocks: { ...next.blocks, [block.id]: block },
    }
  } else if (event.type === "content_block_delta") {
    const id = requiredString(event.data.block_id, "block_id")
    const current = next.blocks[id]
    if (!current || current.stopped) {
      throw new LiveTurnSequenceError("delta requires an open content block")
    }
    const delta = stringValue(event.data.delta)
    const updated = { ...current, content: current.content + delta }
    next = {
      ...next,
      blocks: { ...next.blocks, [id]: updated },
      provisionalAnswer:
        updated.type === "markdown"
          ? next.blockOrder
              .map((blockId) => (blockId === id ? updated : next.blocks[blockId]))
              .filter((block) => block?.type === "markdown")
              .map((block) => block.content)
              .join("")
          : next.provisionalAnswer,
    }
  } else if (event.type === "content_block_stop") {
    const id = requiredString(event.data.block_id, "block_id")
    const current = next.blocks[id]
    if (!current) throw new LiveTurnSequenceError("stop requires an existing block")
    const reset = event.data.reset === true
    next = {
      ...next,
      blocks: {
        ...next.blocks,
        [id]: { ...current, content: reset ? "" : current.content, stopped: true },
      },
      provisionalAnswer: reset ? "" : next.provisionalAnswer,
    }
  } else if (event.type === "activity_update") {
    next = { ...next, activities: [...next.activities, event.data] }
  } else if (event.type === "evidence_progress") {
    next = {
      ...next,
      evidenceProgress: reduceEvidenceProgress(
        next.evidenceProgress,
        parseEvidenceProgressEvent(event.data),
      ),
    }
  } else if (event.type === "tool_progress") {
    next = { ...next, tools: [...next.tools, event.data] }
  } else if (
    event.type === "qa_final" ||
    event.type === "resource_final" ||
    event.type === "assessment_final"
  ) {
    next = { ...next, committed: true }
  } else if (event.type === "interrupt" || event.type === "stopped") {
    next = {
      ...next,
      lifecycle: "waiting",
      evidenceProgress: abortEvidenceProgress(next.evidenceProgress),
    }
  } else if (event.type === "stream_error") {
    next = {
      ...next,
      lifecycle: "failed",
      error: stringValue(event.data.message) || "stream_error",
      provisionalAnswer: "",
      evidenceProgress: abortEvidenceProgress(next.evidenceProgress),
    }
  } else if (event.type === "stream_done") {
    if (
      next.evidenceProgress.order.length > 0 &&
      !next.evidenceProgress.terminal &&
      !next.evidenceProgress.aborted
    ) {
      throw new LiveTurnSequenceError(
        "stream_done received before evidence progress terminated",
      )
    }
    next = { ...next, lifecycle: next.lifecycle === "running" ? "completed" : next.lifecycle }
  }
  return next
}

function emptyLiveTurn(event: AgentStreamEventV2): LiveTurnState {
  return {
    streamId: event.streamId,
    requestId: event.requestId,
    threadId: event.threadId,
    lifecycle: "running",
    blockOrder: [],
    blocks: {},
    activities: [],
    tools: [],
    evidenceProgress: emptyEvidenceProgressTimeline(),
    provisionalAnswer: "",
    lastSequence: event.sequence,
    committed: false,
    error: "",
  }
}

function parseBlock(event: AgentStreamEventV2, content: string): LiveContentBlock {
  const type = requiredString(event.data.block_type, "block_type")
  if (type !== "markdown" && type !== "text" && type !== "tool") {
    throw new LiveTurnSequenceError("block_type is invalid")
  }
  const index = event.data.block_index
  if (!Number.isInteger(index) || (index as number) < 0) {
    throw new LiveTurnSequenceError("block_index must be non-negative")
  }
  return {
    id: requiredString(event.data.block_id, "block_id"),
    index: index as number,
    type,
    provisional: event.data.provisional === true,
    content,
    stopped: false,
  }
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new LiveTurnSequenceError(`${field} is required`)
  }
  return value
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : ""
}
