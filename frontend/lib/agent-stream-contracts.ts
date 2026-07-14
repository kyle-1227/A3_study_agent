export const AGENT_STREAM_SCHEMA_VERSION = "agent_stream_v2" as const

export const AGENT_STREAM_EVENT_TYPES = [
  "stream_start",
  "content_block_start",
  "content_block_delta",
  "content_block_stop",
  "activity_update",
  "tool_progress",
  "artifact_progress",
  "qa_final",
  "resource_final",
  "assessment_final",
  "interrupt",
  "stopped",
  "stream_error",
  "stream_done",
] as const

export type AgentStreamEventType = (typeof AGENT_STREAM_EVENT_TYPES)[number]

export interface AgentStreamEventV2 {
  schemaVersion: typeof AGENT_STREAM_SCHEMA_VERSION
  type: AgentStreamEventType
  streamId: string
  eventId: string
  sequence: number
  requestId: string
  threadId: string
  createdAt: string
  data: Record<string, unknown>
}

const EVENT_TYPES = new Set<string>(AGENT_STREAM_EVENT_TYPES)

export class AgentStreamContractError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "AgentStreamContractError"
  }
}

export function parseAgentStreamEvent(value: unknown): AgentStreamEventV2 {
  const data = record(value, "agent_stream_v2")
  exactKeys(data, [
    "schema_version",
    "type",
    "stream_id",
    "event_id",
    "sequence",
    "request_id",
    "thread_id",
    "created_at",
    "data",
  ])
  if (data.schema_version !== AGENT_STREAM_SCHEMA_VERSION) {
    fail("schema_version must equal agent_stream_v2")
  }
  if (typeof data.type !== "string" || !EVENT_TYPES.has(data.type)) {
    fail("type is not an agent_stream_v2 event")
  }
  const streamId = boundedString(data.stream_id, "stream_id", 160)
  const sequence = positiveInteger(data.sequence, "sequence")
  const eventId = boundedString(data.event_id, "event_id", 220)
  if (eventId !== `${streamId}:${sequence}`) {
    fail("event_id must equal '<stream_id>:<sequence>'")
  }
  const createdAt = boundedString(data.created_at, "created_at", 80)
  if (!Number.isFinite(Date.parse(createdAt)) || !/(?:Z|[+-]\d\d:\d\d)$/.test(createdAt)) {
    fail("created_at must be timezone-aware")
  }
  return {
    schemaVersion: AGENT_STREAM_SCHEMA_VERSION,
    type: data.type as AgentStreamEventType,
    streamId,
    eventId,
    sequence,
    requestId: boundedString(data.request_id, "request_id", 160),
    threadId: boundedString(data.thread_id, "thread_id", 160),
    createdAt,
    data: record(data.data, "data"),
  }
}

function record(value: unknown, field: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(`${field} must be an object`)
  }
  return value as Record<string, unknown>
}

function exactKeys(data: Record<string, unknown>, expected: string[]): void {
  const allowed = new Set(expected)
  const extra = Object.keys(data).find((key) => !allowed.has(key))
  if (extra) fail(`unexpected field: ${extra}`)
  const missing = expected.find((key) => !(key in data))
  if (missing) fail(`missing field: ${missing}`)
}

function boundedString(value: unknown, field: string, maxLength: number): string {
  if (typeof value !== "string" || value.length === 0) fail(`${field} is required`)
  if (value.length > maxLength) fail(`${field} exceeds ${maxLength} characters`)
  return value
}

function positiveInteger(value: unknown, field: string): number {
  if (!Number.isInteger(value) || (value as number) < 1) {
    fail(`${field} must be a positive integer`)
  }
  return value as number
}

function fail(message: string): never {
  throw new AgentStreamContractError(message)
}
