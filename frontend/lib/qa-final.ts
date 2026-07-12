import { ContractParseError } from "@/lib/observability-contracts"

export type QAScope = "academic" | "general" | "a3_agent"
export type QAGroundingStatus =
  | "judged_evidence"
  | "general_knowledge"
  | "capability_registry"
  | "insufficient_evidence"
  | "not_live_verified"

export interface QASuggestion {
  label: string
  action: string
  resourceType: string
}

export interface QAFinalEventV1 {
  type: "qa_final"
  schemaVersion: 1
  qaId: string
  payloadHash: string
  qaScope: QAScope
  response: {
    answer: string
    uncertaintyNote: string
    groundingStatus: QAGroundingStatus
    suggestions: QASuggestion[]
  }
  threadId: string
  requestId: string
  createdAt: string
}

export interface QAFinalMessage {
  id: string
  role: "user" | "assistant"
  content: string
  requestId?: string
  threadId?: string
  qaFinal?: QAFinalEventV1
  qaFinalDedupeKey?: string
}

export interface QAFinalAttachResult<T extends QAFinalMessage> {
  messages: T[]
  messageId: string
  dedupeKey: string
  attached: boolean
}

const QA_ID_PATTERN = /^qa:v1:[0-9a-f]{64}$/
const SHA256_PATTERN = /^[0-9a-f]{64}$/
const QA_SCOPES = new Set<QAScope>(["academic", "general", "a3_agent"])
const GROUNDING_STATUSES = new Set<QAGroundingStatus>([
  "judged_evidence",
  "general_knowledge",
  "capability_registry",
  "insufficient_evidence",
  "not_live_verified",
])

export function parseQAFinalEvent(value: unknown): QAFinalEventV1 {
  const contract = "qa_final_v1"
  const data = record(value, contract)
  exactKeys(
    data,
    [
      "type",
      "schema_version",
      "qa_id",
      "payload_hash",
      "qa_scope",
      "response",
      "thread_id",
      "request_id",
      "created_at",
    ],
    contract,
  )
  if (data.type !== "qa_final") fail(contract, "type must equal qa_final")
  if (data.schema_version !== 1) fail(contract, "schema_version must equal 1")
  const qaId = boundedString(data.qa_id, "qa_id", contract, 70)
  const payloadHash = boundedString(data.payload_hash, "payload_hash", contract, 64)
  if (!QA_ID_PATTERN.test(qaId)) fail(contract, "qa_id is invalid")
  if (!SHA256_PATTERN.test(payloadHash)) fail(contract, "payload_hash is invalid")
  const qaScope = boundedString(data.qa_scope, "qa_scope", contract, 40) as QAScope
  if (!QA_SCOPES.has(qaScope)) fail(contract, "qa_scope is invalid")

  const response = record(data.response, `${contract}.response`)
  exactKeys(
    response,
    ["answer", "uncertainty_note", "grounding_status", "suggestions"],
    `${contract}.response`,
  )
  const answer = boundedString(response.answer, "answer", contract, 6000)
  const uncertaintyNote = optionalBoundedString(
    response.uncertainty_note,
    "uncertainty_note",
    contract,
    1000,
  )
  const groundingStatus = boundedString(
    response.grounding_status,
    "grounding_status",
    contract,
    40,
  ) as QAGroundingStatus
  if (!GROUNDING_STATUSES.has(groundingStatus)) {
    fail(contract, "grounding_status is invalid")
  }
  if (!Array.isArray(response.suggestions) || response.suggestions.length > 3) {
    fail(contract, "suggestions must contain at most 3 items")
  }
  const suggestions = response.suggestions.map((item, index) =>
    parseSuggestion(item, `${contract}.suggestions.${index}`),
  )

  const createdAt = boundedString(data.created_at, "created_at", contract, 80)
  if (!isUtcTimestamp(createdAt)) fail(contract, "created_at must be timezone-aware UTC")

  return {
    type: "qa_final",
    schemaVersion: 1,
    qaId,
    payloadHash,
    qaScope,
    response: { answer, uncertaintyNote, groundingStatus, suggestions },
    threadId: boundedString(data.thread_id, "thread_id", contract, 120),
    requestId: boundedString(data.request_id, "request_id", contract, 120),
    createdAt,
  }
}

export function qaFinalDedupeKey(event: QAFinalEventV1): string {
  return `qa_id:${event.qaId}`
}

export function qaMessageId(event: QAFinalEventV1): string {
  return `assistant-qa-${event.qaId.replace(/[^a-zA-Z0-9_-]+/g, "-")}`
}

export function mergeQAFinalIntoMessage<T extends QAFinalMessage>(
  message: T,
  event: QAFinalEventV1,
): T {
  if (message.role !== "assistant") {
    throw new ContractParseError("qa_final_binding", "target message must be assistant")
  }
  if (message.threadId && message.threadId !== event.threadId) {
    throw new ContractParseError("qa_final_binding", "thread_id does not match target message")
  }
  if (message.requestId && message.requestId !== event.requestId) {
    throw new ContractParseError("qa_final_binding", "request_id does not match target message")
  }
  return {
    ...message,
    content: event.response.answer,
    requestId: event.requestId,
    threadId: event.threadId,
    qaFinal: event,
    qaFinalDedupeKey: qaFinalDedupeKey(event),
  }
}

export function attachQAFinalToMessages<T extends QAFinalMessage>(
  messages: T[],
  event: QAFinalEventV1,
  preferredMessageId = "",
): QAFinalAttachResult<T> {
  const dedupeKey = qaFinalDedupeKey(event)
  const duplicate = messages.find((message) => message.qaFinalDedupeKey === dedupeKey)
  if (duplicate) {
    return { messages, messageId: duplicate.id, dedupeKey, attached: false }
  }

  const requestTarget = messages.find(
    (message) =>
      message.role === "assistant" &&
      message.requestId === event.requestId &&
      message.threadId === event.threadId,
  )
  const preferredTarget = messages.find(
    (message) =>
      message.id === preferredMessageId &&
      message.role === "assistant" &&
      (!message.requestId || message.requestId === event.requestId) &&
      (!message.threadId || message.threadId === event.threadId),
  )
  const target = requestTarget ?? preferredTarget
  if (target?.qaFinal && target.qaFinalDedupeKey !== dedupeKey) {
    throw new ContractParseError("qa_final_binding", "request already has a different QA final")
  }

  const messageId = target?.id ?? qaMessageId(event)
  const base = target ?? ({
    id: messageId,
    role: "assistant",
    content: "",
    requestId: event.requestId,
    threadId: event.threadId,
  } as T)
  const merged = mergeQAFinalIntoMessage(base, event)
  if (!target) {
    return { messages: [...messages, merged], messageId, dedupeKey, attached: true }
  }
  return {
    messages: messages.map((message) => (message.id === target.id ? merged : message)),
    messageId,
    dedupeKey,
    attached: true,
  }
}

function parseSuggestion(value: unknown, contract: string): QASuggestion {
  const data = record(value, contract)
  exactKeys(data, ["label", "action", "resource_type"], contract)
  return {
    label: boundedString(data.label, "label", contract, 160),
    action: boundedString(data.action, "action", contract, 80),
    resourceType: optionalBoundedString(data.resource_type, "resource_type", contract, 80),
  }
}

function record(value: unknown, contract: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(contract, "expected an object")
  }
  return value as Record<string, unknown>
}

function exactKeys(data: Record<string, unknown>, keys: string[], contract: string): void {
  const allowed = new Set(keys)
  const extras = Object.keys(data).filter((key) => !allowed.has(key))
  if (extras.length > 0) fail(contract, `unexpected field: ${extras.sort()[0]}`)
}

function boundedString(
  value: unknown,
  field: string,
  contract: string,
  maxLength: number,
): string {
  if (typeof value !== "string" || !value.trim()) fail(contract, `${field} is required`)
  if (value.length > maxLength) fail(contract, `${field} exceeds ${maxLength} characters`)
  return value
}

function optionalBoundedString(
  value: unknown,
  field: string,
  contract: string,
  maxLength: number,
): string {
  if (value === undefined || value === null || value === "") return ""
  if (typeof value !== "string") fail(contract, `${field} must be a string`)
  if (value.length > maxLength) fail(contract, `${field} exceeds ${maxLength} characters`)
  return value
}

function isUtcTimestamp(value: string): boolean {
  if (!/(?:Z|[+-]00:00)$/.test(value)) return false
  return Number.isFinite(Date.parse(value))
}

function fail(contract: string, reason: string): never {
  throw new ContractParseError(contract, reason)
}
