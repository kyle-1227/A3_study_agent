import { ContractParseError } from "@/lib/observability-contracts"

export type RecommendationResourceType =
  | "review_doc"
  | "mindmap"
  | "quiz"
  | "code_practice"
  | "video_script"
  | "video_animation"
  | "study_plan"

export type RecommendationFinalStatus = "available" | "unavailable"

export type ExplicitRecommendationUnavailableReason =
  | "missing_user_id"
  | "missing_subject"
  | "profile_unavailable"
  | "history_unavailable"
  | "no_eligible_candidates"
  | "unsupported_subject_scope"

export interface RecommendationFinalCandidateV1 {
  resource_id: string
  resource_type: RecommendationResourceType
  subject: string
  topic_id: string
  title: string
}

export interface RecommendationCandidateSnapshotV1 {
  schema_version: "recommendation_candidate_snapshot_v1"
  source_schema_version: "knowledge_graph_v1"
  source_data_version: string
  source_fingerprint: string
  subject: string
  candidate_count: number
  inventory_hash: string
  targets: RecommendationFinalCandidateV1[]
  snapshot_id: string
}

export interface RecommendationFinalItemV1 {
  recommendation_id: string
  resource_id: string
  resource_type: RecommendationResourceType
  topic_id: string
  title: string
  rank: number
  score: number
  reason: string
}

export interface RecommendationFinalV1 {
  schema_version: "recommendation_final_v1"
  type: "recommendation_final"
  thread_id: string
  request_id: string
  terminal_status: RecommendationFinalStatus
  mode: "explicit_request"
  user_id: string | null
  subject: string | null
  learning_guidance_runtime_fingerprint: string
  generated_at: string | null
  recommendations: RecommendationFinalItemV1[]
  candidate_snapshot: RecommendationCandidateSnapshotV1 | null
  unavailable_reason: ExplicitRecommendationUnavailableReason | null
  summary: string
  recommendation_final_id: string
  payload_hash: string
}

export interface RecommendationFinalMessage {
  id: string
  role: "user" | "assistant"
  content: string
  requestId?: string
  threadId?: string
  recommendationFinal?: RecommendationFinalV1
  recommendationFinalDedupeKey?: string
  qaFinal?: unknown
  resourceFinalPayload?: unknown
}

export interface RecommendationFinalAttachResult<T extends RecommendationFinalMessage> {
  messages: T[]
  messageId: string
  dedupeKey: string
  attached: boolean
}

type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue }

const RESOURCE_TYPES = new Set<RecommendationResourceType>([
  "review_doc",
  "mindmap",
  "quiz",
  "code_practice",
  "video_script",
  "video_animation",
  "study_plan",
])
const UNAVAILABLE_REASONS = new Set<ExplicitRecommendationUnavailableReason>([
  "missing_user_id",
  "missing_subject",
  "profile_unavailable",
  "history_unavailable",
  "no_eligible_candidates",
  "unsupported_subject_scope",
])
const TOP_LEVEL_FIELDS = [
  "schema_version",
  "type",
  "thread_id",
  "request_id",
  "terminal_status",
  "mode",
  "user_id",
  "subject",
  "learning_guidance_runtime_fingerprint",
  "generated_at",
  "recommendations",
  "candidate_snapshot",
  "unavailable_reason",
  "summary",
  "recommendation_final_id",
  "payload_hash",
] as const
const SNAPSHOT_FIELDS = [
  "schema_version",
  "source_schema_version",
  "source_data_version",
  "source_fingerprint",
  "subject",
  "candidate_count",
  "inventory_hash",
  "targets",
  "snapshot_id",
] as const
const CANDIDATE_FIELDS = [
  "resource_id",
  "resource_type",
  "subject",
  "topic_id",
  "title",
] as const
const RECOMMENDATION_FIELDS = [
  "recommendation_id",
  "resource_id",
  "resource_type",
  "topic_id",
  "title",
  "rank",
  "score",
  "reason",
] as const

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const SHA256_PATTERN = /^[0-9a-f]{64}$/
const PAYLOAD_HASH_PATTERN = /^recommendation-final-payload:v1:[0-9a-f]{64}$/
const FINAL_ID_PATTERN = /^recommendation-final:v1:[0-9a-f]{64}$/
const SNAPSHOT_ID_PATTERN = /^recommendation-candidates:v1:[0-9a-f]{64}$/
const INVENTORY_HASH_PATTERN = /^recommendation-inventory:v1:[0-9a-f]{64}$/
const CANONICAL_ISO_PATTERN =
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{6})?[+-]\d{2}:\d{2}$/
const PAYLOAD_HASH_PREFIX = "recommendation-final-payload:v1"
const FINAL_ID_PREFIX = "recommendation-final:v1"
const SNAPSHOT_ID_PREFIX = "recommendation-candidates:v1"

export function parseRecommendationFinalV1(value: unknown): RecommendationFinalV1 {
  const contract = "recommendation_final_v1"
  const data = record(value, contract)
  exactKeys(data, TOP_LEVEL_FIELDS, contract)
  if (data.schema_version !== "recommendation_final_v1") {
    fail(contract, "schema_version must equal recommendation_final_v1")
  }
  if (data.type !== "recommendation_final") {
    fail(contract, "type must equal recommendation_final")
  }
  if (data.mode !== "explicit_request") {
    fail(contract, "mode must equal explicit_request")
  }

  const threadId = normalizedString(data.thread_id, "thread_id", contract, 160)
  const requestId = normalizedString(data.request_id, "request_id", contract, 160)
  if (!UUID_PATTERN.test(requestId)) fail(contract, "request_id must be a canonical UUID")

  let terminalStatus: RecommendationFinalStatus
  if (data.terminal_status === "available" || data.terminal_status === "unavailable") {
    terminalStatus = data.terminal_status
  } else {
    fail(contract, "terminal_status is invalid")
  }

  const generatedAt = nullableNormalizedString(
    data.generated_at,
    "generated_at",
    contract,
    64,
  )
  if (
    generatedAt !== null &&
    !isCanonicalTimezoneAwareIso(generatedAt)
  ) {
    fail(contract, "generated_at must use canonical timezone-aware ISO 8601")
  }

  const recommendations = arrayValue(
    data.recommendations,
    "recommendations",
    contract,
    50,
  ).map((item, index) =>
    parseRecommendationItem(item, `${contract}.recommendations.${index}`),
  )
  const candidateSnapshot =
    data.candidate_snapshot === null
      ? null
      : parseCandidateSnapshot(data.candidate_snapshot, `${contract}.candidate_snapshot`)
  const unavailableReason = parseUnavailableReason(
    data.unavailable_reason,
    `${contract}.unavailable_reason`,
  )
  const runtimeFingerprint = normalizedString(
    data.learning_guidance_runtime_fingerprint,
    "learning_guidance_runtime_fingerprint",
    contract,
    64,
  )
  if (!SHA256_PATTERN.test(runtimeFingerprint)) {
    fail(contract, "learning_guidance_runtime_fingerprint is invalid")
  }
  const recommendationFinalId = normalizedString(
    data.recommendation_final_id,
    "recommendation_final_id",
    contract,
    96,
  )
  if (!FINAL_ID_PATTERN.test(recommendationFinalId)) {
    fail(contract, "recommendation_final_id is invalid")
  }
  const payloadHash = normalizedString(data.payload_hash, "payload_hash", contract, 104)
  if (!PAYLOAD_HASH_PATTERN.test(payloadHash)) fail(contract, "payload_hash is invalid")

  const event: RecommendationFinalV1 = {
    schema_version: "recommendation_final_v1",
    type: "recommendation_final",
    thread_id: threadId,
    request_id: requestId,
    terminal_status: terminalStatus,
    mode: "explicit_request",
    user_id: nullableNormalizedString(data.user_id, "user_id", contract, 160),
    subject: nullableNormalizedString(data.subject, "subject", contract, 120),
    learning_guidance_runtime_fingerprint: runtimeFingerprint,
    generated_at: generatedAt,
    recommendations,
    candidate_snapshot: candidateSnapshot,
    unavailable_reason: unavailableReason,
    summary: normalizedString(data.summary, "summary", contract, 2_000),
    recommendation_final_id: recommendationFinalId,
    payload_hash: payloadHash,
  }
  validateTerminalTruth(event, contract)
  validateDerivedIdentity(event, contract)
  return event
}

export function recommendationFinalDedupeKey(event: RecommendationFinalV1): string {
  return `recommendation_final_id:${event.recommendation_final_id}`
}

export function recommendationMessageId(event: RecommendationFinalV1): string {
  return `assistant-recommendation-${event.recommendation_final_id.replace(
    /[^a-zA-Z0-9_-]+/g,
    "-",
  )}`
}

export function mergeRecommendationFinalIntoMessage<T extends RecommendationFinalMessage>(
  message: T,
  event: RecommendationFinalV1,
): T {
  if (message.role !== "assistant") {
    throw new ContractParseError(
      "recommendation_final_binding",
      "target message must be assistant",
    )
  }
  if (message.threadId && message.threadId !== event.thread_id) {
    throw new ContractParseError(
      "recommendation_final_binding",
      "thread_id does not match target message",
    )
  }
  if (message.requestId && message.requestId !== event.request_id) {
    throw new ContractParseError(
      "recommendation_final_binding",
      "request_id does not match target message",
    )
  }
  if (message.qaFinal || message.resourceFinalPayload) {
    throw new ContractParseError(
      "recommendation_final_binding",
      "request already has a different authoritative final",
    )
  }
  return {
    ...message,
    content: "",
    requestId: event.request_id,
    threadId: event.thread_id,
    recommendationFinal: event,
    recommendationFinalDedupeKey: recommendationFinalDedupeKey(event),
  }
}

export function attachRecommendationFinalToMessages<T extends RecommendationFinalMessage>(
  messages: T[],
  event: RecommendationFinalV1,
  preferredMessageId = "",
): RecommendationFinalAttachResult<T> {
  const dedupeKey = recommendationFinalDedupeKey(event)
  const duplicate = messages.find(
    (message) => message.recommendationFinalDedupeKey === dedupeKey,
  )
  if (duplicate) {
    return { messages, messageId: duplicate.id, dedupeKey, attached: false }
  }

  const requestTarget = messages.find(
    (message) =>
      message.role === "assistant" &&
      message.requestId === event.request_id &&
      message.threadId === event.thread_id,
  )
  const preferredTarget = messages.find(
    (message) =>
      message.id === preferredMessageId &&
      message.role === "assistant" &&
      (!message.requestId || message.requestId === event.request_id) &&
      (!message.threadId || message.threadId === event.thread_id),
  )
  const target = requestTarget ?? preferredTarget
  if (
    target?.recommendationFinal &&
    target.recommendationFinalDedupeKey !== dedupeKey
  ) {
    throw new ContractParseError(
      "recommendation_final_binding",
      "request already has a different recommendation final",
    )
  }

  const messageId = target?.id ?? recommendationMessageId(event)
  const base =
    target ??
    ({
      id: messageId,
      role: "assistant",
      content: "",
      requestId: event.request_id,
      threadId: event.thread_id,
    } as T)
  const merged = mergeRecommendationFinalIntoMessage(base, event)
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

function parseRecommendationItem(
  value: unknown,
  contract: string,
): RecommendationFinalItemV1 {
  const data = record(value, contract)
  exactKeys(data, RECOMMENDATION_FIELDS, contract)
  const rank = boundedInteger(data.rank, "rank", contract, 1, 50)
  const score = boundedNumber(data.score, "score", contract, 0, 1)
  return {
    recommendation_id: normalizedString(
      data.recommendation_id,
      "recommendation_id",
      contract,
      160,
    ),
    resource_id: normalizedString(data.resource_id, "resource_id", contract, 200),
    resource_type: parseResourceType(data.resource_type, "resource_type", contract),
    topic_id: normalizedString(data.topic_id, "topic_id", contract, 160),
    title: normalizedString(data.title, "title", contract, 240),
    rank,
    score,
    reason: normalizedString(data.reason, "reason", contract, 1_000),
  }
}

function parseCandidateSnapshot(
  value: unknown,
  contract: string,
): RecommendationCandidateSnapshotV1 {
  const data = record(value, contract)
  exactKeys(data, SNAPSHOT_FIELDS, contract)
  if (data.schema_version !== "recommendation_candidate_snapshot_v1") {
    fail(contract, "schema_version must equal recommendation_candidate_snapshot_v1")
  }
  if (data.source_schema_version !== "knowledge_graph_v1") {
    fail(contract, "source_schema_version must equal knowledge_graph_v1")
  }
  const sourceFingerprint = normalizedString(
    data.source_fingerprint,
    "source_fingerprint",
    contract,
    64,
  )
  if (!SHA256_PATTERN.test(sourceFingerprint)) {
    fail(contract, "source_fingerprint is invalid")
  }
  const inventoryHash = normalizedString(
    data.inventory_hash,
    "inventory_hash",
    contract,
    92,
  )
  if (!INVENTORY_HASH_PATTERN.test(inventoryHash)) {
    fail(contract, "inventory_hash is invalid")
  }
  const snapshotId = normalizedString(data.snapshot_id, "snapshot_id", contract, 96)
  if (!SNAPSHOT_ID_PATTERN.test(snapshotId)) fail(contract, "snapshot_id is invalid")
  const subject = normalizedString(data.subject, "subject", contract, 120)
  const targets = arrayValue(data.targets, "targets", contract, 50, 1).map(
    (item, index) => parseCandidate(item, `${contract}.targets.${index}`),
  )
  const snapshot: RecommendationCandidateSnapshotV1 = {
    schema_version: "recommendation_candidate_snapshot_v1",
    source_schema_version: "knowledge_graph_v1",
    source_data_version: normalizedString(
      data.source_data_version,
      "source_data_version",
      contract,
      500,
    ),
    source_fingerprint: sourceFingerprint,
    subject,
    candidate_count: boundedInteger(
      data.candidate_count,
      "candidate_count",
      contract,
      1,
      400_000,
    ),
    inventory_hash: inventoryHash,
    targets,
    snapshot_id: snapshotId,
  }
  requireUnique(
    snapshot.targets.map((target) => target.resource_id),
    "target resource_id",
    contract,
  )
  if (snapshot.candidate_count < snapshot.targets.length) {
    fail(contract, "candidate_count cannot be smaller than target count")
  }
  if (snapshot.targets.some((target) => target.subject !== snapshot.subject)) {
    fail(contract, "candidate snapshot targets must match its subject")
  }
  const { snapshot_id: _snapshotId, ...snapshotContent } = snapshot
  const expectedSnapshotId = stableHash(SNAPSHOT_ID_PREFIX, snapshotContent)
  if (snapshot.snapshot_id !== expectedSnapshotId) {
    fail(contract, "snapshot_id does not match candidate snapshot content")
  }
  return snapshot
}

function parseCandidate(
  value: unknown,
  contract: string,
): RecommendationFinalCandidateV1 {
  const data = record(value, contract)
  exactKeys(data, CANDIDATE_FIELDS, contract)
  return {
    resource_id: normalizedString(data.resource_id, "resource_id", contract, 200),
    resource_type: parseResourceType(data.resource_type, "resource_type", contract),
    subject: normalizedString(data.subject, "subject", contract, 120),
    topic_id: normalizedString(data.topic_id, "topic_id", contract, 160),
    title: normalizedString(data.title, "title", contract, 240),
  }
}

function validateTerminalTruth(event: RecommendationFinalV1, contract: string): void {
  if (event.terminal_status === "available") {
    if (
      event.user_id === null ||
      event.subject === null ||
      event.generated_at === null ||
      event.recommendations.length === 0 ||
      event.candidate_snapshot === null ||
      event.unavailable_reason !== null
    ) {
      fail(
        contract,
        "available recommendation final requires identity, generated_at, recommendations, and candidate snapshot only",
      )
    }
    if (event.candidate_snapshot.subject !== event.subject) {
      fail(contract, "candidate snapshot subject must match the final")
    }
  } else {
    if (
      event.recommendations.length > 0 ||
      event.candidate_snapshot !== null ||
      event.generated_at !== null ||
      event.unavailable_reason === null
    ) {
      fail(contract, "unavailable recommendation final requires only an explicit reason")
    }
    if (event.unavailable_reason === "missing_subject" && event.subject !== null) {
      fail(contract, "missing_subject final cannot contain a subject")
    }
    if (event.unavailable_reason === "missing_user_id" && event.user_id !== null) {
      fail(contract, "missing_user_id final cannot contain a user_id")
    }
    if (event.unavailable_reason !== "missing_user_id" && event.user_id === null) {
      fail(contract, "unavailable reasons other than missing_user_id require a user_id")
    }
    return
  }

  requireUnique(
    event.recommendations.map((item) => item.recommendation_id),
    "recommendation_id",
    contract,
  )
  const ranks = event.recommendations.map((item) => item.rank)
  if (ranks.some((rank, index) => rank !== index + 1)) {
    fail(contract, "recommendation ranks must be contiguous and ordered")
  }
  const snapshot = event.candidate_snapshot
  if (snapshot === null) fail(contract, "available final requires a candidate snapshot")
  const targetById = new Map(snapshot.targets.map((target) => [target.resource_id, target]))
  const recommendationTargetIds = new Set(
    event.recommendations.map((item) => item.resource_id),
  )
  if (
    targetById.size !== recommendationTargetIds.size ||
    [...targetById].some(([resourceId]) => !recommendationTargetIds.has(resourceId))
  ) {
    fail(contract, "candidate snapshot targets must exactly match recommendation targets")
  }
  for (const item of event.recommendations) {
    const target = targetById.get(item.resource_id)
    if (
      !target ||
      target.resource_type !== item.resource_type ||
      target.topic_id !== item.topic_id ||
      target.title !== item.title
    ) {
      fail(contract, "recommendation target differs from candidate snapshot")
    }
  }
}

function validateDerivedIdentity(event: RecommendationFinalV1, contract: string): void {
  const { recommendation_final_id: _finalId, payload_hash: _payloadHash, ...content } =
    event
  const expectedPayloadHash = stableHash(PAYLOAD_HASH_PREFIX, content)
  if (event.payload_hash !== expectedPayloadHash) {
    fail(contract, "payload_hash does not match Recommendation Final V1")
  }
  const expectedFinalId = stableHash(FINAL_ID_PREFIX, {
    thread_id: event.thread_id,
    request_id: event.request_id,
    payload_hash: event.payload_hash,
  })
  if (event.recommendation_final_id !== expectedFinalId) {
    fail(contract, "recommendation_final_id does not match request identity and payload")
  }
}

function parseResourceType(
  value: unknown,
  field: string,
  contract: string,
): RecommendationResourceType {
  const resourceType = normalizedString(value, field, contract, 80)
  if (!RESOURCE_TYPES.has(resourceType as RecommendationResourceType)) {
    fail(contract, `${field} is invalid`)
  }
  return resourceType as RecommendationResourceType
}

function parseUnavailableReason(
  value: unknown,
  contract: string,
): ExplicitRecommendationUnavailableReason | null {
  if (value === null) return null
  const reason = normalizedString(value, "unavailable_reason", contract, 80)
  if (!UNAVAILABLE_REASONS.has(reason as ExplicitRecommendationUnavailableReason)) {
    fail(contract, "unavailable_reason is invalid for an explicit recommendation")
  }
  return reason as ExplicitRecommendationUnavailableReason
}

function exactKeys(
  data: Record<string, unknown>,
  expected: readonly string[],
  contract: string,
): void {
  const allowed = new Set(expected)
  const extras = Object.keys(data)
    .filter((key) => !allowed.has(key))
    .sort()
  if (extras.length > 0) fail(contract, `unexpected field: ${extras[0]}`)
  const missing = expected.find((key) => !(key in data))
  if (missing) fail(contract, `missing field: ${missing}`)
}

function record(value: unknown, contract: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(contract, "expected an object")
  }
  return value as Record<string, unknown>
}

function arrayValue(
  value: unknown,
  field: string,
  contract: string,
  maxLength: number,
  minLength = 0,
): unknown[] {
  if (!Array.isArray(value)) fail(contract, `${field} must be an array`)
  if (value.length < minLength || value.length > maxLength) {
    fail(contract, `${field} must contain between ${minLength} and ${maxLength} items`)
  }
  return value
}

function normalizedString(
  value: unknown,
  field: string,
  contract: string,
  maxLength: number,
): string {
  if (typeof value !== "string" || !value.trim()) fail(contract, `${field} is required`)
  if (value.trim() !== value) fail(contract, `${field} must be normalized`)
  if ([...value].length > maxLength) {
    fail(contract, `${field} exceeds ${maxLength} characters`)
  }
  return value
}

function nullableNormalizedString(
  value: unknown,
  field: string,
  contract: string,
  maxLength: number,
): string | null {
  if (value === null) return null
  return normalizedString(value, field, contract, maxLength)
}

function boundedInteger(
  value: unknown,
  field: string,
  contract: string,
  minimum: number,
  maximum: number,
): number {
  if (
    !Number.isInteger(value) ||
    (value as number) < minimum ||
    (value as number) > maximum
  ) {
    fail(contract, `${field} must be an integer between ${minimum} and ${maximum}`)
  }
  return value as number
}

function boundedNumber(
  value: unknown,
  field: string,
  contract: string,
  minimum: number,
  maximum: number,
): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < minimum ||
    value > maximum
  ) {
    fail(contract, `${field} must be a finite number between ${minimum} and ${maximum}`)
  }
  return value
}

function requireUnique(values: string[], field: string, contract: string): void {
  if (new Set(values).size !== values.length) {
    fail(contract, `${field} values must be unique`)
  }
}

function isCanonicalTimezoneAwareIso(value: string): boolean {
  if (!CANONICAL_ISO_PATTERN.test(value) || !Number.isFinite(Date.parse(value))) {
    return false
  }
  const year = Number(value.slice(0, 4))
  const month = Number(value.slice(5, 7))
  const day = Number(value.slice(8, 10))
  const hour = Number(value.slice(11, 13))
  const minute = Number(value.slice(14, 16))
  const second = Number(value.slice(17, 19))
  const offset = value.slice(-6)
  const offsetHour = Number(offset.slice(1, 3))
  const offsetMinute = Number(offset.slice(4, 6))
  if (
    year < 1 ||
    month < 1 ||
    month > 12 ||
    hour > 23 ||
    minute > 59 ||
    second > 59 ||
    offsetHour > 23 ||
    offsetMinute > 59
  ) {
    return false
  }
  const monthLengths = [
    31,
    isLeapYear(year) ? 29 : 28,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
  ]
  return day >= 1 && day <= monthLengths[month - 1]
}

function isLeapYear(year: number): boolean {
  return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)
}

function stableHash(prefix: string, payload: object): string {
  return `${prefix}:${sha256Hex(stableJson(payload as JsonValue))}`
}

function stableJson(value: JsonValue, field = ""): string {
  if (value === null) return "null"
  if (typeof value === "string") return JSON.stringify(value)
  if (typeof value === "boolean") return value ? "true" : "false"
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("canonical JSON forbids non-finite numbers")
    return field === "score" ? pythonFloatJson(value) : JSON.stringify(value)
  }
  if (Array.isArray(value)) return `[${value.map((item) => stableJson(item)).join(",")}]`
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableJson(value[key], key)}`)
    .join(",")}}`
}

function pythonFloatJson(value: number): string {
  if (Object.is(value, -0)) return "-0.0"
  if (value === 0) return "0.0"
  if (Number.isInteger(value)) return `${value}.0`

  const raw = value.toString()
  if (raw.includes("e")) return normalizePythonExponent(raw)
  if (Math.abs(value) >= 0.0001) return raw

  const negative = raw.startsWith("-")
  const unsigned = negative ? raw.slice(1) : raw
  const fractional = unsigned.slice(2)
  const firstNonZero = [...fractional].findIndex((character) => character !== "0")
  if (firstNonZero < 0) return negative ? "-0.0" : "0.0"
  const digits = fractional.slice(firstNonZero).replace(/0+$/, "")
  const mantissa = digits.length === 1 ? digits : `${digits[0]}.${digits.slice(1)}`
  const exponent = firstNonZero + 1
  return `${negative ? "-" : ""}${mantissa}e-${String(exponent).padStart(2, "0")}`
}

function normalizePythonExponent(value: string): string {
  const [mantissa, exponentText] = value.split("e")
  const exponent = Number(exponentText)
  const sign = exponent >= 0 ? "+" : "-"
  return `${mantissa}e${sign}${String(Math.abs(exponent)).padStart(2, "0")}`
}

function sha256Hex(value: string): string {
  const constants = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ]
  const hash = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]
  const source = new TextEncoder().encode(value)
  const paddedLength = Math.ceil((source.length + 9) / 64) * 64
  const bytes = new Uint8Array(paddedLength)
  bytes.set(source)
  bytes[source.length] = 0x80
  const bitLength = source.length * 8
  const view = new DataView(bytes.buffer)
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x1_0000_0000), false)
  view.setUint32(paddedLength - 4, bitLength >>> 0, false)

  const words = new Uint32Array(64)
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) {
      words[index] = view.getUint32(offset + index * 4, false)
    }
    for (let index = 16; index < 64; index += 1) {
      const previous15 = words[index - 15]
      const previous2 = words[index - 2]
      const sigma0 =
        rotateRight(previous15, 7) ^ rotateRight(previous15, 18) ^ (previous15 >>> 3)
      const sigma1 =
        rotateRight(previous2, 17) ^ rotateRight(previous2, 19) ^ (previous2 >>> 10)
      words[index] =
        (words[index - 16] + sigma0 + words[index - 7] + sigma1) >>> 0
    }

    let [a, b, c, d, e, f, g, h] = hash
    for (let index = 0; index < 64; index += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25)
      const choice = (e & f) ^ (~e & g)
      const temp1 = (h + sum1 + choice + constants[index] + words[index]) >>> 0
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22)
      const majority = (a & b) ^ (a & c) ^ (b & c)
      const temp2 = (sum0 + majority) >>> 0
      h = g
      g = f
      f = e
      e = (d + temp1) >>> 0
      d = c
      c = b
      b = a
      a = (temp1 + temp2) >>> 0
    }
    hash[0] = (hash[0] + a) >>> 0
    hash[1] = (hash[1] + b) >>> 0
    hash[2] = (hash[2] + c) >>> 0
    hash[3] = (hash[3] + d) >>> 0
    hash[4] = (hash[4] + e) >>> 0
    hash[5] = (hash[5] + f) >>> 0
    hash[6] = (hash[6] + g) >>> 0
    hash[7] = (hash[7] + h) >>> 0
  }
  return hash.map((word) => word.toString(16).padStart(8, "0")).join("")
}

function rotateRight(value: number, amount: number): number {
  return (value >>> amount) | (value << (32 - amount))
}

function fail(contract: string, reason: string): never {
  throw new ContractParseError(contract, reason)
}
