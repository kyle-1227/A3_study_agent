import type {
  CodePracticeResult,
  ExerciseResult,
  Message,
  MindmapNode,
  MindmapResult,
  ResourceGenerationState,
  ReviewDocResult,
  StudyPlanResult,
  VideoAnimationResult,
  VideoScriptResult,
} from "@/components/chat-area"
import { ContractParseError } from "@/lib/observability-contracts"

type ResourcePayload = Record<string, unknown>

export type ResourceFinalTerminalStatus =
  | "success"
  | "partial_success"
  | "failed"
  | "controlled_stop"
  | "unknown"

export interface ResourceFinalValidation {
  successCount: number
  partialSuccessCount: number
  failedCount: number
  blockedCount: number
  renderableResourceCount: number
  renderableCount: number
  downloadableCount: number
}

export interface ResourceFinalEvent extends ResourcePayload {
  type: "resource_final"
  schema_version: 1 | 2
  resource_type: string
  resource_id: string
  payload_hash: string
  thread_id: string
  request_id: string
  terminal_status: ResourceFinalTerminalStatus
  validation: ResourceFinalValidation
  answer?: string
  controlled_stop?: boolean
  resource: {
    kind: string
    title: string
    summary: string
    payload: ResourcePayload
    artifact_refs: Record<string, string>
    render_hints: ResourcePayload
  }
}

export interface ResourceFinalOutcome {
  state: ResourceGenerationState
  summary: string
  completionKind:
    | "with_resource"
    | "partial_resource"
    | "controlled_stop"
    | undefined
  hasReceivedResourceFinal: boolean
}

type ResourceFinalIdentity = Partial<
  Pick<
    ResourceFinalEvent,
    "resource_id" | "thread_id" | "request_id" | "resource_type" | "payload_hash"
  >
> & { resource?: { kind?: string } }

const RESOURCE_ID_PATTERN = /^resource:v1:[0-9a-f]{64}$/
const PAYLOAD_HASH_PATTERN = /^payload:v1:[0-9a-f]{64}$/
const RESOURCE_TYPE_PATTERN = /^[a-z][a-z0-9_]{0,79}$/
const TERMINAL_STATUSES = new Set<ResourceFinalTerminalStatus>([
  "success",
  "partial_success",
  "failed",
  "controlled_stop",
  "unknown",
])
const VALIDATION_FIELDS = [
  "success_count",
  "partial_success_count",
  "failed_count",
  "blocked_count",
  "renderable_resource_count",
  "renderable_count",
  "downloadable_count",
] as const
const MAX_RENDER_PAYLOAD_BYTES = 512 * 1024

export function parseResourceFinalEvent(value: unknown): ResourceFinalEvent {
  const contract = "resource_final_v2"
  const data = record(value, contract)
  if (data.type !== "resource_final") fail(contract, "type must equal resource_final")
  if (data.schema_version !== 1 && data.schema_version !== 2) {
    fail(contract, "schema_version must equal 1 or 2")
  }

  const resourceType = boundedString(data.resource_type, "resource_type", contract, 80)
  if (!RESOURCE_TYPE_PATTERN.test(resourceType)) fail(contract, "resource_type is invalid")
  const resourceId = boundedString(data.resource_id, "resource_id", contract, 80)
  if (!RESOURCE_ID_PATTERN.test(resourceId)) fail(contract, "resource_id is invalid")
  const payloadHash = boundedString(data.payload_hash, "payload_hash", contract, 80)
  if (!PAYLOAD_HASH_PATTERN.test(payloadHash)) fail(contract, "payload_hash is invalid")

  const resource = record(data.resource, `${contract}.resource`)
  const kind = boundedString(resource.kind, "resource.kind", contract, 80)
  if (kind !== resourceType) fail(contract, "resource.kind must match resource_type")
  const payload = record(resource.payload, `${contract}.resource.payload`)
  const serializedPayload = JSON.stringify(payload)
  if (serializedPayload.length > MAX_RENDER_PAYLOAD_BYTES) {
    fail(contract, "resource.payload exceeds the frontend safety limit")
  }

  const schemaVersion = data.schema_version
  const terminalStatus =
    schemaVersion === 1
      ? "unknown"
      : boundedString(data.terminal_status, "terminal_status", contract, 40)
  if (!TERMINAL_STATUSES.has(terminalStatus as ResourceFinalTerminalStatus)) {
    fail(contract, "terminal_status is invalid")
  }
  const validation =
    schemaVersion === 1
      ? emptyValidation()
      : parseValidation(data.validation, `${contract}.validation`)

  return {
    type: "resource_final",
    schema_version: schemaVersion,
    resource_type: resourceType,
    resource_id: resourceId,
    payload_hash: payloadHash,
    thread_id: boundedString(data.thread_id, "thread_id", contract, 120),
    request_id: boundedString(data.request_id, "request_id", contract, 120),
    terminal_status: terminalStatus as ResourceFinalTerminalStatus,
    validation,
    answer: optionalBoundedString(data.answer, "answer", contract, 12_000),
    controlled_stop: data.controlled_stop === true,
    resource: {
      kind,
      title: boundedString(resource.title, "resource.title", contract, 180),
      summary: optionalBoundedString(resource.summary, "resource.summary", contract, 1_200),
      payload,
      artifact_refs: stringRecord(resource.artifact_refs, `${contract}.artifact_refs`),
      render_hints: record(resource.render_hints, `${contract}.render_hints`),
    },
  }
}

export function resourceFinalDedupeKey(event: ResourceFinalIdentity): string {
  const resourceId = text(event.resource_id)
  if (resourceId) return `resource_id:${resourceId}`
  return [
    "resource_payload",
    text(event.thread_id),
    text(event.request_id),
    text(event.resource_type || event.resource?.kind),
    text(event.payload_hash),
  ].join(":")
}

export function resourceMessageIdFromDedupeKey(dedupeKey: string): string {
  const safeKey = dedupeKey.replace(/[^a-zA-Z0-9_-]+/g, "-").slice(0, 96)
  return `assistant-resource-${safeKey || "unknown"}`
}

export function isCompletedWithoutResourceDiagnostic(event: ResourcePayload): boolean {
  return event.type === "resource_final_diagnostic" && event.status === "completed_without_resource"
}

export function resourceFinalOutcome(event: ResourceFinalEvent): ResourceFinalOutcome | null {
  const summary = event.resource.summary || event.resource.title
  if (event.terminal_status === "success") {
    return {
      state: "completed_with_resource",
      summary,
      completionKind: "with_resource",
      hasReceivedResourceFinal: true,
    }
  }
  if (event.terminal_status === "partial_success") {
    return {
      state: "partial_success",
      summary,
      completionKind: "partial_resource",
      hasReceivedResourceFinal: true,
    }
  }
  if (event.terminal_status === "failed") {
    return {
      state: "failed",
      summary,
      completionKind: undefined,
      hasReceivedResourceFinal: true,
    }
  }
  if (event.terminal_status === "controlled_stop") {
    return {
      state: "controlled_stop",
      summary,
      completionKind: "controlled_stop",
      hasReceivedResourceFinal: true,
    }
  }
  return null
}

export function mergeResourceFinalIntoMessage(
  message: Message,
  event: ResourceFinalEvent,
  apiBaseUrl: string,
): Message {
  if (message.role !== "assistant") {
    throw new ContractParseError("resource_final_binding", "target message must be assistant")
  }
  if (message.threadId && message.threadId !== event.thread_id) {
    throw new ContractParseError("resource_final_binding", "thread_id does not match target message")
  }
  if (message.requestId && message.requestId !== event.request_id) {
    throw new ContractParseError("resource_final_binding", "request_id does not match target message")
  }

  const payload = event.resource.payload
  const finalAnswer = event.answer || event.resource.summary
  const next: Message = {
    ...message,
    content: finalAnswer || message.content,
    requestId: event.request_id,
    threadId: event.thread_id,
    resourceFinalPayload: event,
    resourceFinalDedupeKey: resourceFinalDedupeKey(event),
  }
  const fallbackTitle = event.resource.title

  const reviewDocs = reviewDocArtifacts(payload)
    .map((artifact) => normalizeReviewDoc(artifact, apiBaseUrl, fallbackTitle))
    .filter((artifact): artifact is ReviewDocResult => artifact !== null)
  const reviewDoc = normalizeReviewDoc(
    objectValue(payload.review_doc),
    apiBaseUrl,
    fallbackTitle,
  )
  const mindmap = normalizeMindmap(objectValue(payload.mindmap), apiBaseUrl, fallbackTitle)
  const exercise = normalizeExercise(
    objectValue(payload.exercise_artifact),
    apiBaseUrl,
    fallbackTitle,
  )
  const codePractice = normalizeCodePractice(
    objectValue(payload.code_practice_artifact),
    apiBaseUrl,
    fallbackTitle,
  )
  const videoScript = normalizeVideoScript(
    objectValue(payload.video_script_artifact),
    apiBaseUrl,
    fallbackTitle,
  )
  const videoAnimation = normalizeVideoAnimation(
    objectValue(payload.video_animation_artifact),
    apiBaseUrl,
    fallbackTitle,
  )
  const studyPlan = normalizeStudyPlan(
    objectValue(payload.study_plan),
    apiBaseUrl,
    fallbackTitle,
  )

  if (reviewDocs.length > 0) {
    next.reviewDoc = undefined
    next.reviewDocs = reviewDocs
  } else if (reviewDoc) {
    next.reviewDoc = reviewDoc
  }
  if (mindmap) next.mindmap = mindmap
  if (exercise) next.exercise = exercise
  if (codePractice) next.codePractice = codePractice
  if (videoScript) next.videoScript = videoScript
  if (videoAnimation) next.videoAnimation = videoAnimation
  if (studyPlan) next.studyPlan = studyPlan

  return next
}

function reviewDocArtifacts(payload: ResourcePayload): ResourcePayload[] {
  return Array.isArray(payload.review_doc_artifacts)
    ? payload.review_doc_artifacts.filter(isResourcePayload)
    : []
}

function normalizeReviewDoc(
  value: ResourcePayload | null,
  apiBaseUrl: string,
  fallbackTitle: string,
): ReviewDocResult | null {
  if (!value) return null
  const title = text(value.title) || fallbackTitle
  const markdown = text(value.markdown)
  const markdownUrl = absoluteUrl(value.markdown_url, apiBaseUrl)
  const docxUrl = absoluteUrl(value.docx_url, apiBaseUrl)
  if (!title || (!markdown && !markdownUrl && !docxUrl)) return null
  return {
    subject: text(value.subject),
    title,
    filename: text(value.filename),
    markdownUrl,
    docxFilename: text(value.docx_filename),
    docxUrl,
    markdown,
  }
}

function normalizeMindmap(
  value: ResourcePayload | null,
  apiBaseUrl: string,
  fallbackTitle: string,
): MindmapResult | null {
  if (!value) return null
  const title = text(value.title) || fallbackTitle
  const tree = normalizeMindmapNode(value.tree)
  if (!title || !tree) return null
  return {
    title,
    tree,
    xmindUrl: absoluteUrl(value.xmind_url, apiBaseUrl),
  }
}

function normalizeMindmapNode(value: unknown): MindmapNode | null {
  const raw = objectValue(value)
  const title = text(raw?.title)
  if (!raw || !title) return null
  const children = Array.isArray(raw.children)
    ? raw.children
        .map((child) => normalizeMindmapNode(child))
        .filter((child): child is MindmapNode => child !== null)
    : undefined
  return { title, note: text(raw.note) || undefined, children }
}

function normalizeExercise(
  value: ResourcePayload | null,
  apiBaseUrl: string,
  fallbackTitle: string,
): ExerciseResult | null {
  if (!value) return null
  const title = text(value.title) || fallbackTitle
  if (!title) return null
  return {
    title,
    filename: text(value.filename),
    markdownUrl: absoluteUrl(value.markdown_url, apiBaseUrl),
    docxFilename: text(value.docx_filename),
    docxUrl: absoluteUrl(value.docx_url, apiBaseUrl),
  }
}

function normalizeCodePractice(
  value: ResourcePayload | null,
  apiBaseUrl: string,
  fallbackTitle: string,
): CodePracticeResult | null {
  if (!value) return null
  const title = text(value.title) || fallbackTitle
  const markdown = text(value.markdown)
  const markdownUrl = absoluteUrl(value.markdown_url, apiBaseUrl)
  const docxUrl = absoluteUrl(value.docx_url, apiBaseUrl)
  const pythonUrl = absoluteUrl(value.python_url || value.source_url, apiBaseUrl)
  if (!title || (!markdown && !markdownUrl && !docxUrl && !pythonUrl)) return null
  return {
    title,
    filename: text(value.filename),
    markdownUrl,
    docxFilename: text(value.docx_filename),
    docxUrl,
    pythonFilename: text(value.python_filename || value.source_filename),
    pythonUrl,
    markdown,
  }
}

function normalizeVideoScript(
  value: ResourcePayload | null,
  apiBaseUrl: string,
  fallbackTitle: string,
): VideoScriptResult | null {
  if (!value) return null
  const title = text(value.title) || fallbackTitle
  const markdown = text(value.markdown)
  const srt = text(value.srt)
  const markdownUrl = absoluteUrl(value.markdown_url, apiBaseUrl)
  const docxUrl = absoluteUrl(value.docx_url, apiBaseUrl)
  const srtUrl = absoluteUrl(value.srt_url, apiBaseUrl)
  if (!title || (!markdown && !srt && !markdownUrl && !docxUrl && !srtUrl)) return null
  return {
    title,
    filename: text(value.filename),
    markdownUrl,
    docxFilename: text(value.docx_filename),
    docxUrl,
    srtFilename: text(value.srt_filename),
    srtUrl,
    markdown,
    srt,
  }
}

function normalizeVideoAnimation(
  value: ResourcePayload | null,
  apiBaseUrl: string,
  fallbackTitle: string,
): VideoAnimationResult | null {
  if (!value) return null
  const title = text(value.title) || fallbackTitle
  const htmlUrl = absoluteUrl(value.html_url, apiBaseUrl)
  const mp4Url = absoluteUrl(value.mp4_url, apiBaseUrl)
  const srtUrl = absoluteUrl(value.srt_url, apiBaseUrl)
  const jsonUrl = absoluteUrl(value.json_url, apiBaseUrl)
  if (!title || (!htmlUrl && !mp4Url && !srtUrl && !jsonUrl)) return null
  return {
    title,
    htmlUrl,
    mp4Url,
    srtUrl,
    jsonUrl,
    durationSeconds: numberValue(value.duration_seconds),
    fullDurationSeconds: numberValue(value.full_duration_seconds),
    renderDurationSeconds: numberValue(value.render_duration_seconds),
    renderMode: text(value.render_mode),
    renderSuccess: value.render_success === true,
    mp4Available: value.mp4_available === true,
    isPreviewVideo: value.is_preview_video === true,
    videoValidForTeaching: value.video_valid_for_teaching === true,
    renderLog: text(value.render_log),
  }
}

function normalizeStudyPlan(
  value: ResourcePayload | null,
  apiBaseUrl: string,
  fallbackTitle: string,
): StudyPlanResult | null {
  if (!value) return null
  const title = text(value.title) || fallbackTitle
  const markdown = text(value.markdown)
  const markdownUrl = absoluteUrl(value.markdown_url, apiBaseUrl)
  const docxUrl = absoluteUrl(value.docx_url, apiBaseUrl)
  if (!title || (!markdown && !markdownUrl && !docxUrl)) return null
  return {
    title,
    filename: text(value.filename),
    markdownUrl,
    docxFilename: text(value.docx_filename),
    docxUrl,
    markdown,
  }
}

function parseValidation(value: unknown, contract: string): ResourceFinalValidation {
  const data = record(value, contract)
  const extra = Object.keys(data).filter(
    (key) => !(VALIDATION_FIELDS as readonly string[]).includes(key),
  )
  if (extra.length > 0) fail(contract, `unexpected field: ${extra.sort()[0]}`)
  const counts = Object.fromEntries(
    VALIDATION_FIELDS.map((key) => [key, boundedCount(data[key], key, contract)]),
  ) as Record<(typeof VALIDATION_FIELDS)[number], number>
  return {
    successCount: counts.success_count,
    partialSuccessCount: counts.partial_success_count,
    failedCount: counts.failed_count,
    blockedCount: counts.blocked_count,
    renderableResourceCount: counts.renderable_resource_count,
    renderableCount: counts.renderable_count,
    downloadableCount: counts.downloadable_count,
  }
}

function emptyValidation(): ResourceFinalValidation {
  return {
    successCount: 0,
    partialSuccessCount: 0,
    failedCount: 0,
    blockedCount: 0,
    renderableResourceCount: 0,
    renderableCount: 0,
    downloadableCount: 0,
  }
}

function stringRecord(value: unknown, contract: string): Record<string, string> {
  const data = record(value, contract)
  if (Object.keys(data).length > 80) fail(contract, "too many artifact references")
  return Object.fromEntries(
    Object.entries(data).map(([key, item]) => [
      boundedString(key, "artifact reference key", contract, 120),
      boundedString(item, `artifact reference ${key}`, contract, 800),
    ]),
  )
}

function absoluteUrl(value: unknown, apiBaseUrl: string): string {
  const raw = text(value)
  if (!raw) return ""
  if (raw.startsWith("/")) return `${apiBaseUrl}${raw}`
  try {
    const parsed = new URL(raw)
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? raw : ""
  } catch {
    return ""
  }
}

function boundedCount(value: unknown, field: string, contract: string): number {
  if (!Number.isInteger(value) || (value as number) < 0 || (value as number) > 10_000) {
    fail(contract, `${field} must be an integer between 0 and 10000`)
  }
  return value as number
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

function record(value: unknown, contract: string): ResourcePayload {
  if (!isResourcePayload(value)) fail(contract, "expected an object")
  return value
}

function text(value: unknown): string {
  return typeof value === "string" ? value : ""
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function objectValue(value: unknown): ResourcePayload | null {
  return isResourcePayload(value) ? value : null
}

function isResourcePayload(value: unknown): value is ResourcePayload {
  return Boolean(value && typeof value === "object" && !Array.isArray(value))
}

function fail(contract: string, reason: string): never {
  throw new ContractParseError(contract, reason)
}
