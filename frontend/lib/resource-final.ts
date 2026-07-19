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
import type {
  ExerciseLevel,
  ExerciseQuestionType,
  PublicExerciseCardV1,
} from "@/lib/assessment-contracts"
import { ContractParseError } from "@/lib/observability-contracts"

export type ResourceFinalExerciseQuestionType = ExerciseQuestionType
export type ResourceFinalExerciseLevel = ExerciseLevel
export type ResourceFinalExerciseItem = PublicExerciseCardV1

export interface ResourceFinalExerciseArtifact {
  schema_version: "exercise_public_artifact_v1"
  title: string
  items: ResourceFinalExerciseItem[]
}

type ResourcePayload = Record<string, unknown> & {
  exercise_artifact?: ResourceFinalExerciseArtifact
  exercise_items?: ResourceFinalExerciseItem[]
}

export type ResourceFinalTerminalStatus =
  | "success"
  | "partial_success"
  | "failed"
  | "controlled_stop"
export type ResourceFinalResourceStatus = "success" | "partial_success"
export type ResourceFinalResourceKind =
  | "mindmap"
  | "quiz"
  | "review_doc"
  | "code_practice"
  | "video_script"
  | "video_animation"
  | "study_plan"

export interface ResourceFinalValidation {
  resourceCount: number
  successCount: number
  partialSuccessCount: number
  failedCount: number
  blockedCount: number
  renderableCount: number
  downloadableCount: number
}

export interface ResourceFinalResourceValidation {
  resourceType: ResourceFinalResourceKind
  valid: true
  terminalStatus: ResourceFinalResourceStatus
  renderableCount: number
  downloadableCount: number
  verifiedLocalCount: number
  remoteUnverifiedCount: number
  warnings: string[]
}

export interface ResourceFinalResource {
  kind: ResourceFinalResourceKind
  status: ResourceFinalResourceStatus
  resource_id: string
  payload_hash: string
  title: string
  summary: string
  payload: ResourcePayload
  artifact_refs: Record<string, string>
  validation: ResourceFinalResourceValidation
}

export interface ResourceFinalRecommendation {
  recommendation_id: string
  resource_id: string
  resource_type: ResourceFinalResourceKind
  trigger: "automatic" | "explicit_request"
  rank: number
  title: string
  reason: string
}

export interface ResourceFinalBlockedResource {
  resource_type: ResourceFinalResourceKind
  status: "blocked_insufficient_evidence"
  reason_code: string
  blocked_requirement_ids: string[]
}

export interface ResourceFinalError {
  resource_type: ResourceFinalResourceKind
  error_code: string
  error_type: string
  message_sanitized: string
}

export interface ResourceFinalEvent {
  type: "resource_final"
  schema_version: "resource_final_v3"
  resource_final_id: string
  payload_hash: string
  thread_id: string
  request_id: string
  terminal_status: ResourceFinalTerminalStatus
  resources: ResourceFinalResource[]
  recommendations: ResourceFinalRecommendation[]
  blocked_resources: ResourceFinalBlockedResource[]
  errors: ResourceFinalError[]
  validation: ResourceFinalValidation
  summary: string
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
  Pick<ResourceFinalEvent, "resource_final_id" | "thread_id" | "request_id" | "payload_hash">
>

const RESOURCE_FINAL_ID_PATTERN = /^resource-final:v3:[0-9a-f]{64}$/
const RESOURCE_ID_PATTERN = /^resource:v3:[0-9a-f]{64}$/
const QUESTION_ID_PATTERN = /^question:v1:[0-9a-f]{64}$/
const PAYLOAD_HASH_PATTERN = /^payload:v3:[0-9a-f]{64}$/
const CODE_PATTERN = /^[a-z][a-z0-9_.-]{0,119}$/
const EXERCISE_ITEM_FIELDS = [
  "schema_version",
  "question_id",
  "question_type",
  "level",
  "question",
  "choices",
  "tags",
] as const
const EXERCISE_ARTIFACT_FIELDS = ["schema_version", "title", "items"] as const
const FINAL_VALIDATION_FIELDS = [
  "schema_version",
  "resource_count",
  "success_count",
  "partial_success_count",
  "failed_count",
  "blocked_count",
  "renderable_count",
  "downloadable_count",
] as const
const TOP_LEVEL_FIELDS = [
  "schema_version",
  "type",
  "thread_id",
  "request_id",
  "terminal_status",
  "resources",
  "recommendations",
  "blocked_resources",
  "errors",
  "validation",
  "summary",
  "resource_final_id",
  "payload_hash",
] as const
const MAX_RENDER_PAYLOAD_BYTES = 512 * 1024

export function parseResourceFinalEvent(value: unknown): ResourceFinalEvent {
  const contract = "resource_final_v3"
  const data = record(value, contract)
  rejectUnexpectedFields(data, TOP_LEVEL_FIELDS, contract)
  if (data.type !== "resource_final") fail(contract, "type must equal resource_final")
  if (data.schema_version !== "resource_final_v3") {
    fail(contract, "schema_version must equal resource_final_v3")
  }
  const resourceFinalId = boundedString(
    data.resource_final_id,
    "resource_final_id",
    contract,
    96,
  )
  if (!RESOURCE_FINAL_ID_PATTERN.test(resourceFinalId)) {
    fail(contract, "resource_final_id is invalid")
  }
  const payloadHash = boundedString(data.payload_hash, "payload_hash", contract, 80)
  if (!PAYLOAD_HASH_PATTERN.test(payloadHash)) fail(contract, "payload_hash is invalid")
  const terminalStatusValue = boundedString(
    data.terminal_status,
    "terminal_status",
    contract,
    40,
  )
  let terminalStatus: ResourceFinalTerminalStatus
  if (terminalStatusValue === "success") terminalStatus = "success"
  else if (terminalStatusValue === "partial_success") terminalStatus = "partial_success"
  else if (terminalStatusValue === "failed") terminalStatus = "failed"
  else if (terminalStatusValue === "controlled_stop") terminalStatus = "controlled_stop"
  else {
    fail(contract, "terminal_status is invalid")
  }
  const resources = arrayValue(data.resources, "resources", contract, 80).map(
    (item, index) => parseResource(item, `${contract}.resources.${index}`),
  )
  const recommendations = arrayValue(
    data.recommendations,
    "recommendations",
    contract,
    80,
  ).map((item, index) =>
    parseRecommendation(item, `${contract}.recommendations.${index}`),
  )
  const blockedResources = arrayValue(
    data.blocked_resources,
    "blocked_resources",
    contract,
    80,
  ).map((item, index) =>
    parseBlockedResource(item, `${contract}.blocked_resources.${index}`),
  )
  const errors = arrayValue(data.errors, "errors", contract, 80).map((item, index) =>
    parseResourceError(item, `${contract}.errors.${index}`),
  )
  const validation = parseValidation(data.validation, `${contract}.validation`)
  const event: ResourceFinalEvent = {
    type: "resource_final",
    schema_version: "resource_final_v3",
    resource_final_id: resourceFinalId,
    payload_hash: payloadHash,
    thread_id: boundedString(data.thread_id, "thread_id", contract, 120),
    request_id: boundedString(data.request_id, "request_id", contract, 120),
    terminal_status: terminalStatus,
    resources,
    recommendations,
    blocked_resources: blockedResources,
    errors,
    validation,
    summary: boundedString(data.summary, "summary", contract, 1_200),
  }
  validateTerminalTruth(event, contract)
  return event
}

export function resourceFinalDedupeKey(event: ResourceFinalIdentity): string {
  const resourceFinalId = text(event.resource_final_id)
  if (resourceFinalId) return `resource_final_id:${resourceFinalId}`
  return [
    "resource_final_payload",
    text(event.thread_id),
    text(event.request_id),
    text(event.payload_hash),
  ].join(":")
}

export function resourceMessageIdFromDedupeKey(dedupeKey: string): string {
  const safeKey = dedupeKey.replace(/[^a-zA-Z0-9_-]+/g, "-").slice(0, 96)
  return `assistant-resource-${safeKey || "unknown"}`
}

export function resourceFinalOutcome(event: ResourceFinalEvent): ResourceFinalOutcome | null {
  const summary = event.summary
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

  const next: Message = {
    ...message,
    content: event.summary,
    requestId: event.request_id,
    threadId: event.thread_id,
    resourceFinalPayload: { ...event },
    resourceFinalDedupeKey: resourceFinalDedupeKey(event),
    resourceScopeNotice:
      message.resourceScopeNotice ||
      event.resources.some(
        (resource) =>
          resource.status === "partial_success" &&
          resource.validation.warnings.includes("evidence_scope_limited"),
      ),
  }

  for (const resource of event.resources) {
    const payload = resource.payload
    const fallbackTitle = resource.title
    if (resource.kind === "review_doc") {
      const reviewDocs = reviewDocArtifacts(payload)
        .map((artifact) => normalizeReviewDoc(artifact, apiBaseUrl, fallbackTitle))
        .filter((artifact): artifact is ReviewDocResult => artifact !== null)
      const reviewDoc = normalizeReviewDoc(
        objectValue(payload.review_doc),
        apiBaseUrl,
        fallbackTitle,
      )
      if (reviewDocs.length > 0) {
        next.reviewDoc = undefined
        next.reviewDocs = reviewDocs
      } else if (reviewDoc) {
        next.reviewDoc = reviewDoc
      }
    } else if (resource.kind === "mindmap") {
      const mindmap = normalizeMindmap(
        objectValue(payload.mindmap),
        apiBaseUrl,
        fallbackTitle,
      )
      if (mindmap) next.mindmap = mindmap
    } else if (resource.kind === "quiz") {
      const exercise = normalizeExercise(
        objectValue(payload.exercise_artifact),
        apiBaseUrl,
        fallbackTitle,
        resource.resource_id,
        payload.exercise_items ?? [],
      )
      if (exercise) next.exercise = exercise
    } else if (resource.kind === "code_practice") {
      const codePractice = normalizeCodePractice(
        objectValue(payload.code_practice_artifact),
        apiBaseUrl,
        fallbackTitle,
      )
      if (codePractice) next.codePractice = codePractice
    } else if (resource.kind === "video_script") {
      const videoScript = normalizeVideoScript(
        objectValue(payload.video_script_artifact),
        apiBaseUrl,
        fallbackTitle,
      )
      if (videoScript) next.videoScript = videoScript
    } else if (resource.kind === "video_animation") {
      const videoAnimation = normalizeVideoAnimation(
        objectValue(payload.video_animation_artifact),
        apiBaseUrl,
        fallbackTitle,
      )
      if (videoAnimation) next.videoAnimation = videoAnimation
    } else if (resource.kind === "study_plan") {
      const studyPlan = normalizeStudyPlan(
        objectValue(payload.study_plan),
        apiBaseUrl,
        fallbackTitle,
      )
      if (studyPlan) next.studyPlan = studyPlan
    }
  }

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
  resourceId: string,
  questions: ResourceFinalExerciseItem[],
): ExerciseResult | null {
  if (!value) return null
  const title = text(value.title) || fallbackTitle
  if (!title) return null
  return {
    title,
    resourceId,
    questions,
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
  const document = objectValue(value.document)
  const markdown = text(value.markdown) || text(document?.markdown)
  const markdownUrl = absoluteUrl(value.markdown_url || document?.markdown_url, apiBaseUrl)
  const docxUrl = absoluteUrl(value.docx_url || document?.docx_url, apiBaseUrl)
  if (!title || (!markdown && !markdownUrl && !docxUrl)) return null
  return {
    title,
    filename: text(value.filename) || text(document?.filename),
    markdownUrl,
    docxFilename: text(value.docx_filename) || text(document?.docx_filename),
    docxUrl,
    markdown,
  }
}

function parseResource(value: unknown, contract: string): ResourceFinalResource {
  const data = record(value, contract)
  rejectUnexpectedFields(
    data,
    [
      "kind",
      "status",
      "resource_id",
      "payload_hash",
      "title",
      "summary",
      "payload",
      "artifact_refs",
      "validation",
    ],
    contract,
  )
  const kind = parseResourceKind(data.kind, "kind", contract)
  const status = boundedString(data.status, "status", contract, 40)
  if (status !== "success" && status !== "partial_success") {
    fail(contract, "status must equal success or partial_success")
  }
  const resourceId = boundedString(data.resource_id, "resource_id", contract, 96)
  if (!RESOURCE_ID_PATTERN.test(resourceId)) fail(contract, "resource_id is invalid")
  const payloadHash = boundedString(data.payload_hash, "payload_hash", contract, 80)
  if (!PAYLOAD_HASH_PATTERN.test(payloadHash)) fail(contract, "payload_hash is invalid")
  let payload = record(data.payload, `${contract}.payload`)
  if (JSON.stringify(payload).length > MAX_RENDER_PAYLOAD_BYTES) {
    fail(contract, "payload exceeds the frontend safety limit")
  }
  const expectedPayloadKeys: Record<ResourceFinalResourceKind, readonly string[]> = {
    mindmap: ["mindmap", "mindmap_artifact", "mindmap_tree"],
    quiz: ["exercise_artifact", "exercise_items"],
    review_doc: ["review_doc", "review_doc_artifacts"],
    code_practice: ["code_practice_artifact"],
    video_script: ["video_script_artifact"],
    video_animation: ["video_animation_artifact"],
    study_plan: ["study_plan"],
  }
  const populated = expectedPayloadKeys[kind].some((key) => hasRenderValue(payload[key]))
  if (!populated) fail(contract, `${kind} payload has no renderable value`)
  if (kind === "quiz") {
    payload = parseQuizPayload(payload, data.title, `${contract}.payload`)
  }
  const validation = parseResourceValidation(
    data.validation,
    `${contract}.validation`,
  )
  if (validation.resourceType !== kind) {
    fail(contract, "validation.resource_type must match kind")
  }
  if (validation.terminalStatus !== status) {
    fail(contract, "validation.terminal_status must match status")
  }
  return {
    kind,
    status,
    resource_id: resourceId,
    payload_hash: payloadHash,
    title: boundedString(data.title, "title", contract, 240),
    summary: boundedString(data.summary, "summary", contract, 1_200),
    payload,
    artifact_refs: stringRecord(data.artifact_refs, `${contract}.artifact_refs`),
    validation,
  }
}

function parseQuizPayload(
  payload: ResourcePayload,
  resourceTitleValue: unknown,
  contract: string,
): ResourcePayload {
  rejectUnexpectedFields(payload, ["exercise_artifact", "exercise_items"], contract)
  const resourceTitle = boundedString(
    resourceTitleValue,
    "resource title",
    contract,
    240,
  )
  const exerciseItems = parseExerciseItems(
    payload.exercise_items,
    `${contract}.exercise_items`,
  )
  const artifactData = record(
    payload.exercise_artifact,
    `${contract}.exercise_artifact`,
  )
  rejectUnexpectedFields(
    artifactData,
    EXERCISE_ARTIFACT_FIELDS,
    `${contract}.exercise_artifact`,
  )
  if (artifactData.schema_version !== "exercise_public_artifact_v1") {
    fail(
      `${contract}.exercise_artifact`,
      "schema_version must equal exercise_public_artifact_v1",
    )
  }
  const artifactTitle = boundedString(
    artifactData.title,
    "title",
    `${contract}.exercise_artifact`,
    240,
  )
  if (artifactTitle !== resourceTitle) {
    fail(`${contract}.exercise_artifact`, "title must match resource title")
  }
  const artifactItems = parseExerciseItems(
    artifactData.items,
    `${contract}.exercise_artifact.items`,
  )
  if (JSON.stringify(artifactItems) !== JSON.stringify(exerciseItems)) {
    fail(contract, "exercise_artifact items must match exercise_items exactly")
  }
  return {
    exercise_artifact: {
      schema_version: "exercise_public_artifact_v1",
      title: artifactTitle,
      items: artifactItems,
    },
    exercise_items: exerciseItems,
  }
}

function parseExerciseItems(
  value: unknown,
  contract: string,
): ResourceFinalExerciseItem[] {
  const items = arrayValue(value, "exercise items", contract, 1_000)
  if (items.length === 0) fail(contract, "exercise items must not be empty")
  const parsed = items.map((item, index) =>
    parseExerciseItem(item, `${contract}.${index}`),
  )
  requireUnique(
    parsed.map((item) => item.question_id),
    "question_id",
    contract,
  )
  return parsed
}

function parseExerciseItem(
  value: unknown,
  contract: string,
): ResourceFinalExerciseItem {
  const data = record(value, contract)
  rejectUnexpectedFields(data, EXERCISE_ITEM_FIELDS, contract)
  if (data.schema_version !== "exercise_card_v1") {
    fail(contract, "schema_version must equal exercise_card_v1")
  }
  const questionId = boundedString(data.question_id, "question_id", contract, 80)
  if (!QUESTION_ID_PATTERN.test(questionId)) fail(contract, "question_id is invalid")
  const questionType = parseExerciseQuestionType(
    data.question_type,
    "question_type",
    contract,
  )
  const level = parseExerciseLevel(data.level, "level", contract)
  const choices = stringArray(
    data.choices,
    "choices",
    contract,
    20,
    MAX_RENDER_PAYLOAD_BYTES,
  )
  const tags = stringArray(
    data.tags,
    "tags",
    contract,
    80,
    MAX_RENDER_PAYLOAD_BYTES,
  )
  if (tags.length === 0) fail(contract, "tags must not be empty")
  requireUnique(choices, "choice", contract)
  requireUnique(tags, "tag", contract)
  if (questionType === "free_text" && choices.length > 0) {
    fail(contract, "free_text questions must not define choices")
  }
  if (questionType === "single_choice" && choices.length < 2) {
    fail(contract, "single_choice questions require at least two choices")
  }
  return {
    schema_version: "exercise_card_v1",
    question_id: questionId,
    question_type: questionType,
    level,
    question: boundedString(data.question, "question", contract, 10_000),
    choices,
    tags,
  }
}

function parseExerciseQuestionType(
  value: unknown,
  field: string,
  contract: string,
): ResourceFinalExerciseQuestionType {
  const questionType = boundedString(value, field, contract, 40)
  if (questionType === "free_text") return "free_text"
  if (questionType === "single_choice") return "single_choice"
  fail(contract, `${field} is invalid`)
}

function parseExerciseLevel(
  value: unknown,
  field: string,
  contract: string,
): ResourceFinalExerciseLevel {
  const level = boundedString(value, field, contract, 40)
  if (level === "basic") return "basic"
  if (level === "intermediate") return "intermediate"
  if (level === "application") return "application"
  if (level === "self_check") return "self_check"
  fail(contract, `${field} is invalid`)
}

function parseResourceValidation(
  value: unknown,
  contract: string,
): ResourceFinalResourceValidation {
  const data = record(value, contract)
  rejectUnexpectedFields(
    data,
    [
      "schema_version",
      "resource_type",
      "valid",
      "terminal_status",
      "renderable_count",
      "downloadable_count",
      "verified_local_count",
      "remote_unverified_count",
      "failure_reason",
      "warnings",
    ],
    contract,
  )
  if (data.schema_version !== "resource_validation_v1") {
    fail(contract, "schema_version must equal resource_validation_v1")
  }
  if (data.valid !== true) fail(contract, "valid must equal true")
  if (data.failure_reason !== "") fail(contract, "failure_reason must be empty")
  const terminalStatus = boundedString(
    data.terminal_status,
    "terminal_status",
    contract,
    40,
  )
  if (terminalStatus !== "success" && terminalStatus !== "partial_success") {
    fail(contract, "terminal_status must be success or partial_success")
  }
  const warnings = stringArray(data.warnings, "warnings", contract, 24, 1_200)
  const verifiedLocalCount = boundedCount(
    data.verified_local_count,
    "verified_local_count",
    contract,
  )
  const remoteUnverifiedCount = boundedCount(
    data.remote_unverified_count,
    "remote_unverified_count",
    contract,
  )
  const downloadableCount = boundedCount(
    data.downloadable_count,
    "downloadable_count",
    contract,
  )
  if (verifiedLocalCount + remoteUnverifiedCount > downloadableCount) {
    fail(contract, "reference counts exceed downloadable_count")
  }
  return {
    resourceType: parseResourceKind(data.resource_type, "resource_type", contract),
    valid: true,
    terminalStatus,
    renderableCount: boundedCount(data.renderable_count, "renderable_count", contract),
    downloadableCount,
    verifiedLocalCount,
    remoteUnverifiedCount,
    warnings,
  }
}

function parseRecommendation(
  value: unknown,
  contract: string,
): ResourceFinalRecommendation {
  const data = record(value, contract)
  rejectUnexpectedFields(
    data,
    [
      "recommendation_id",
      "resource_id",
      "resource_type",
      "trigger",
      "rank",
      "title",
      "reason",
    ],
    contract,
  )
  const trigger = boundedString(data.trigger, "trigger", contract, 40)
  if (trigger !== "automatic" && trigger !== "explicit_request") {
    fail(contract, "trigger is invalid")
  }
  const rank = boundedCount(data.rank, "rank", contract)
  if (rank < 1 || rank > 100) fail(contract, "rank must be between 1 and 100")
  return {
    recommendation_id: boundedString(
      data.recommendation_id,
      "recommendation_id",
      contract,
      160,
    ),
    resource_id: boundedString(data.resource_id, "resource_id", contract, 200),
    resource_type: parseResourceKind(data.resource_type, "resource_type", contract),
    trigger,
    rank,
    title: boundedString(data.title, "title", contract, 240),
    reason: boundedString(data.reason, "reason", contract, 1_200),
  }
}

function parseBlockedResource(
  value: unknown,
  contract: string,
): ResourceFinalBlockedResource {
  const data = record(value, contract)
  rejectUnexpectedFields(
    data,
    ["resource_type", "status", "reason_code", "blocked_requirement_ids"],
    contract,
  )
  if (data.status !== "blocked_insufficient_evidence") {
    fail(contract, "status must equal blocked_insufficient_evidence")
  }
  const reasonCode = boundedString(data.reason_code, "reason_code", contract, 120)
  if (!CODE_PATTERN.test(reasonCode)) fail(contract, "reason_code is invalid")
  const ids = stringArray(
    data.blocked_requirement_ids,
    "blocked_requirement_ids",
    contract,
    80,
    160,
  )
  if (new Set(ids).size !== ids.length) {
    fail(contract, "blocked_requirement_ids must be unique")
  }
  return {
    resource_type: parseResourceKind(data.resource_type, "resource_type", contract),
    status: "blocked_insufficient_evidence",
    reason_code: reasonCode,
    blocked_requirement_ids: ids,
  }
}

function parseResourceError(value: unknown, contract: string): ResourceFinalError {
  const data = record(value, contract)
  rejectUnexpectedFields(
    data,
    ["resource_type", "error_code", "error_type", "message_sanitized"],
    contract,
  )
  const errorCode = boundedString(data.error_code, "error_code", contract, 120)
  if (!CODE_PATTERN.test(errorCode)) fail(contract, "error_code is invalid")
  return {
    resource_type: parseResourceKind(data.resource_type, "resource_type", contract),
    error_code: errorCode,
    error_type: boundedString(data.error_type, "error_type", contract, 160),
    message_sanitized: boundedString(
      data.message_sanitized,
      "message_sanitized",
      contract,
      1_200,
    ),
  }
}

function parseValidation(value: unknown, contract: string): ResourceFinalValidation {
  const data = record(value, contract)
  rejectUnexpectedFields(data, FINAL_VALIDATION_FIELDS, contract)
  if (data.schema_version !== "resource_final_validation_v3") {
    fail(contract, "schema_version must equal resource_final_validation_v3")
  }
  const counts = Object.fromEntries(
    FINAL_VALIDATION_FIELDS.filter((key) => key !== "schema_version").map((key) => [
      key,
      boundedCount(data[key], key, contract),
    ]),
  ) as Record<Exclude<(typeof FINAL_VALIDATION_FIELDS)[number], "schema_version">, number>
  return {
    resourceCount: counts.resource_count,
    successCount: counts.success_count,
    partialSuccessCount: counts.partial_success_count,
    failedCount: counts.failed_count,
    blockedCount: counts.blocked_count,
    renderableCount: counts.renderable_count,
    downloadableCount: counts.downloadable_count,
  }
}

function validateTerminalTruth(event: ResourceFinalEvent, contract: string): void {
  const successCount = event.resources.filter((item) => item.status === "success").length
  const partialCount = event.resources.filter(
    (item) => item.status === "partial_success",
  ).length
  const expected = {
    resourceCount: event.resources.length,
    successCount,
    partialSuccessCount: partialCount,
    failedCount: event.errors.length,
    blockedCount: event.blocked_resources.length,
    renderableCount: event.resources.reduce(
      (total, item) => total + item.validation.renderableCount,
      0,
    ),
    downloadableCount: event.resources.reduce(
      (total, item) => total + item.validation.downloadableCount,
      0,
    ),
  }
  for (const [field, value] of Object.entries(expected)) {
    if (event.validation[field as keyof ResourceFinalValidation] !== value) {
      fail(contract, `validation.${field} does not match observed resources`)
    }
  }
  if (event.terminal_status === "success") {
    if (
      event.resources.length === 0 ||
      partialCount > 0 ||
      event.errors.length > 0 ||
      event.blocked_resources.length > 0
    ) {
      fail(contract, "success terminal truth is inconsistent")
    }
  } else if (event.terminal_status === "partial_success") {
    if (
      event.resources.length === 0 ||
      (partialCount === 0 && event.errors.length === 0 && event.blocked_resources.length === 0)
    ) {
      fail(contract, "partial_success terminal truth is inconsistent")
    }
  } else if (event.terminal_status === "failed") {
    if (event.resources.length > 0 || event.errors.length === 0) {
      fail(contract, "failed terminal truth is inconsistent")
    }
  } else if (
    event.resources.length > 0 ||
    event.errors.length > 0 ||
    event.blocked_resources.length === 0
  ) {
    fail(contract, "controlled_stop terminal truth is inconsistent")
  }
  requireUnique(event.resources.map((item) => item.resource_id), "resource_id", contract)
  requireUnique(
    event.recommendations.map((item) => item.recommendation_id),
    "recommendation_id",
    contract,
  )
  requireUnique(
    event.recommendations.map((item) => String(item.rank)),
    "recommendation rank",
    contract,
  )
  const resourceById = new Map(event.resources.map((item) => [item.resource_id, item]))
  for (const recommendation of event.recommendations) {
    if (recommendation.trigger !== "automatic") continue
    const target = resourceById.get(recommendation.resource_id)
    if (!target) {
      fail(contract, "automatic recommendation must target a generated resource")
    }
    if (target.kind !== recommendation.resource_type) {
      fail(contract, "automatic recommendation resource type must match its target")
    }
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

function parseResourceKind(
  value: unknown,
  field: string,
  contract: string,
): ResourceFinalResourceKind {
  const kind = boundedString(value, field, contract, 80)
  if (kind === "mindmap") return "mindmap"
  if (kind === "quiz") return "quiz"
  if (kind === "review_doc") return "review_doc"
  if (kind === "code_practice") return "code_practice"
  if (kind === "video_script") return "video_script"
  if (kind === "video_animation") return "video_animation"
  if (kind === "study_plan") return "study_plan"
  fail(contract, `${field} is invalid`)
}

function arrayValue(
  value: unknown,
  field: string,
  contract: string,
  maxLength: number,
): unknown[] {
  if (!Array.isArray(value)) fail(contract, `${field} must be an array`)
  if (value.length > maxLength) fail(contract, `${field} has too many items`)
  return value
}

function stringArray(
  value: unknown,
  field: string,
  contract: string,
  maxItems: number,
  maxLength: number,
): string[] {
  return arrayValue(value, field, contract, maxItems).map((item, index) =>
    boundedString(item, `${field}.${index}`, contract, maxLength),
  )
}

function rejectUnexpectedFields(
  value: ResourcePayload,
  fields: readonly string[],
  contract: string,
): void {
  const allowed = new Set(fields)
  const extra = Object.keys(value).filter((key) => !allowed.has(key)).sort()
  if (extra.length > 0) fail(contract, `unexpected field: ${extra[0]}`)
}

function requireUnique(values: string[], field: string, contract: string): void {
  if (new Set(values).size !== values.length) {
    fail(contract, `${field} values must be unique`)
  }
}

function hasRenderValue(value: unknown): boolean {
  if (value === null || value === undefined || value === "") return false
  if (Array.isArray(value)) return value.length > 0
  if (isResourcePayload(value)) return Object.keys(value).length > 0
  return true
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
