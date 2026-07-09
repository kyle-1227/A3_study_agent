import type {
  CodePracticeResult,
  ExerciseResult,
  Message,
  MindmapNode,
  MindmapResult,
  ReviewDocResult,
  StudyPlanResult,
  VideoAnimationResult,
  VideoScriptResult,
} from "@/components/chat-area"

type ResourcePayload = Record<string, unknown>

export type ResourceFinalEvent = ResourcePayload & {
  type?: string
  resource_type?: string
  resource_id?: string
  payload_hash?: string
  thread_id?: string
  request_id?: string
  answer?: string
  resource?: {
    kind?: string
    title?: string
    summary?: string
    payload?: ResourcePayload
    artifact_refs?: Record<string, string>
    render_hints?: ResourcePayload
  }
}

export function resourceFinalDedupeKey(event: ResourceFinalEvent): string {
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

export function mergeResourceFinalIntoMessage(
  message: Message,
  event: ResourceFinalEvent,
  apiBaseUrl: string,
): Message {
  const payload = normalizedPayload(event)
  const finalAnswer = text(event.answer) || text(event.resource?.summary)
  const next: Message = {
    ...message,
    content: finalAnswer || message.content,
    resourceFinalPayload: event,
    resourceFinalDedupeKey: resourceFinalDedupeKey(event),
  }

  const reviewDocs = reviewDocArtifacts(payload).map((artifact) =>
    normalizeReviewDoc(artifact, apiBaseUrl),
  )
  const reviewDoc = objectValue(payload.review_doc)
  const mindmap = objectValue(payload.mindmap)
  const exerciseArtifact = objectValue(payload.exercise_artifact)
  const codePracticeArtifact = objectValue(payload.code_practice_artifact)
  const videoScriptArtifact = objectValue(payload.video_script_artifact)
  const videoAnimationArtifact = objectValue(payload.video_animation_artifact)
  const studyPlan = objectValue(payload.study_plan)

  if (reviewDocs.length > 0) {
    next.reviewDoc = undefined
    next.reviewDocs = reviewDocs
  } else if (reviewDoc) {
    next.reviewDoc = normalizeReviewDoc(reviewDoc, apiBaseUrl)
  }
  if (mindmap) next.mindmap = normalizeMindmap(mindmap, apiBaseUrl)
  if (exerciseArtifact) next.exercise = normalizeExercise(exerciseArtifact, apiBaseUrl)
  if (codePracticeArtifact) {
    next.codePractice = normalizeCodePractice(codePracticeArtifact, apiBaseUrl)
  }
  if (videoScriptArtifact) {
    next.videoScript = normalizeVideoScript(videoScriptArtifact, apiBaseUrl)
  }
  if (videoAnimationArtifact) {
    next.videoAnimation = normalizeVideoAnimation(videoAnimationArtifact, apiBaseUrl)
  }
  if (studyPlan) next.studyPlan = normalizeStudyPlan(studyPlan, apiBaseUrl)

  return next
}

function normalizedPayload(event: ResourceFinalEvent): ResourcePayload {
  const normalized = objectValue(event.resource?.payload)
  if (normalized) return normalized
  return event
}

function reviewDocArtifacts(payload: ResourcePayload): ResourcePayload[] {
  return Array.isArray(payload.review_doc_artifacts)
    ? payload.review_doc_artifacts.filter(isResourcePayload)
    : []
}

function normalizeReviewDoc(
  value: ResourcePayload,
  apiBaseUrl: string,
): ReviewDocResult {
  return {
    subject: text(value.subject),
    title: text(value.title) || "Review Document",
    filename: text(value.filename),
    markdownUrl: absoluteUrl(value.markdown_url, apiBaseUrl),
    docxFilename: text(value.docx_filename),
    docxUrl: absoluteUrl(value.docx_url, apiBaseUrl),
    markdown: text(value.markdown),
  }
}

function normalizeMindmap(
  value: ResourcePayload,
  apiBaseUrl: string,
): MindmapResult {
  return {
    title: text(value.title) || "Knowledge Mindmap",
    tree: normalizeMindmapNode(value.tree, text(value.title) || "Mindmap"),
    xmindUrl: absoluteUrl(value.xmind_url, apiBaseUrl),
  }
}

function normalizeMindmapNode(value: unknown, fallbackTitle: string): MindmapNode {
  const raw = objectValue(value)
  if (!raw) return { title: fallbackTitle }
  const children = Array.isArray(raw.children)
    ? raw.children
        .map((child) => normalizeMindmapNode(child, "Topic"))
        .filter((child) => child.title)
    : undefined
  return {
    title: text(raw.title) || fallbackTitle,
    note: text(raw.note) || undefined,
    children,
  }
}

function normalizeExercise(
  value: ResourcePayload,
  apiBaseUrl: string,
): ExerciseResult {
  return {
    title: text(value.title) || "Exercise Resource",
    filename: text(value.filename),
    markdownUrl: absoluteUrl(value.markdown_url, apiBaseUrl),
    docxFilename: text(value.docx_filename),
    docxUrl: absoluteUrl(value.docx_url, apiBaseUrl),
  }
}

function normalizeCodePractice(
  value: ResourcePayload,
  apiBaseUrl: string,
): CodePracticeResult {
  return {
    title: text(value.title) || "Code Practice",
    filename: text(value.filename),
    markdownUrl: absoluteUrl(value.markdown_url, apiBaseUrl),
    docxFilename: text(value.docx_filename),
    docxUrl: absoluteUrl(value.docx_url, apiBaseUrl),
    pythonFilename: text(value.python_filename),
    pythonUrl: absoluteUrl(value.python_url, apiBaseUrl),
    markdown: text(value.markdown),
  }
}

function normalizeVideoScript(
  value: ResourcePayload,
  apiBaseUrl: string,
): VideoScriptResult {
  return {
    title: text(value.title) || "Video Script",
    filename: text(value.filename),
    markdownUrl: absoluteUrl(value.markdown_url, apiBaseUrl),
    docxFilename: text(value.docx_filename),
    docxUrl: absoluteUrl(value.docx_url, apiBaseUrl),
    srtFilename: text(value.srt_filename),
    srtUrl: absoluteUrl(value.srt_url, apiBaseUrl),
    markdown: text(value.markdown),
    srt: text(value.srt),
  }
}

function normalizeVideoAnimation(
  value: ResourcePayload,
  apiBaseUrl: string,
): VideoAnimationResult {
  return {
    title: text(value.title) || "Video Animation",
    htmlUrl: absoluteUrl(value.html_url, apiBaseUrl),
    mp4Url: absoluteUrl(value.mp4_url, apiBaseUrl),
    srtUrl: absoluteUrl(value.srt_url, apiBaseUrl),
    jsonUrl: absoluteUrl(value.json_url, apiBaseUrl),
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
  value: ResourcePayload,
  apiBaseUrl: string,
): StudyPlanResult {
  return {
    title: text(value.title) || "Personalized Study Plan",
    filename: text(value.filename),
    markdownUrl: absoluteUrl(value.markdown_url, apiBaseUrl),
    docxFilename: text(value.docx_filename),
    docxUrl: absoluteUrl(value.docx_url, apiBaseUrl),
    markdown: text(value.markdown),
  }
}

function absoluteUrl(value: unknown, apiBaseUrl: string): string {
  const raw = text(value)
  if (!raw) return ""
  return raw.startsWith("/") ? `${apiBaseUrl}${raw}` : raw
}

function text(value: unknown): string {
  return typeof value === "string" ? value : ""
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined
}

function objectValue(value: unknown): ResourcePayload | null {
  return isResourcePayload(value) ? value : null
}

function isResourcePayload(value: unknown): value is ResourcePayload {
  return Boolean(value && typeof value === "object" && !Array.isArray(value))
}
