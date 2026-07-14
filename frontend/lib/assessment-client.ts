import {
  consumeAgentStreamV2,
  type AgentStreamReplay,
} from "@/lib/agent-stream-client"
import type { AgentStreamEventV2 } from "@/lib/agent-stream-contracts"
import {
  AssessmentContractError,
  parseAssessmentAttemptV1,
  parseAssessmentExpectedIdentityV1,
  parseAssessmentFinalV1,
  type AssessmentAttemptV1,
  type AssessmentExpectedIdentityV1,
  type AssessmentFinalV1,
} from "@/lib/assessment-contracts"

export interface ConsumeAssessmentStreamOptions {
  initialBody: ReadableStream<Uint8Array>
  expected: AssessmentExpectedIdentityV1
  reconnect?: AgentStreamReplay
  signal?: AbortSignal
}

export interface SubmitAssessmentAttemptOptions {
  apiBaseUrl: string
  threadId: string
  attempt: AssessmentAttemptV1
  fetchImpl: typeof fetch
  headers?: HeadersInit
  reconnect?: AgentStreamReplay
  signal?: AbortSignal
}

export type AssessmentClientErrorCode =
  | "assessment_client_configuration_invalid"
  | "assessment_http_failed"
  | "assessment_response_body_missing"
  | "assessment_response_content_type_invalid"
  | "assessment_stream_incomplete"
  | "assessment_stream_invalid_json"
  | "assessment_stream_protocol_error"
  | "assessment_stream_transport_failed"

export class AssessmentClientError extends Error {
  readonly code: AssessmentClientErrorCode

  constructor(code: AssessmentClientErrorCode) {
    super(code)
    this.name = "AssessmentClientError"
    this.code = code
  }
}

export class AssessmentHttpError extends AssessmentClientError {
  readonly status: number

  constructor(status: number) {
    super("assessment_http_failed")
    this.name = "AssessmentHttpError"
    this.status = status
  }
}

export class AssessmentRemoteError extends Error {
  readonly errorType: string
  readonly recoverable: false

  constructor(errorType: string) {
    super("assessment_attempt_failed")
    this.name = "AssessmentRemoteError"
    this.errorType = errorType
    this.recoverable = false
  }
}

type AssessmentTerminal =
  | { type: "assessment_final"; final: AssessmentFinalV1 }
  | { type: "stream_error"; errorType: string }

interface AssessmentStreamState {
  streamId: string
  highestSequence: number
  fingerprints: Map<number, string>
  started: boolean
  terminal: AssessmentTerminal | null
  done: boolean
}

const ASSESSMENT_STREAM_EVENT_TYPES = new Set([
  "stream_start",
  "assessment_final",
  "stream_error",
  "stream_done",
])
const ERROR_CODE_PATTERN = /^[a-z][a-z0-9_.-]{0,119}$/

export async function submitAssessmentAttempt({
  apiBaseUrl,
  threadId,
  attempt,
  fetchImpl,
  headers,
  reconnect,
  signal,
}: SubmitAssessmentAttemptOptions): Promise<AssessmentFinalV1> {
  const parsedAttempt = parseAssessmentAttemptV1(attempt)
  const expected = parseAssessmentExpectedIdentityV1({
    thread_id: threadId,
    request_id: parsedAttempt.request_id,
    resource_id: parsedAttempt.resource_id,
    question_id: parsedAttempt.question_id,
    time_spent_seconds: parsedAttempt.time_spent_seconds,
  })
  const endpoint = assessmentAttemptEndpoint(apiBaseUrl, expected.thread_id)
  const requestHeaders = new Headers(headers)
  requestHeaders.set("Accept", "text/event-stream")
  requestHeaders.set("Content-Type", "application/json")

  let response: Response
  try {
    response = await fetchImpl(endpoint, {
      method: "POST",
      headers: requestHeaders,
      body: JSON.stringify(parsedAttempt),
      signal,
    })
  } catch (error) {
    if (isAbort(error, signal)) throw abortError()
    throw new AssessmentClientError("assessment_stream_transport_failed")
  }

  if (!response.ok) throw new AssessmentHttpError(response.status)
  const contentType = response.headers.get("Content-Type")?.toLowerCase() ?? ""
  if (!contentType.includes("text/event-stream")) {
    throw new AssessmentClientError("assessment_response_content_type_invalid")
  }
  if (!response.body) {
    throw new AssessmentClientError("assessment_response_body_missing")
  }

  return consumeAssessmentStreamV2({
    initialBody: response.body,
    expected,
    reconnect,
    signal,
  })
}

export async function consumeAssessmentStreamV2({
  initialBody,
  expected,
  reconnect,
  signal,
}: ConsumeAssessmentStreamOptions): Promise<AssessmentFinalV1> {
  const identity = parseAssessmentExpectedIdentityV1(expected)
  const state: AssessmentStreamState = {
    streamId: "",
    highestSequence: 0,
    fingerprints: new Map(),
    started: false,
    terminal: null,
    done: false,
  }

  try {
    await consumeAgentStreamV2({
      initialBody,
      onEvent: (event) => acceptStreamEvent(event, identity, state),
      reconnect: guardedReplay(reconnect, signal),
      signal,
    })
  } catch (error) {
    if (isAbort(error, signal)) throw abortError()
    if (error instanceof AssessmentClientError || error instanceof AssessmentContractError) {
      throw error
    }
    if (error instanceof SyntaxError) {
      throw new AssessmentClientError("assessment_stream_invalid_json")
    }
    throw new AssessmentClientError("assessment_stream_protocol_error")
  }

  if (!state.done || !state.terminal) {
    throw new AssessmentClientError("assessment_stream_protocol_error")
  }
  if (state.terminal.type === "stream_error") {
    throw new AssessmentRemoteError(state.terminal.errorType)
  }
  return state.terminal.final
}

function acceptStreamEvent(
  event: AgentStreamEventV2,
  expected: AssessmentExpectedIdentityV1,
  state: AssessmentStreamState,
): void {
  if (!ASSESSMENT_STREAM_EVENT_TYPES.has(event.type)) protocolError()
  if (event.requestId !== expected.request_id || event.threadId !== expected.thread_id) {
    protocolError()
  }
  if (state.streamId && state.streamId !== event.streamId) protocolError()

  const fingerprint = canonicalJson(event)
  const prior = state.fingerprints.get(event.sequence)
  if (prior !== undefined) {
    if (prior !== fingerprint) protocolError()
    return
  }
  if (event.sequence !== state.highestSequence + 1 || state.done) protocolError()

  state.fingerprints.set(event.sequence, fingerprint)
  state.highestSequence = event.sequence
  state.streamId = event.streamId
  applyUniqueEvent(event, expected, state)
}

function applyUniqueEvent(
  event: AgentStreamEventV2,
  expected: AssessmentExpectedIdentityV1,
  state: AssessmentStreamState,
): void {
  if (event.type === "stream_start") {
    if (state.started || event.sequence !== 1 || state.terminal) {
      protocolError()
    }
    exactKeys(event.data, ["retry_ms"])
    positiveInteger(event.data.retry_ms)
    state.started = true
    return
  }

  if (!state.started) protocolError()
  if (event.type === "assessment_final") {
    if (state.terminal) protocolError()
    state.terminal = {
      type: "assessment_final",
      final: parseAssessmentFinalV1(event.data, expected),
    }
    return
  }
  if (event.type === "stream_error") {
    if (state.terminal) protocolError()
    state.terminal = { type: "stream_error", errorType: parseStreamError(event.data) }
    return
  }

  if (event.type !== "stream_done") protocolError()
  exactKeys(event.data, ["terminal_type"])
  if (!state.terminal || event.data.terminal_type !== state.terminal.type) protocolError()
  state.done = true
}

function parseStreamError(value: Record<string, unknown>): string {
  const allowed = new Set([
    "error_type",
    "message",
    "recoverable",
    "reason",
    "stage",
    "exception_type",
  ])
  const extra = Object.keys(value).find((key) => !allowed.has(key))
  if (extra || !("error_type" in value) || !("message" in value) || !("recoverable" in value)) {
    protocolError()
  }
  const errorType = requiredString(value.error_type, 120)
  if (!ERROR_CODE_PATTERN.test(errorType)) protocolError()
  requiredString(value.message, 500)
  if (value.recoverable !== false) protocolError()
  for (const field of ["reason", "stage", "exception_type"] as const) {
    if (field in value) requiredString(value[field], 240)
  }
  return errorType
}

function guardedReplay(
  reconnect: AgentStreamReplay | undefined,
  submissionSignal?: AbortSignal,
): AgentStreamReplay {
  if (!reconnect) {
    return async () => {
      throw new AssessmentClientError("assessment_stream_incomplete")
    }
  }
  return async (streamId, lastEventId, streamSignal) => {
    try {
      return await reconnect(streamId, lastEventId, streamSignal)
    } catch (error) {
      if (isAbort(error, streamSignal ?? submissionSignal)) throw abortError()
      throw new AssessmentClientError("assessment_stream_transport_failed")
    }
  }
}

function assessmentAttemptEndpoint(apiBaseUrl: string, threadId: string): string {
  if (typeof apiBaseUrl !== "string" || !apiBaseUrl.trim() || apiBaseUrl !== apiBaseUrl.trim()) {
    throw new AssessmentClientError("assessment_client_configuration_invalid")
  }
  let parsed: URL
  try {
    parsed = new URL(apiBaseUrl)
  } catch {
    throw new AssessmentClientError("assessment_client_configuration_invalid")
  }
  if (
    (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new AssessmentClientError("assessment_client_configuration_invalid")
  }
  const base = apiBaseUrl.replace(/\/+$/, "")
  return `${base}/threads/${encodeURIComponent(threadId)}/assessment-attempts`
}

function exactKeys(data: Record<string, unknown>, keys: string[]): void {
  const allowed = new Set(keys)
  if (Object.keys(data).some((key) => !allowed.has(key))) protocolError()
  if (keys.some((key) => !(key in data))) protocolError()
}

function requiredString(value: unknown, maxLength: number): string {
  if (typeof value !== "string" || !value || value.length > maxLength) protocolError()
  return value
}

function positiveInteger(value: unknown): number {
  if (!Number.isInteger(value) || (value as number) < 1) protocolError()
  return value as number
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`
  if (value && typeof value === "object") {
    const data = value as Record<string, unknown>
    return `{${Object.keys(data)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(data[key])}`)
      .join(",")}}`
  }
  const serialized = JSON.stringify(value)
  if (serialized === undefined) protocolError()
  return serialized
}

function protocolError(): never {
  throw new AssessmentClientError("assessment_stream_protocol_error")
}

function isAbort(error: unknown, signal?: AbortSignal): boolean {
  return signal?.aborted === true || (error instanceof DOMException && error.name === "AbortError")
}

function abortError(): DOMException {
  return new DOMException("Aborted", "AbortError")
}
