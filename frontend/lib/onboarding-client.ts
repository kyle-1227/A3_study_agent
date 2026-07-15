import {
  OnboardingContractError,
  parseLearningGuidanceCatalogV1,
  parseOnboardRequestV2,
  parseOnboardResultV2,
  type LearningGuidanceCatalogV1,
  type OnboardRequestV2,
  type OnboardResultV2,
} from "@/lib/onboarding-contracts"

export const ONBOARDING_ATTEMPT_STORAGE_PREFIX = "a3_onboarding_attempt_v2:"
const ATTEMPT_SCHEMA_VERSION = "onboarding_submission_attempt_v2" as const
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const USER_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9:._/-]{0,159}$/
const SAFE_REMOTE_CODES = new Set([
  "learning_guidance_profile_writer_unavailable",
  "learning_guidance_runtime_unavailable",
  "profile_write_identity_invalid",
  "profile_write_topic_invalid",
  "profile_write_profile_mismatch",
  "profile_write_request_conflict",
  "profile_write_reserved_metadata_invalid",
  "profile_write_binding_invalid",
  "profile_write_binding_missing",
  "profile_write_goal_identity_invalid",
])

export interface OnboardingSubmissionAttemptV2 {
  readonly schema_version: typeof ATTEMPT_SCHEMA_VERSION
  readonly user_id: string
  readonly request_id: string
  readonly payload: OnboardRequestV2 | null
}

export interface KeyValueStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export type OnboardingClientErrorCode =
  | "onboarding_client_configuration_invalid"
  | "onboarding_transport_failed"
  | "onboarding_request_aborted"
  | "onboarding_http_failed"
  | "onboarding_response_content_type_invalid"
  | "onboarding_response_json_invalid"

export class OnboardingClientError extends Error {
  readonly code: OnboardingClientErrorCode
  readonly status: number | null
  readonly remoteCode: string | null

  constructor(
    code: OnboardingClientErrorCode,
    options: { status?: number; remoteCode?: string } = {},
  ) {
    super(code)
    this.name = "OnboardingClientError"
    this.code = code
    this.status = options.status ?? null
    this.remoteCode = options.remoteCode ?? null
  }
}

export type OnboardingAttemptErrorCode =
  | "onboarding_attempt_identity_invalid"
  | "onboarding_attempt_storage_read_failed"
  | "onboarding_attempt_storage_write_failed"
  | "onboarding_attempt_storage_remove_failed"
  | "onboarding_attempt_record_invalid"
  | "onboarding_attempt_payload_conflict"

export class OnboardingAttemptError extends Error {
  readonly code: OnboardingAttemptErrorCode

  constructor(code: OnboardingAttemptErrorCode) {
    super(code)
    this.name = "OnboardingAttemptError"
    this.code = code
  }
}

export async function fetchLearningGuidanceCatalog(options: {
  apiBaseUrl: string
  fetchImpl: typeof fetch
  signal?: AbortSignal
}): Promise<LearningGuidanceCatalogV1> {
  const apiBaseUrl = validatedApiBaseUrl(options.apiBaseUrl)
  const response = await executeRequest(
    options.fetchImpl,
    `${apiBaseUrl}/learning-guidance/catalog`,
    {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: options.signal,
    },
    options.signal,
  )
  await requireSuccess(response)
  return parseLearningGuidanceCatalogV1(await readJson(response))
}

export async function submitOnboardingV2(options: {
  apiBaseUrl: string
  catalog: LearningGuidanceCatalogV1
  request: OnboardRequestV2
  fetchImpl: typeof fetch
  signal?: AbortSignal
}): Promise<OnboardResultV2> {
  const apiBaseUrl = validatedApiBaseUrl(options.apiBaseUrl)
  const request = parseOnboardRequestV2(options.request, options.catalog)
  const response = await executeRequest(
    options.fetchImpl,
    `${apiBaseUrl}/onboard`,
    {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
      signal: options.signal,
    },
    options.signal,
  )
  await requireSuccess(response)
  return parseOnboardResultV2(await readJson(response), {
    requestId: request.profile.request_id,
    userId: request.profile.user_id,
    skillsCount: request.profile.skills.length,
    goalsCount: request.profile.goals.length,
    preferencesCount: request.profile.preferences.length,
  })
}

export function getOrCreateOnboardingAttempt(options: {
  storage: KeyValueStorage
  userId: string
  uuidFactory: () => string
}): OnboardingSubmissionAttemptV2 {
  const userId = validatedUserId(options.userId)
  const key = onboardingAttemptStorageKey(userId)
  let raw: string | null
  try {
    raw = options.storage.getItem(key)
  } catch {
    throw new OnboardingAttemptError("onboarding_attempt_storage_read_failed")
  }
  if (raw !== null) {
    const existing = parseAttemptRecord(raw)
    if (existing.user_id !== userId) {
      throw new OnboardingAttemptError("onboarding_attempt_identity_invalid")
    }
    return existing
  }

  const attempt = parseAttemptValue({
    schema_version: ATTEMPT_SCHEMA_VERSION,
    user_id: userId,
    request_id: options.uuidFactory(),
    payload: null,
  })
  persistAttempt(options.storage, key, attempt)
  return attempt
}

export function freezeOnboardingAttempt(options: {
  storage: KeyValueStorage
  userId: string
  catalog: LearningGuidanceCatalogV1
  request: OnboardRequestV2
}): OnboardRequestV2 {
  const userId = validatedUserId(options.userId)
  const request = parseOnboardRequestV2(options.request, options.catalog)
  const key = onboardingAttemptStorageKey(userId)
  let raw: string | null
  try {
    raw = options.storage.getItem(key)
  } catch {
    throw new OnboardingAttemptError("onboarding_attempt_storage_read_failed")
  }
  if (raw === null) {
    throw new OnboardingAttemptError("onboarding_attempt_record_invalid")
  }
  const attempt = parseAttemptRecord(raw)
  if (
    attempt.user_id !== userId ||
    request.profile.user_id !== userId ||
    request.profile.request_id !== attempt.request_id
  ) {
    throw new OnboardingAttemptError("onboarding_attempt_identity_invalid")
  }
  if (attempt.payload !== null) {
    const stored = parseOnboardRequestV2(attempt.payload, options.catalog)
    if (canonicalJson(stored) !== canonicalJson(request)) {
      throw new OnboardingAttemptError("onboarding_attempt_payload_conflict")
    }
    return stored
  }

  const frozenAttempt = parseAttemptValue({ ...attempt, payload: request })
  persistAttempt(options.storage, key, frozenAttempt)
  if (frozenAttempt.payload === null) {
    throw new OnboardingAttemptError("onboarding_attempt_record_invalid")
  }
  return parseOnboardRequestV2(frozenAttempt.payload, options.catalog)
}

export function removeOnboardingAttempt(storage: KeyValueStorage, userId: string): void {
  const key = onboardingAttemptStorageKey(validatedUserId(userId))
  try {
    storage.removeItem(key)
  } catch {
    throw new OnboardingAttemptError("onboarding_attempt_storage_remove_failed")
  }
}

function onboardingAttemptStorageKey(userId: string): string {
  return `${ONBOARDING_ATTEMPT_STORAGE_PREFIX}${userId}`
}

function parseAttemptRecord(raw: string): OnboardingSubmissionAttemptV2 {
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    throw new OnboardingAttemptError("onboarding_attempt_record_invalid")
  }
  return parseAttemptValue(value)
}

function parseAttemptValue(value: unknown): OnboardingSubmissionAttemptV2 {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new OnboardingAttemptError("onboarding_attempt_record_invalid")
  }
  const data = value as Record<string, unknown>
  const actual = Object.keys(data).sort()
  const expected = ["schema_version", "user_id", "request_id", "payload"].sort()
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new OnboardingAttemptError("onboarding_attempt_record_invalid")
  }
  if (
    data.schema_version !== ATTEMPT_SCHEMA_VERSION ||
    typeof data.user_id !== "string" ||
    !USER_ID_PATTERN.test(data.user_id) ||
    typeof data.request_id !== "string" ||
    !UUID_PATTERN.test(data.request_id)
  ) {
    throw new OnboardingAttemptError("onboarding_attempt_record_invalid")
  }
  let payload: OnboardRequestV2 | null = null
  if (data.payload !== null) {
    try {
      payload = parseOnboardRequestV2(data.payload)
    } catch (error) {
      if (error instanceof OnboardingContractError) {
        throw new OnboardingAttemptError("onboarding_attempt_record_invalid")
      }
      throw error
    }
    if (
      payload.profile.user_id !== data.user_id ||
      payload.profile.request_id !== data.request_id
    ) {
      throw new OnboardingAttemptError("onboarding_attempt_record_invalid")
    }
  }
  return Object.freeze({
    schema_version: ATTEMPT_SCHEMA_VERSION,
    user_id: data.user_id,
    request_id: data.request_id,
    payload,
  })
}

function persistAttempt(
  storage: KeyValueStorage,
  key: string,
  attempt: OnboardingSubmissionAttemptV2,
): void {
  try {
    storage.setItem(key, canonicalJson(attempt))
  } catch {
    throw new OnboardingAttemptError("onboarding_attempt_storage_write_failed")
  }
}

function validatedUserId(value: string): string {
  if (typeof value !== "string" || value !== value.trim() || !USER_ID_PATTERN.test(value)) {
    throw new OnboardingAttemptError("onboarding_attempt_identity_invalid")
  }
  return value
}

function validatedApiBaseUrl(value: string): string {
  if (typeof value !== "string" || value !== value.trim() || !value || value.endsWith("/")) {
    throw new OnboardingClientError("onboarding_client_configuration_invalid")
  }
  let parsed: URL
  try {
    parsed = new URL(value)
  } catch {
    throw new OnboardingClientError("onboarding_client_configuration_invalid")
  }
  if (
    (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new OnboardingClientError("onboarding_client_configuration_invalid")
  }
  return value
}

async function executeRequest(
  fetchImpl: typeof fetch,
  endpoint: string,
  init: RequestInit,
  signal?: AbortSignal,
): Promise<Response> {
  try {
    return await fetchImpl(endpoint, init)
  } catch {
    if (signal?.aborted) {
      throw new OnboardingClientError("onboarding_request_aborted")
    }
    throw new OnboardingClientError("onboarding_transport_failed")
  }
}

async function requireSuccess(response: Response): Promise<void> {
  if (response.ok) return
  throw new OnboardingClientError("onboarding_http_failed", {
    status: response.status,
    remoteCode: await safeRemoteCode(response),
  })
}

async function safeRemoteCode(response: Response): Promise<string | undefined> {
  const contentType = response.headers.get("Content-Type")?.toLowerCase() ?? ""
  if (!contentType.includes("application/json")) return undefined
  try {
    const value: unknown = await response.json()
    if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined
    const detail = (value as Record<string, unknown>).detail
    return typeof detail === "string" && SAFE_REMOTE_CODES.has(detail) ? detail : undefined
  } catch {
    return undefined
  }
}

async function readJson(response: Response): Promise<unknown> {
  const contentType = response.headers.get("Content-Type")?.toLowerCase() ?? ""
  if (!contentType.includes("application/json")) {
    throw new OnboardingClientError("onboarding_response_content_type_invalid")
  }
  try {
    return await response.json()
  } catch {
    throw new OnboardingClientError("onboarding_response_json_invalid")
  }
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(value)
}
