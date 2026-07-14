export const ASSESSMENT_ATTEMPT_SCHEMA_VERSION = "assessment_attempt_v1" as const
export const ASSESSMENT_FINAL_SCHEMA_VERSION = "assessment_final_v1" as const

export type AssessmentTerminalStatus = "correct" | "incorrect"
export type AssessmentErrorType = "concept" | "logic" | "implementation"
export type AdaptivePracticeTaskType = "similar" | "harder" | "review"
export type ExerciseQuestionType = "free_text" | "single_choice"
export type ExerciseLevel = "basic" | "intermediate" | "application" | "self_check"

export interface PublicExerciseCardV1 {
  schema_version: "exercise_card_v1"
  question_id: string
  question_type: ExerciseQuestionType
  level: ExerciseLevel
  question: string
  choices: string[]
  tags: string[]
}

export interface AssessmentSubmissionInput {
  resourceId: string
  question: PublicExerciseCardV1
  answer: string
  timeSpentSeconds: number
}

export interface AssessmentAttemptV1 {
  schema_version: typeof ASSESSMENT_ATTEMPT_SCHEMA_VERSION
  request_id: string
  resource_id: string
  question_id: string
  answer: string
  time_spent_seconds: number
}

export interface AssessmentErrorClassificationV1 {
  schema_version: "assessment_error_classification_v1"
  error_type: AssessmentErrorType
  concept_gap: string
  suggestion: string
  confidence: number
}

export interface AdaptivePracticeTaskV1 {
  schema_version: "adaptive_practice_task_v1"
  question_id: string
  task_type: AdaptivePracticeTaskType
  question: string
  answer: string
  explanation: string
  reason: string
  tags: string[]
  difficulty: number
}

export interface AssessmentFinalV1 {
  schema_version: typeof ASSESSMENT_FINAL_SCHEMA_VERSION
  type: "assessment_final"
  thread_id: string
  request_id: string
  resource_id: string
  question_id: string
  terminal_status: AssessmentTerminalStatus
  is_correct: boolean
  time_spent_seconds: number
  error_classification: AssessmentErrorClassificationV1 | null
  adaptive_tasks: AdaptivePracticeTaskV1[]
  payload_hash: string
}

export interface AssessmentExpectedIdentityV1 {
  thread_id: string
  request_id: string
  resource_id: string
  question_id: string
  time_spent_seconds: number
}

const REQUEST_ID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const THREAD_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9:._/-]{0,159}$/
const RESOURCE_ID_PATTERN = /^resource:v3:[0-9a-f]{64}$/
const QUESTION_ID_PATTERN = /^question:v1:[0-9a-f]{64}$/
const FINAL_HASH_PATTERN = /^assessment-final:v1:[0-9a-f]{64}$/
const ERROR_TYPES = new Set<AssessmentErrorType>(["concept", "logic", "implementation"])
const TASK_TYPES = new Set<AdaptivePracticeTaskType>(["similar", "harder", "review"])
const QUESTION_TYPES = new Set<ExerciseQuestionType>(["free_text", "single_choice"])
const EXERCISE_LEVELS = new Set<ExerciseLevel>([
  "basic",
  "intermediate",
  "application",
  "self_check",
])

export class AssessmentContractError extends Error {
  readonly contract: string

  constructor(contract: string, reason: string) {
    super(`${contract}: ${reason}`)
    this.name = "AssessmentContractError"
    this.contract = contract
  }
}

export function parseAssessmentAttemptV1(value: unknown): AssessmentAttemptV1 {
  const contract = ASSESSMENT_ATTEMPT_SCHEMA_VERSION
  const data = record(value, contract)
  exactKeys(
    data,
    [
      "schema_version",
      "request_id",
      "resource_id",
      "question_id",
      "answer",
      "time_spent_seconds",
    ],
    contract,
  )
  if (data.schema_version !== ASSESSMENT_ATTEMPT_SCHEMA_VERSION) {
    fail(contract, "schema_version must equal assessment_attempt_v1")
  }
  const requestId = patternString(data.request_id, "request_id", REQUEST_ID_PATTERN, 160, contract)
  const resourceId = patternString(
    data.resource_id,
    "resource_id",
    RESOURCE_ID_PATTERN,
    80,
    contract,
  )
  const questionId = patternString(
    data.question_id,
    "question_id",
    QUESTION_ID_PATTERN,
    80,
    contract,
  )
  const answer = boundedString(data.answer, "answer", 10_000, contract)
  if (!answer.trim()) fail(contract, "answer must not be blank")

  return {
    schema_version: ASSESSMENT_ATTEMPT_SCHEMA_VERSION,
    request_id: requestId,
    resource_id: resourceId,
    question_id: questionId,
    answer,
    time_spent_seconds: boundedNumber(
      data.time_spent_seconds,
      "time_spent_seconds",
      0,
      86_400,
      contract,
    ),
  }
}

export function parsePublicExerciseCardV1(value: unknown): PublicExerciseCardV1 {
  const contract = "exercise_card_v1"
  const data = record(value, contract)
  exactKeys(
    data,
    [
      "schema_version",
      "question_id",
      "question_type",
      "level",
      "question",
      "choices",
      "tags",
    ],
    contract,
  )
  if (data.schema_version !== contract) {
    fail(contract, "schema_version must equal exercise_card_v1")
  }
  if (!Array.isArray(data.choices) || data.choices.length > 20) {
    fail(contract, "choices must contain at most 20 items")
  }
  if (!Array.isArray(data.tags) || data.tags.length < 1 || data.tags.length > 80) {
    fail(contract, "tags must contain between 1 and 80 items")
  }
  const choices = data.choices.map((choice, index) =>
    nonBlankUnboundedString(choice, `choices.${index}`, contract),
  )
  const tags = data.tags.map((tag, index) =>
    nonBlankUnboundedString(tag, `tags.${index}`, contract),
  )
  if (new Set(choices).size !== choices.length) fail(contract, "choices must be unique")
  if (new Set(tags).size !== tags.length) fail(contract, "tags must be unique")
  const questionType = enumString(
    data.question_type,
    "question_type",
    QUESTION_TYPES,
    contract,
  )
  if (questionType === "free_text" && choices.length !== 0) {
    fail(contract, "free_text questions must not define choices")
  }
  if (questionType === "single_choice" && choices.length < 2) {
    fail(contract, "single_choice questions require at least two choices")
  }
  return {
    schema_version: "exercise_card_v1",
    question_id: patternString(
      data.question_id,
      "question_id",
      QUESTION_ID_PATTERN,
      80,
      contract,
    ),
    question_type: questionType,
    level: enumString(data.level, "level", EXERCISE_LEVELS, contract),
    question: nonBlankString(data.question, "question", 10_000, contract),
    choices,
    tags,
  }
}

export function parseAssessmentSubmissionInput(
  value: unknown,
): AssessmentSubmissionInput {
  const contract = "assessment_submission_input"
  const data = record(value, contract)
  exactKeys(data, ["resourceId", "question", "answer", "timeSpentSeconds"], contract)
  const answer = boundedString(data.answer, "answer", 10_000, contract)
  if (!answer.trim()) fail(contract, "answer must not be blank")
  return {
    resourceId: patternString(
      data.resourceId,
      "resourceId",
      RESOURCE_ID_PATTERN,
      80,
      contract,
    ),
    question: parsePublicExerciseCardV1(data.question),
    answer,
    timeSpentSeconds: boundedNumber(
      data.timeSpentSeconds,
      "timeSpentSeconds",
      0,
      86_400,
      contract,
    ),
  }
}

export function parseAssessmentExpectedIdentityV1(
  value: unknown,
): AssessmentExpectedIdentityV1 {
  const contract = "assessment_expected_identity_v1"
  const data = record(value, contract)
  exactKeys(
    data,
    ["thread_id", "request_id", "resource_id", "question_id", "time_spent_seconds"],
    contract,
  )
  return {
    thread_id: patternString(data.thread_id, "thread_id", THREAD_ID_PATTERN, 160, contract),
    request_id: patternString(data.request_id, "request_id", REQUEST_ID_PATTERN, 160, contract),
    resource_id: patternString(data.resource_id, "resource_id", RESOURCE_ID_PATTERN, 80, contract),
    question_id: patternString(data.question_id, "question_id", QUESTION_ID_PATTERN, 80, contract),
    time_spent_seconds: boundedNumber(
      data.time_spent_seconds,
      "time_spent_seconds",
      0,
      86_400,
      contract,
    ),
  }
}

export function parseAssessmentFinalV1(
  value: unknown,
  expected?: AssessmentExpectedIdentityV1,
): AssessmentFinalV1 {
  const contract = ASSESSMENT_FINAL_SCHEMA_VERSION
  const data = record(value, contract)
  exactKeys(
    data,
    [
      "schema_version",
      "type",
      "thread_id",
      "request_id",
      "resource_id",
      "question_id",
      "terminal_status",
      "is_correct",
      "time_spent_seconds",
      "error_classification",
      "adaptive_tasks",
      "payload_hash",
    ],
    contract,
  )
  if (data.schema_version !== ASSESSMENT_FINAL_SCHEMA_VERSION) {
    fail(contract, "schema_version must equal assessment_final_v1")
  }
  if (data.type !== "assessment_final") fail(contract, "type must equal assessment_final")

  const terminalStatus = enumString(
    data.terminal_status,
    "terminal_status",
    new Set<AssessmentTerminalStatus>(["correct", "incorrect"]),
    contract,
  )
  if (typeof data.is_correct !== "boolean") fail(contract, "is_correct must be a boolean")
  if (!Array.isArray(data.adaptive_tasks) || data.adaptive_tasks.length > 3) {
    fail(contract, "adaptive_tasks must contain at most 3 items")
  }

  const final: AssessmentFinalV1 = {
    schema_version: ASSESSMENT_FINAL_SCHEMA_VERSION,
    type: "assessment_final",
    thread_id: patternString(data.thread_id, "thread_id", THREAD_ID_PATTERN, 160, contract),
    request_id: patternString(data.request_id, "request_id", REQUEST_ID_PATTERN, 160, contract),
    resource_id: patternString(data.resource_id, "resource_id", RESOURCE_ID_PATTERN, 80, contract),
    question_id: patternString(data.question_id, "question_id", QUESTION_ID_PATTERN, 80, contract),
    terminal_status: terminalStatus,
    is_correct: data.is_correct,
    time_spent_seconds: boundedNumber(
      data.time_spent_seconds,
      "time_spent_seconds",
      0,
      86_400,
      contract,
    ),
    error_classification:
      data.error_classification === null
        ? null
        : parseErrorClassification(data.error_classification),
    adaptive_tasks: data.adaptive_tasks.map((task, index) =>
      parseAdaptiveTask(task, `${contract}.adaptive_tasks.${index}`),
    ),
    payload_hash: patternString(
      data.payload_hash,
      "payload_hash",
      FINAL_HASH_PATTERN,
      84,
      contract,
    ),
  }

  validateTerminalTruth(final)
  if (expected) validateExpectedIdentity(final, parseAssessmentExpectedIdentityV1(expected))
  return final
}

export function assessmentFinalDedupeKey(value: AssessmentFinalV1): string {
  return value.payload_hash
}

function parseErrorClassification(value: unknown): AssessmentErrorClassificationV1 {
  const contract = "assessment_error_classification_v1"
  const data = record(value, contract)
  exactKeys(
    data,
    ["schema_version", "error_type", "concept_gap", "suggestion", "confidence"],
    contract,
  )
  if (data.schema_version !== contract) {
    fail(contract, "schema_version must equal assessment_error_classification_v1")
  }
  return {
    schema_version: contract,
    error_type: enumString(data.error_type, "error_type", ERROR_TYPES, contract),
    concept_gap: nonBlankString(data.concept_gap, "concept_gap", 1_000, contract),
    suggestion: nonBlankString(data.suggestion, "suggestion", 2_000, contract),
    confidence: boundedNumber(data.confidence, "confidence", 0, 1, contract),
  }
}

function parseAdaptiveTask(value: unknown, contract: string): AdaptivePracticeTaskV1 {
  const data = record(value, contract)
  exactKeys(
    data,
    [
      "schema_version",
      "question_id",
      "task_type",
      "question",
      "answer",
      "explanation",
      "reason",
      "tags",
      "difficulty",
    ],
    contract,
  )
  if (data.schema_version !== "adaptive_practice_task_v1") {
    fail(contract, "schema_version must equal adaptive_practice_task_v1")
  }
  if (!Array.isArray(data.tags) || data.tags.length < 1 || data.tags.length > 80) {
    fail(contract, "tags must contain between 1 and 80 items")
  }
  const tags = data.tags.map((tag, index) =>
    canonicalUnboundedString(tag, `tags.${index}`, contract),
  )
  if (new Set(tags).size !== tags.length) fail(contract, "tags must be unique")

  return {
    schema_version: "adaptive_practice_task_v1",
    question_id: patternString(
      data.question_id,
      "question_id",
      QUESTION_ID_PATTERN,
      80,
      contract,
    ),
    task_type: enumString(data.task_type, "task_type", TASK_TYPES, contract),
    question: canonicalString(data.question, "question", 10_000, contract),
    answer: canonicalString(data.answer, "answer", 10_000, contract),
    explanation: canonicalString(data.explanation, "explanation", 10_000, contract),
    reason: canonicalString(data.reason, "reason", 2_000, contract),
    tags,
    difficulty: boundedNumber(data.difficulty, "difficulty", 0, 1, contract),
  }
}

function validateTerminalTruth(value: AssessmentFinalV1): void {
  const contract = ASSESSMENT_FINAL_SCHEMA_VERSION
  if (value.terminal_status === "correct") {
    if (!value.is_correct) fail(contract, "correct status requires is_correct=true")
    if (value.error_classification !== null || value.adaptive_tasks.length !== 0) {
      fail(contract, "correct result cannot include remediation")
    }
    return
  }
  if (value.is_correct) fail(contract, "incorrect status requires is_correct=false")
  if (value.error_classification === null) {
    fail(contract, "incorrect result requires error_classification")
  }
  if (value.adaptive_tasks.length < 1) {
    fail(contract, "incorrect result requires adaptive_tasks")
  }
  const questionIds = value.adaptive_tasks.map((task) => task.question_id)
  if (new Set(questionIds).size !== questionIds.length) {
    fail(contract, "adaptive task question_id values must be unique")
  }
}

function validateExpectedIdentity(
  value: AssessmentFinalV1,
  expected: AssessmentExpectedIdentityV1,
): void {
  const contract = "assessment_final_binding"
  if (value.thread_id !== expected.thread_id) fail(contract, "thread_id does not match")
  if (value.request_id !== expected.request_id) fail(contract, "request_id does not match")
  if (value.resource_id !== expected.resource_id) fail(contract, "resource_id does not match")
  if (value.question_id !== expected.question_id) fail(contract, "question_id does not match")
  if (value.time_spent_seconds !== expected.time_spent_seconds) {
    fail(contract, "time_spent_seconds does not match")
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
  const extra = Object.keys(data)
    .filter((key) => !allowed.has(key))
    .sort()[0]
  if (extra) fail(contract, `unexpected field: ${extra}`)
  const missing = keys.find((key) => !(key in data))
  if (missing) fail(contract, `missing field: ${missing}`)
}

function boundedString(
  value: unknown,
  field: string,
  maxLength: number,
  contract: string,
): string {
  if (typeof value !== "string" || value.length < 1) fail(contract, `${field} is required`)
  if (value.length > maxLength) fail(contract, `${field} exceeds ${maxLength} characters`)
  return value
}

function nonBlankString(
  value: unknown,
  field: string,
  maxLength: number,
  contract: string,
): string {
  const parsed = boundedString(value, field, maxLength, contract)
  if (!parsed.trim()) fail(contract, `${field} must not be blank`)
  return parsed
}

function nonBlankUnboundedString(value: unknown, field: string, contract: string): string {
  if (typeof value !== "string" || !value.trim()) {
    fail(contract, `${field} must not be blank`)
  }
  return value
}

function canonicalString(
  value: unknown,
  field: string,
  maxLength: number,
  contract: string,
): string {
  const parsed = nonBlankString(value, field, maxLength, contract)
  if (parsed !== parsed.trim()) fail(contract, `${field} must be canonical`)
  return parsed
}

function canonicalUnboundedString(value: unknown, field: string, contract: string): string {
  const parsed = nonBlankUnboundedString(value, field, contract)
  if (parsed !== parsed.trim()) fail(contract, `${field} must be canonical`)
  return parsed
}

function patternString(
  value: unknown,
  field: string,
  pattern: RegExp,
  maxLength: number,
  contract: string,
): string {
  const parsed = boundedString(value, field, maxLength, contract)
  if (!pattern.test(parsed)) fail(contract, `${field} is invalid`)
  return parsed
}

function enumString<T extends string>(
  value: unknown,
  field: string,
  choices: ReadonlySet<T>,
  contract: string,
): T {
  if (typeof value !== "string" || !choices.has(value as T)) {
    fail(contract, `${field} is invalid`)
  }
  return value as T
}

function boundedNumber(
  value: unknown,
  field: string,
  minimum: number,
  maximum: number,
  contract: string,
): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(contract, `${field} must be a finite number`)
  }
  if (value < minimum || value > maximum) {
    fail(contract, `${field} must be between ${minimum} and ${maximum}`)
  }
  return value
}

function fail(contract: string, reason: string): never {
  throw new AssessmentContractError(contract, reason)
}
