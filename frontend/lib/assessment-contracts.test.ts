import { describe, expect, it } from "vitest"

import {
  AssessmentContractError,
  assessmentFinalDedupeKey,
  parseAssessmentAttemptV1,
  parseAssessmentFinalV1,
  parseAssessmentSubmissionInput,
  parsePublicExerciseCardV1,
} from "@/lib/assessment-contracts"

const REQUEST_ID = "00000000-0000-4000-8000-000000000301"
const RESOURCE_ID = `resource:v3:${"a".repeat(64)}`
const QUESTION_ID = `question:v1:${"b".repeat(64)}`
const ADAPTIVE_QUESTION_ID = `question:v1:${"c".repeat(64)}`
const PAYLOAD_HASH = `assessment-final:v1:${"d".repeat(64)}`

function attempt(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: "assessment_attempt_v1",
    request_id: REQUEST_ID,
    resource_id: RESOURCE_ID,
    question_id: QUESTION_ID,
    answer: "4",
    time_spent_seconds: 7.5,
    ...overrides,
  }
}

function card(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: "exercise_card_v1",
    question_id: QUESTION_ID,
    question_type: "free_text",
    level: "basic",
    question: "What is 2 + 2?",
    choices: [],
    tags: ["arithmetic"],
    ...overrides,
  }
}

function classification(): Record<string, unknown> {
  return {
    schema_version: "assessment_error_classification_v1",
    error_type: "concept",
    concept_gap: "The addition fact is not stable.",
    suggestion: "Review number composition.",
    confidence: 0.95,
  }
}

function adaptiveTask(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: "adaptive_practice_task_v1",
    question_id: ADAPTIVE_QUESTION_ID,
    task_type: "review",
    question: "What is 1 + 2?",
    answer: "3",
    explanation: "One plus two equals three.",
    reason: "Review a simpler fact.",
    tags: ["arithmetic"],
    difficulty: 0.2,
    ...overrides,
  }
}

function final(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: "assessment_final_v1",
    type: "assessment_final",
    thread_id: "thread-assessment-1",
    request_id: REQUEST_ID,
    resource_id: RESOURCE_ID,
    question_id: QUESTION_ID,
    terminal_status: "correct",
    is_correct: true,
    time_spent_seconds: 7.5,
    error_classification: null,
    adaptive_tasks: [],
    payload_hash: PAYLOAD_HASH,
    ...overrides,
  }
}

const expected = {
  thread_id: "thread-assessment-1",
  request_id: REQUEST_ID,
  resource_id: RESOURCE_ID,
  question_id: QUESTION_ID,
  time_spent_seconds: 7.5,
}

describe("assessment contracts", () => {
  it("preserves the exact submitted answer and request identity", () => {
    const value = parseAssessmentAttemptV1(attempt({ answer: "  Exact Answer  " }))
    expect(value.answer).toBe("  Exact Answer  ")
    expect(value.request_id).toBe(REQUEST_ID)
  })

  it("strictly parses public exercise cards and UI submission input", () => {
    const parsedCard = parsePublicExerciseCardV1(
      card({ question_type: "single_choice", choices: ["3", "4"] }),
    )
    const submission = parseAssessmentSubmissionInput({
      resourceId: RESOURCE_ID,
      question: parsedCard,
      answer: "4",
      timeSpentSeconds: 3,
    })
    expect(submission.question.question_id).toBe(QUESTION_ID)
    expect(submission.resourceId).toBe(RESOURCE_ID)
  })

  it("rejects card drift and inconsistent question types", () => {
    expect(() => parsePublicExerciseCardV1(card({ private_answer: "4" }))).toThrow(
      AssessmentContractError,
    )
    expect(() => parsePublicExerciseCardV1(card({ choices: ["3", "4"] }))).toThrow(
      /free_text/,
    )
    expect(() =>
      parsePublicExerciseCardV1(
        card({ question_type: "single_choice", choices: ["same", "same"] }),
      ),
    ).toThrow(/unique/)
  })

  it("parses and binds a correct final to every request identity field", () => {
    const value = parseAssessmentFinalV1(final(), expected)
    expect(value.terminal_status).toBe("correct")
    expect(assessmentFinalDedupeKey(value)).toBe(PAYLOAD_HASH)
  })

  it("parses a complete incorrect final", () => {
    const value = parseAssessmentFinalV1(
      final({
        terminal_status: "incorrect",
        is_correct: false,
        error_classification: classification(),
        adaptive_tasks: [adaptiveTask()],
      }),
      expected,
    )
    expect(value.error_classification?.error_type).toBe("concept")
    expect(value.adaptive_tasks[0]?.answer).toBe("3")
  })

  it("rejects missing, extra, malformed, and semantically false fields", () => {
    const missing = final()
    delete missing.payload_hash
    expect(() => parseAssessmentFinalV1(missing)).toThrow(/missing field: payload_hash/)
    expect(() => parseAssessmentFinalV1(final({ extra: true }))).toThrow(/unexpected field/)
    expect(() => parseAssessmentFinalV1(final({ payload_hash: "short" }))).toThrow(
      /payload_hash/,
    )
    expect(() => parseAssessmentFinalV1(final({ is_correct: false }))).toThrow(
      /correct status/,
    )
    expect(() =>
      parseAssessmentFinalV1(
        final({ terminal_status: "incorrect", is_correct: false }),
      ),
    ).toThrow(/error_classification/)
  })

  it.each([
    ["thread_id", { thread_id: "other-thread" }],
    ["request_id", { request_id: "00000000-0000-4000-8000-000000000999" }],
    ["resource_id", { resource_id: `resource:v3:${"e".repeat(64)}` }],
    ["question_id", { question_id: `question:v1:${"f".repeat(64)}` }],
    ["time_spent_seconds", { time_spent_seconds: 8 }],
  ])("rejects %s binding drift", (field, overrides) => {
    expect(() => parseAssessmentFinalV1(final(overrides), expected)).toThrow(field)
  })

  it("rejects attempt schema drift without echoing the submitted answer", () => {
    const privateCanary = "submitted-answer-private-canary-814"
    let thrown: unknown
    try {
      parseAssessmentAttemptV1(attempt({ answer: privateCanary, extra: true }))
    } catch (error) {
      thrown = error
    }
    expect(thrown).toBeInstanceOf(AssessmentContractError)
    expect(String(thrown)).not.toContain(privateCanary)
  })
})
