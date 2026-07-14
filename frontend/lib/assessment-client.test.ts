import { describe, expect, it, vi } from "vitest"

import {
  AssessmentClientError,
  AssessmentHttpError,
  AssessmentRemoteError,
  consumeAssessmentStreamV2,
  submitAssessmentAttempt,
} from "@/lib/assessment-client"
import type {
  AssessmentAttemptV1,
  AssessmentExpectedIdentityV1,
} from "@/lib/assessment-contracts"

const encoder = new TextEncoder()
const REQUEST_ID = "00000000-0000-4000-8000-000000000301"
const THREAD_ID = "thread-assessment-1"
const RESOURCE_ID = `resource:v3:${"a".repeat(64)}`
const QUESTION_ID = `question:v1:${"b".repeat(64)}`
const PAYLOAD_HASH = `assessment-final:v1:${"c".repeat(64)}`

const ATTEMPT: AssessmentAttemptV1 = {
  schema_version: "assessment_attempt_v1",
  request_id: REQUEST_ID,
  resource_id: RESOURCE_ID,
  question_id: QUESTION_ID,
  answer: "submitted-answer-private-canary-814",
  time_spent_seconds: 7.5,
}

const EXPECTED: AssessmentExpectedIdentityV1 = {
  thread_id: THREAD_ID,
  request_id: REQUEST_ID,
  resource_id: RESOURCE_ID,
  question_id: QUESTION_ID,
  time_spent_seconds: 7.5,
}

function final(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: "assessment_final_v1",
    type: "assessment_final",
    thread_id: THREAD_ID,
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

function envelope(
  sequence: number,
  type: "stream_start" | "assessment_final" | "stream_error" | "stream_done",
  data: Record<string, unknown>,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    schema_version: "agent_stream_v2",
    type,
    stream_id: "stream-assessment-1",
    event_id: `stream-assessment-1:${sequence}`,
    sequence,
    request_id: REQUEST_ID,
    thread_id: THREAD_ID,
    created_at: "2026-07-14T00:00:00Z",
    data,
    ...overrides,
  }
}

function frame(
  sequence: number,
  type: "stream_start" | "assessment_final" | "stream_error" | "stream_done",
  data: Record<string, unknown>,
  options: { payload?: Record<string, unknown>; retry?: number; newline?: "\n" | "\r\n" } = {},
): string {
  const newline = options.newline ?? "\n"
  const payload = options.payload ?? envelope(sequence, type, data)
  const lines = [
    `event: ${type}`,
    `id: stream-assessment-1:${sequence}`,
    ...(options.retry === undefined ? [] : [`retry: ${options.retry}`]),
    `data: ${JSON.stringify(payload)}`,
    "",
    "",
  ]
  return lines.join(newline)
}

function completeStream(): string {
  return (
    frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 }) +
    frame(2, "assessment_final", final()) +
    frame(3, "stream_done", { terminal_type: "assessment_final" })
  )
}

function body(value: string): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(value))
      controller.close()
    },
  })
}

function chunkedBody(value: string, boundaries: number[]): ReadableStream<Uint8Array> {
  const bytes = encoder.encode(value)
  return new ReadableStream({
    start(controller) {
      let offset = 0
      for (const boundary of boundaries) {
        controller.enqueue(bytes.slice(offset, boundary))
        offset = boundary
      }
      controller.enqueue(bytes.slice(offset))
      controller.close()
    },
  })
}

function fetchResponse(
  streamBody: ReadableStream<Uint8Array>,
  init: ResponseInit = {},
): typeof fetch {
  return vi.fn(async () =>
    new Response(streamBody, {
      status: 200,
      headers: { "Content-Type": "text/event-stream; charset=utf-8" },
      ...init,
    }),
  ) as unknown as typeof fetch
}

describe("assessment SSE client", () => {
  it("submits the exact request_id once and returns one authoritative final", async () => {
    const fetchImpl = fetchResponse(body(completeStream()))
    const reconnect = vi.fn()
    const result = await submitAssessmentAttempt({
      apiBaseUrl: "https://api.example.test",
      threadId: THREAD_ID,
      attempt: ATTEMPT,
      fetchImpl,
      reconnect,
    })

    expect(result.payload_hash).toBe(PAYLOAD_HASH)
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    expect(reconnect).not.toHaveBeenCalled()
    const [url, init] = vi.mocked(fetchImpl).mock.calls[0] ?? []
    expect(url).toBe(
      `https://api.example.test/threads/${encodeURIComponent(THREAD_ID)}/assessment-attempts`,
    )
    expect(init?.method).toBe("POST")
    expect(JSON.parse(String(init?.body))).toEqual(ATTEMPT)
    expect(JSON.parse(String(init?.body)).request_id).toBe(REQUEST_ID)
  })

  it("uses the shared parser for CRLF and UTF-8 chunks", async () => {
    const incorrect = final({
      terminal_status: "incorrect",
      is_correct: false,
      error_classification: {
        schema_version: "assessment_error_classification_v1",
        error_type: "concept",
        concept_gap: "概念缺口",
        suggestion: "复习加法",
        confidence: 0.9,
      },
      adaptive_tasks: [
        {
          schema_version: "adaptive_practice_task_v1",
          question_id: `question:v1:${"d".repeat(64)}`,
          task_type: "review",
          question: "一加二等于几？",
          answer: "3",
          explanation: "一加二等于三。",
          reason: "先复习简单事实。",
          tags: ["加法"],
          difficulty: 0.2,
        },
      ],
      payload_hash: `assessment-final:v1:${"e".repeat(64)}`,
    })
    const stream =
      frame(1, "stream_start", { retry_ms: 1 }, { retry: 1, newline: "\r\n" }) +
      frame(2, "assessment_final", incorrect, { newline: "\r\n" }) +
      frame(3, "stream_done", { terminal_type: "assessment_final" }, { newline: "\r\n" })
    const result = await consumeAssessmentStreamV2({
      initialBody: chunkedBody(stream, [1, 7, 31, 119, 227]),
      expected: EXPECTED,
    })
    expect(result.adaptive_tasks[0]?.question).toBe("一加二等于几？")
  })

  it("deduplicates an identical repeated sequence", async () => {
    const finalFrame = frame(2, "assessment_final", final())
    const result = await consumeAssessmentStreamV2({
      initialBody: body(
        frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 }) +
          finalFrame +
          finalFrame +
          frame(3, "stream_done", { terminal_type: "assessment_final" }),
      ),
      expected: EXPECTED,
    })
    expect(result.payload_hash).toBe(PAYLOAD_HASH)
  })

  it("rejects conflicting duplicate sequences and sequence gaps", async () => {
    const conflict =
      frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 }) +
      frame(2, "assessment_final", final()) +
      frame(2, "assessment_final", final({ payload_hash: `assessment-final:v1:${"f".repeat(64)}` }))
    await expect(
      consumeAssessmentStreamV2({ initialBody: body(conflict), expected: EXPECTED }),
    ).rejects.toMatchObject({ code: "assessment_stream_protocol_error" })

    const gap =
      frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 }) +
      frame(3, "assessment_final", final(), {
        payload: envelope(3, "assessment_final", final()),
      })
    await expect(
      consumeAssessmentStreamV2({ initialBody: body(gap), expected: EXPECTED }),
    ).rejects.toMatchObject({ code: "assessment_stream_protocol_error" })
  })

  it.each([
    ["envelope thread", envelope(2, "assessment_final", final(), { thread_id: "other" })],
    [
      "envelope request",
      envelope(2, "assessment_final", final(), {
        request_id: "00000000-0000-4000-8000-000000000999",
      }),
    ],
    ["final resource", envelope(2, "assessment_final", final({ resource_id: `resource:v3:${"8".repeat(64)}` }))],
    ["final question", envelope(2, "assessment_final", final({ question_id: `question:v1:${"9".repeat(64)}` }))],
    ["final hash", envelope(2, "assessment_final", final({ payload_hash: "invalid" }))],
  ])("rejects %s identity or hash drift", async (_label, payload) => {
    const value =
      frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 }) +
      frame(2, "assessment_final", final(), { payload })
    await expect(
      consumeAssessmentStreamV2({ initialBody: body(value), expected: EXPECTED }),
    ).rejects.toBeInstanceOf(Error)
  })

  it("requires assessment_final followed by a matching stream_done", async () => {
    const mismatch =
      frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 }) +
      frame(2, "assessment_final", final()) +
      frame(3, "stream_done", { terminal_type: "stream_error" })
    await expect(
      consumeAssessmentStreamV2({ initialBody: body(mismatch), expected: EXPECTED }),
    ).rejects.toMatchObject({ code: "assessment_stream_protocol_error" })

    const twoTerminals =
      frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 }) +
      frame(2, "assessment_final", final()) +
      frame(3, "stream_error", {
        error_type: "assessment_attempt_failed",
        message: "Assessment attempt failed",
        recoverable: false,
      })
    await expect(
      consumeAssessmentStreamV2({ initialBody: body(twoTerminals), expected: EXPECTED }),
    ).rejects.toMatchObject({ code: "assessment_stream_protocol_error" })
  })

  it("does not resubmit or reconnect an incomplete stream implicitly", async () => {
    const fetchImpl = fetchResponse(
      body(frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 })),
    )
    await expect(
      submitAssessmentAttempt({
        apiBaseUrl: "https://api.example.test",
        threadId: THREAD_ID,
        attempt: ATTEMPT,
        fetchImpl,
      }),
    ).rejects.toMatchObject({ code: "assessment_stream_incomplete" })
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it("replays only when an explicit replay callback is supplied", async () => {
    vi.useFakeTimers()
    try {
      const reconnect = vi.fn(async () =>
        body(
          frame(2, "assessment_final", final()) +
            frame(3, "stream_done", { terminal_type: "assessment_final" }),
        ),
      )
      const pending = consumeAssessmentStreamV2({
        initialBody: body(frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 })),
        expected: EXPECTED,
        reconnect,
      })
      await vi.runAllTimersAsync()
      await expect(pending).resolves.toMatchObject({ payload_hash: PAYLOAD_HASH })
      expect(reconnect).toHaveBeenCalledWith(
        "stream-assessment-1",
        "stream-assessment-1:1",
        undefined,
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not expose a submitted answer through remote or HTTP errors", async () => {
    const privateAnswer = ATTEMPT.answer
    const remote =
      frame(1, "stream_start", { retry_ms: 1 }, { retry: 1 }) +
      frame(2, "stream_error", {
        error_type: "assessment_attempt_failed",
        message: privateAnswer,
        recoverable: false,
      }) +
      frame(3, "stream_done", { terminal_type: "stream_error" })
    let remoteError: unknown
    try {
      await consumeAssessmentStreamV2({ initialBody: body(remote), expected: EXPECTED })
    } catch (error) {
      remoteError = error
    }
    expect(remoteError).toBeInstanceOf(AssessmentRemoteError)
    expect(String(remoteError)).not.toContain(privateAnswer)

    const httpFetch = vi.fn(async () =>
      new Response(privateAnswer, { status: 409, headers: { "Content-Type": "text/plain" } }),
    ) as unknown as typeof fetch
    let httpError: unknown
    try {
      await submitAssessmentAttempt({
        apiBaseUrl: "https://api.example.test",
        threadId: THREAD_ID,
        attempt: ATTEMPT,
        fetchImpl: httpFetch,
      })
    } catch (error) {
      httpError = error
    }
    expect(httpError).toBeInstanceOf(AssessmentHttpError)
    expect(String(httpError)).not.toContain(privateAnswer)
  })

  it("rejects invalid response media type without parsing the body", async () => {
    const fetchImpl = fetchResponse(body(completeStream()), {
      headers: { "Content-Type": "application/json" },
    })
    await expect(
      submitAssessmentAttempt({
        apiBaseUrl: "https://api.example.test",
        threadId: THREAD_ID,
        attempt: ATTEMPT,
        fetchImpl,
      }),
    ).rejects.toEqual(
      expect.objectContaining<Partial<AssessmentClientError>>({
        code: "assessment_response_content_type_invalid",
      }),
    )
  })
})
