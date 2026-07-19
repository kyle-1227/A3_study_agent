import { describe, expect, it } from "vitest"

import type { Message } from "@/components/chat-area"
import type { ResourceFinalExerciseItem } from "@/lib/resource-final"
import {
  mergeResourceFinalIntoMessage,
  parseResourceFinalEvent,
  resourceFinalDedupeKey,
  resourceFinalOutcome,
} from "@/lib/resource-final"

const RESOURCE_FINAL_ID = `resource-final:v3:${"a".repeat(64)}`
const FINAL_PAYLOAD_HASH = `payload:v3:${"b".repeat(64)}`

function resourceId(seed: string): string {
  return `resource:v3:${seed.repeat(64)}`
}

function payloadHash(seed: string): string {
  return `payload:v3:${seed.repeat(64)}`
}

function resourceValidation(
  resourceType: "mindmap" | "quiz" | "review_doc",
  terminalStatus: "success" | "partial_success" = "success",
  warnings: string[] = [],
): Record<string, unknown> {
  return {
    schema_version: "resource_validation_v1",
    resource_type: resourceType,
    valid: true,
    terminal_status: terminalStatus,
    renderable_count: 1,
    downloadable_count: 1,
    verified_local_count: 1,
    remote_unverified_count: 0,
    failure_reason: "",
    warnings,
  }
}

function publicExerciseCard(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    schema_version: "exercise_card_v1",
    question_id: `question:v1:${"1".repeat(64)}`,
    question_type: "free_text",
    level: "basic",
    question: "What does a Python list store?",
    choices: [],
    tags: ["python", "collections"],
    ...overrides,
  }
}

function quizResource(
  exerciseItems: unknown[] = [publicExerciseCard()],
  artifactItems: unknown[] = exerciseItems,
): Record<string, unknown> {
  return {
    kind: "quiz",
    status: "success",
    resource_id: resourceId("2"),
    payload_hash: payloadHash("3"),
    title: "Python Quiz",
    summary: "Validated public exercise cards are ready",
    payload: {
      exercise_artifact: {
        schema_version: "exercise_public_artifact_v1",
        title: "Python Quiz",
        items: artifactItems,
      },
      exercise_items: exerciseItems,
    },
    artifact_refs: { markdown_url: "/artifacts/python-quiz.md" },
    validation: resourceValidation("quiz"),
  }
}

function mindmapResource(
  terminalStatus: "success" | "partial_success" = "success",
  warnings: string[] = [],
): Record<string, unknown> {
  return {
    kind: "mindmap",
    status: terminalStatus,
    resource_id: resourceId("c"),
    payload_hash: payloadHash("d"),
    title: "Machine learning map",
    summary: "Mindmap ready",
    payload: {
      mindmap: {
        title: "Machine learning map",
        tree: {
          title: "Machine learning",
          children: [{ title: "Supervised learning" }],
        },
        xmind_url: "/artifacts/map.xmind",
      },
    },
    artifact_refs: { xmind_url: "/artifacts/map.xmind" },
    validation: resourceValidation("mindmap", terminalStatus, warnings),
  }
}

function reviewDocumentResource(): Record<string, unknown> {
  return {
    kind: "review_doc",
    status: "success",
    resource_id: resourceId("e"),
    payload_hash: payloadHash("f"),
    title: "Review notes",
    summary: "Review document ready",
    payload: {
      review_doc: {
        title: "Review notes",
        markdown: "# Review notes",
        markdown_url: "/artifacts/review.md",
      },
      review_doc_artifacts: [],
    },
    artifact_refs: { markdown_url: "/artifacts/review.md" },
    validation: resourceValidation("review_doc"),
  }
}

function finalValidation(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: "resource_final_validation_v3",
    resource_count: 1,
    success_count: 1,
    partial_success_count: 0,
    failed_count: 0,
    blocked_count: 0,
    renderable_count: 1,
    downloadable_count: 1,
    ...overrides,
  }
}

function rawEvent(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    type: "resource_final",
    schema_version: "resource_final_v3",
    resource_final_id: RESOURCE_FINAL_ID,
    payload_hash: FINAL_PAYLOAD_HASH,
    thread_id: "thread-1",
    request_id: "request-1",
    terminal_status: "success",
    resources: [mindmapResource()],
    recommendations: [],
    blocked_resources: [],
    errors: [],
    validation: finalValidation(),
    summary: "Resource bundle ready",
    ...overrides,
  }
}

describe("Resource Final V3 contract helpers", () => {
  it("strictly parses a single-resource success event", () => {
    const event = parseResourceFinalEvent(rawEvent())
    expect(event.schema_version).toBe("resource_final_v3")
    expect(event.terminal_status).toBe("success")
    expect(event.validation.renderableCount).toBe(1)
    expect(event.resources[0]?.kind).toBe("mindmap")
  })

  it("strictly parses and preserves exact public exercise cards", () => {
    const cards = [
      publicExerciseCard(),
      publicExerciseCard({
        question_id: `question:v1:${"2".repeat(64)}`,
        question_type: "single_choice",
        level: "application",
        question: "Which value is mutable?",
        choices: ["tuple", "list"],
        tags: ["python", "mutability"],
      }),
    ]
    const event = parseResourceFinalEvent(
      rawEvent({ resources: [quizResource(cards)] }),
    )
    const parsedCards = event.resources[0]?.payload.exercise_items
    const firstCard: ResourceFinalExerciseItem | undefined = parsedCards?.[0]

    expect(parsedCards).toEqual(cards)
    expect(event.resources[0]?.payload.exercise_artifact?.items).toEqual(cards)
    expect(firstCard?.schema_version).toBe("exercise_card_v1")
    expect(firstCard?.choices).toEqual([])

    const merged = mergeResourceFinalIntoMessage(
      {
        id: "assistant-quiz",
        role: "assistant",
        content: "",
        threadId: "thread-1",
        requestId: "request-1",
      },
      event,
      "http://api.test",
    )
    expect(merged.exercise?.resourceId).toBe(resourceId("2"))
    expect(merged.exercise?.questions).toEqual(cards)
  })

  it.each([
    ["answer", "server-only answer"],
    ["answer_key", { accepted_answers: ["server-only answer"] }],
    ["accepted_answers", ["server-only answer"]],
    ["explanation", "server-only explanation"],
    ["pitfall", "server-only pitfall"],
    ["answer_explanation", "server-only explanation"],
    ["match_mode", "exact"],
    ["legacy_field", "unsupported"],
  ])("rejects private or unknown exercise-card field %s", (field, privateValue) => {
    const leakedCard = publicExerciseCard({ [field]: privateValue })

    expect(() =>
      parseResourceFinalEvent(
        rawEvent({ resources: [quizResource([leakedCard])] }),
      ),
    ).toThrow(new RegExp(`unexpected field: ${field}`))
  })

  it("rejects private fields copied only into exercise_artifact items", () => {
    const publicCard = publicExerciseCard()
    const leakedArtifactCard = {
      ...publicCard,
      answer_key: { accepted_answers: ["server-only answer"] },
    }

    expect(() =>
      parseResourceFinalEvent(
        rawEvent({
          resources: [quizResource([publicCard], [leakedArtifactCard])],
        }),
      ),
    ).toThrow(/unexpected field: answer_key/)
  })

  it("rejects invalid public exercise-card structure and business rules", () => {
    const missingTags = publicExerciseCard()
    delete missingTags.tags
    expect(() =>
      parseResourceFinalEvent(
        rawEvent({ resources: [quizResource([missingTags])] }),
      ),
    ).toThrow(/tags must be an array/)

    const freeTextWithChoices = publicExerciseCard({ choices: ["A", "B"] })
    expect(() =>
      parseResourceFinalEvent(
        rawEvent({ resources: [quizResource([freeTextWithChoices])] }),
      ),
    ).toThrow(/free_text questions must not define choices/)

    const duplicate = publicExerciseCard()
    expect(() =>
      parseResourceFinalEvent(
        rawEvent({ resources: [quizResource([duplicate, { ...duplicate }])] }),
      ),
    ).toThrow(/question_id values must be unique/)
  })

  it("merges every resource in a multi-resource event", () => {
    const event = parseResourceFinalEvent(
      rawEvent({
        resources: [mindmapResource(), reviewDocumentResource()],
        validation: finalValidation({
          resource_count: 2,
          success_count: 2,
          renderable_count: 2,
          downloadable_count: 2,
        }),
      }),
    )
    const message: Message = {
      id: "assistant-1",
      role: "assistant",
      content: "",
      threadId: "thread-1",
      requestId: "request-1",
    }
    const restored = mergeResourceFinalIntoMessage(message, event, "http://api.test")
    expect(restored.content).toBe("Resource bundle ready")
    expect(restored.mindmap?.xmindUrl).toBe("http://api.test/artifacts/map.xmind")
    expect(restored.mindmap?.tree.children?.[0].title).toBe("Supervised learning")
    expect(restored.reviewDoc?.markdown).toBe("# Review notes")
  })

  it("maps valid partial, failed, and controlled-stop terminal truth", () => {
    const partial = parseResourceFinalEvent(
      rawEvent({
        terminal_status: "partial_success",
        resources: [mindmapResource("partial_success")],
        validation: finalValidation({ success_count: 0, partial_success_count: 1 }),
      }),
    )
    const failed = parseResourceFinalEvent(
      rawEvent({
        terminal_status: "failed",
        resources: [],
        errors: [
          {
            resource_type: "mindmap",
            error_code: "mindmap.provider_error",
            error_type: "ProviderError",
            message_sanitized: "Provider request failed",
          },
        ],
        validation: finalValidation({
          resource_count: 0,
          success_count: 0,
          failed_count: 1,
          renderable_count: 0,
          downloadable_count: 0,
        }),
      }),
    )
    const controlledStop = parseResourceFinalEvent(
      rawEvent({
        terminal_status: "controlled_stop",
        resources: [],
        blocked_resources: [
          {
            resource_type: "mindmap",
            status: "blocked_insufficient_evidence",
            reason_code: "evidence.missing_parent",
            blocked_requirement_ids: ["requirement-1"],
          },
        ],
        validation: finalValidation({
          resource_count: 0,
          success_count: 0,
          blocked_count: 1,
          renderable_count: 0,
          downloadable_count: 0,
        }),
      }),
    )
    expect(resourceFinalOutcome(partial)?.state).toBe("partial_success")
    expect(resourceFinalOutcome(failed)?.state).toBe("failed")
    expect(resourceFinalOutcome(controlledStop)?.state).toBe("controlled_stop")
  })

  it("shows a friendly scope notice only for evidence-limited partial resources", () => {
    const event = parseResourceFinalEvent(
      rawEvent({
        terminal_status: "partial_success",
        resources: [mindmapResource("partial_success", ["evidence_scope_limited"])],
        validation: finalValidation({ success_count: 0, partial_success_count: 1 }),
      }),
    )
    const merged = mergeResourceFinalIntoMessage(
      {
        id: "assistant-scope",
        role: "assistant",
        content: "",
        threadId: "thread-1",
        requestId: "request-1",
      },
      event,
      "http://api.test",
    )

    expect(merged.resourceScopeNotice).toBe(true)
  })

  it("rejects count tampering and inconsistent terminal truth", () => {
    expect(() =>
      parseResourceFinalEvent(
        rawEvent({ validation: finalValidation({ resource_count: 2 }) }),
      ),
    ).toThrow(/resourceCount/)
    expect(() =>
      parseResourceFinalEvent(rawEvent({ terminal_status: "failed" })),
    ).toThrow(/failed terminal truth/)
  })

  it("rejects invalid ids, hashes, extra fields, and empty payloads", () => {
    expect(() =>
      parseResourceFinalEvent(rawEvent({ resource_final_id: "resource-final:v3:short" })),
    ).toThrow(/resource_final_id/)
    expect(() =>
      parseResourceFinalEvent(rawEvent({ payload_hash: "payload:v3:short" })),
    ).toThrow(/payload_hash/)
    expect(() => parseResourceFinalEvent(rawEvent({ legacy_resource: {} }))).toThrow(
      /unexpected field/,
    )
    expect(() =>
      parseResourceFinalEvent(
        rawEvent({ resources: [{ ...mindmapResource(), payload: {} }] }),
      ),
    ).toThrow(/payload has no renderable value/)
  })

  it("rejects duplicate recommendation identity and rank", () => {
    const recommendation = {
      recommendation_id: "recommendation-1",
      resource_id: mindmapResource().resource_id,
      resource_type: "mindmap",
      trigger: "automatic",
      rank: 1,
      title: "Review this map",
      reason: "Matches the current learning goal",
    }
    expect(() =>
      parseResourceFinalEvent(
        rawEvent({ recommendations: [recommendation, { ...recommendation }] }),
      ),
    ).toThrow(/recommendation_id/)
  })

  it("rejects an automatic recommendation outside generated resources", () => {
    const recommendation = {
      recommendation_id: "recommendation-1",
      resource_id: "resource-not-generated",
      resource_type: "mindmap",
      trigger: "automatic",
      rank: 1,
      title: "Review this map",
      reason: "Matches the current learning goal",
    }
    expect(() =>
      parseResourceFinalEvent(rawEvent({ recommendations: [recommendation] })),
    ).toThrow(/must target a generated resource/)
  })

  it("deduplicates by Resource Final V3 identity", () => {
    const event = parseResourceFinalEvent(rawEvent())
    expect(resourceFinalDedupeKey(event)).toBe(
      `resource_final_id:${RESOURCE_FINAL_ID}`,
    )
    expect(
      resourceFinalDedupeKey({
        thread_id: "thread-1",
        request_id: "request-2",
        payload_hash: payloadHash("1"),
      }),
    ).not.toBe(
      resourceFinalDedupeKey({
        thread_id: "thread-1",
        request_id: "request-1",
        payload_hash: FINAL_PAYLOAD_HASH,
      }),
    )
  })

  it("rejects binding a final event to another request or thread", () => {
    const event = parseResourceFinalEvent(rawEvent())
    expect(() =>
      mergeResourceFinalIntoMessage(
        {
          id: "assistant-1",
          role: "assistant",
          content: "",
          threadId: "thread-other",
          requestId: "request-1",
        },
        event,
        "http://api.test",
      ),
    ).toThrow(/thread_id/)
    expect(() =>
      mergeResourceFinalIntoMessage(
        {
          id: "assistant-1",
          role: "assistant",
          content: "",
          threadId: "thread-1",
          requestId: "request-other",
        },
        event,
        "http://api.test",
      ),
    ).toThrow(/request_id/)
  })
})
