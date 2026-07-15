import { describe, expect, it } from "vitest"

import {
  attachRecommendationFinalToMessages,
  parseRecommendationFinalV1,
  recommendationFinalDedupeKey,
  type RecommendationFinalMessage,
} from "@/lib/recommendation-final"
import {
  availableRecommendationFinalWire,
  unavailableRecommendationFinalWire,
} from "@/test/recommendation-final-fixtures"

describe("recommendation final contract", () => {
  it("parses a backend-signed available terminal and its nested snapshot", () => {
    const event = parseRecommendationFinalV1(availableRecommendationFinalWire())

    expect(event.terminal_status).toBe("available")
    expect(event.recommendations[0]).toMatchObject({
      rank: 1,
      resource_type: "quiz",
      score: 0.75,
    })
    expect(event.candidate_snapshot?.targets[0].resource_id).toBe("python.loops.quiz")
    expect(recommendationFinalDedupeKey(event)).toContain(
      "recommendation-final:v1:",
    )
  })

  it("accepts no_eligible_candidates but rejects the automatic-only reason", () => {
    const event = parseRecommendationFinalV1(unavailableRecommendationFinalWire())
    expect(event.unavailable_reason).toBe("no_eligible_candidates")

    expect(() =>
      parseRecommendationFinalV1({
        ...unavailableRecommendationFinalWire(),
        unavailable_reason: "generated_resources_unavailable",
      }),
    ).toThrow(/invalid for an explicit recommendation/)
  })

  it("rejects top-level and nested schema drift without coercion", () => {
    expect(() =>
      parseRecommendationFinalV1({ ...availableRecommendationFinalWire(), legacy: true }),
    ).toThrow(/unexpected field: legacy/)

    const rankDrift = availableRecommendationFinalWire()
    ;(rankDrift.recommendations as Record<string, unknown>[])[0].rank = "1"
    expect(() => parseRecommendationFinalV1(rankDrift)).toThrow(/rank must be an integer/)

    const nestedDrift = availableRecommendationFinalWire()
    ;(nestedDrift.candidate_snapshot as Record<string, unknown>).legacy = true
    expect(() => parseRecommendationFinalV1(nestedDrift)).toThrow(
      /unexpected field: legacy/,
    )

    expect(() =>
      parseRecommendationFinalV1({
        ...availableRecommendationFinalWire(),
        generated_at: "2026-02-30T08:00:00+00:00",
      }),
    ).toThrow(/generated_at must use canonical/)
  })

  it("rejects payload, snapshot, and final identity tampering", () => {
    expect(() =>
      parseRecommendationFinalV1({
        ...availableRecommendationFinalWire(),
        summary: "Tampered summary.",
      }),
    ).toThrow(/payload_hash does not match/)

    const snapshotTamper = availableRecommendationFinalWire()
    ;(snapshotTamper.candidate_snapshot as Record<string, unknown>).source_data_version =
      "other"
    expect(() => parseRecommendationFinalV1(snapshotTamper)).toThrow(
      /snapshot_id does not match/,
    )

    expect(() =>
      parseRecommendationFinalV1({
        ...availableRecommendationFinalWire(),
        recommendation_final_id: `recommendation-final:v1:${"0".repeat(64)}`,
      }),
    ).toThrow(/recommendation_final_id does not match/)
  })

  it.each([
    {
      score: 1,
      payloadHash:
        "recommendation-final-payload:v1:adaec8bf61386a70afc4d3620c617b3068b850fb5eff92a934a13d63f14001cb",
      finalId:
        "recommendation-final:v1:9a16be5fc14d441fea626f0e935515e26fcb3bb57b20e68c826ec3db4ed8be00",
    },
    {
      score: 0.00001,
      payloadHash:
        "recommendation-final-payload:v1:7ecadc08f56cef6a96a434228eee1adead0035e74ad7f462f71be31e7771f8fb",
      finalId:
        "recommendation-final:v1:4c1d5b014e360c069cb7ab36a2af8a1eca13737dcc84dd139e163a0b1026e91a",
    },
  ])("matches Python canonical float hashing for score $score", ({ score, payloadHash, finalId }) => {
    const wire = availableRecommendationFinalWire()
    ;(wire.recommendations as Record<string, unknown>[])[0].score = score
    wire.payload_hash = payloadHash
    wire.recommendation_final_id = finalId

    expect(parseRecommendationFinalV1(wire).recommendations[0].score).toBe(score)
  })

  it("binds one validated terminal to its exact assistant request and deduplicates replay", () => {
    const event = parseRecommendationFinalV1(availableRecommendationFinalWire())
    const messages: RecommendationFinalMessage[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "provisional text must not commit",
        requestId: event.request_id,
        threadId: event.thread_id,
      },
    ]
    const first = attachRecommendationFinalToMessages(messages, event, "assistant-1")
    expect(first.attached).toBe(true)
    expect(first.messages).toHaveLength(1)
    expect(first.messages[0].content).toBe("")
    expect(first.messages[0].recommendationFinal).toEqual(event)

    const replay = attachRecommendationFinalToMessages(first.messages, event)
    expect(replay.attached).toBe(false)
    expect(replay.messages).toBe(first.messages)
  })

  it("does not attach a prior request to the current placeholder or overwrite another final", () => {
    const event = parseRecommendationFinalV1(availableRecommendationFinalWire())
    const current: RecommendationFinalMessage[] = [
      {
        id: "assistant-current",
        role: "assistant",
        content: "",
        requestId: "00000000-0000-4000-8000-000000000002",
        threadId: event.thread_id,
      },
    ]
    const attached = attachRecommendationFinalToMessages(
      current,
      event,
      "assistant-current",
    )
    expect(attached.messages).toHaveLength(2)
    expect(attached.messages[0].recommendationFinal).toBeUndefined()

    expect(() =>
      attachRecommendationFinalToMessages(
        [
          {
            id: "assistant-conflict",
            role: "assistant",
            content: "A committed QA answer",
            requestId: event.request_id,
            threadId: event.thread_id,
            qaFinal: { type: "qa_final" },
          },
        ],
        event,
      ),
    ).toThrow(/different authoritative final/)
  })
})
