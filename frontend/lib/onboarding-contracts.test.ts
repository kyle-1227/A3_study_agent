import { describe, expect, it } from "vitest"

import {
  buildOnboardRequestV2,
  OnboardingContractError,
  parseLearningGuidanceCatalogV1,
  parseOnboardRequestV2,
  parseOnboardResultV2,
  type LearningGuidanceCatalogV1,
  type OnboardRequestV2,
} from "@/lib/onboarding-contracts"

const REQUEST_ID = "00000000-0000-4000-8000-000000000401"
const USER_ID = "u_onboarding_contract"

function catalogWire(): Record<string, unknown> {
  return {
    schema_version: "learning_guidance_catalog_v1",
    data_version: "2026.07.15",
    artifact_fingerprint: "a".repeat(64),
    subjects: [
      {
        subject_id: "math",
        title: "数学",
        topics: [
          { topic_id: "math.algebra", title: "代数" },
          { topic_id: "math.geometry", title: "几何" },
        ],
      },
    ],
  }
}

function catalog(): LearningGuidanceCatalogV1 {
  return parseLearningGuidanceCatalogV1(catalogWire())
}

function request(): OnboardRequestV2 {
  return buildOnboardRequestV2(
    {
      requestId: REQUEST_ID,
      userId: USER_ID,
      nickname: "学习者",
      grade: "大一",
      dislikes: ["死记硬背"],
      topics: [
        {
          subject: "math",
          topic_id: "math.algebra",
          level: 0.5,
          confidence: 0.75,
          goal: "掌握代数推导",
          importance: 1,
          progress: 0.25,
        },
        {
          subject: "math",
          topic_id: "math.geometry",
          level: 0.25,
          confidence: 0.5,
          goal: "理解几何证明",
          importance: 0.75,
          progress: 0,
        },
      ],
      preferences: { prefer_examples: 0.8, prefer_visual: 0.5 },
    },
    catalog(),
  )
}

describe("learning guidance catalog v1", () => {
  it("strictly parses the production subject and topic identity inventory", () => {
    const value = catalog()

    expect(value.subjects[0].topics[1].topic_id).toBe("math.geometry")
    expect(Object.isFrozen(value)).toBe(true)
    expect(Object.isFrozen(value.subjects)).toBe(true)
    expect(Object.isFrozen(value.subjects[0].topics)).toBe(true)
  })

  it.each([
    ["wrong schema", { schema_version: "learning_guidance_catalog_v0" }],
    ["unknown field", { extra: true }],
    ["invalid fingerprint", { artifact_fingerprint: "not-a-hash" }],
  ])("rejects %s", (_label, override) => {
    expect(() => parseLearningGuidanceCatalogV1({ ...catalogWire(), ...override })).toThrow(
      OnboardingContractError,
    )
  })

  it("rejects duplicate topic identities across subjects", () => {
    const wire = catalogWire()
    wire.subjects = [
      ...(wire.subjects as unknown[]),
      {
        subject_id: "physics",
        title: "物理",
        topics: [{ topic_id: "math.algebra", title: "重复主题" }],
      },
    ]

    expect(() => parseLearningGuidanceCatalogV1(wire)).toThrow(/globally unique/)
  })
})

describe("onboard v2 request", () => {
  it("builds one explicit skill and goal per topic and expands selected preferences", () => {
    const value = request()

    expect(value.profile.skills).toHaveLength(2)
    expect(value.profile.goals).toHaveLength(2)
    expect(value.profile.preferences).toHaveLength(4)
    expect(value.profile.preferences.map((item) => [item.topic_id, item.dimension])).toEqual([
      ["math.algebra", "prefer_examples"],
      ["math.geometry", "prefer_examples"],
      ["math.algebra", "prefer_visual"],
      ["math.geometry", "prefer_visual"],
    ])
    expect(Object.isFrozen(value.profile.skills)).toBe(true)
    expect(Object.isFrozen(value.profile.skills[0])).toBe(true)
  })

  it("orders selected topics by the server catalog instead of selection order", () => {
    const value = buildOnboardRequestV2(
      {
        requestId: REQUEST_ID,
        userId: USER_ID,
        nickname: "",
        grade: "大一",
        dislikes: [],
        topics: [
          {
            subject: "math",
            topic_id: "math.geometry",
            level: 0.5,
            confidence: 0.5,
            goal: "学习几何",
            importance: 0.5,
            progress: 0,
          },
          {
            subject: "math",
            topic_id: "math.algebra",
            level: 0.5,
            confidence: 0.5,
            goal: "学习代数",
            importance: 0.5,
            progress: 0,
          },
        ],
        preferences: {},
      },
      catalog(),
    )

    expect(value.profile.skills.map((item) => item.topic_id)).toEqual([
      "math.algebra",
      "math.geometry",
    ])
  })

  it("rejects the retired subjects and skill_levels payload", () => {
    expect(() =>
      parseOnboardRequestV2({
        user_id: USER_ID,
        nickname: "",
        subjects: ["math"],
        skill_levels: { math: 0.5 },
        goals: ["学习数学"],
        learning_style: {},
        grade: "大一",
        dislikes: [],
      }),
    ).toThrow(OnboardingContractError)
  })

  it("rejects skill and goal coverage drift at topic granularity", () => {
    const value = structuredClone(request()) as unknown as Record<string, unknown>
    const profile = value.profile as Record<string, unknown>
    const goals = profile.goals as Array<Record<string, unknown>>
    goals[1].topic_id = "math.algebra"

    expect(() => parseOnboardRequestV2(value, catalog())).toThrow(/exactly one goal per topic/)
  })

  it("rejects an unknown catalog topic instead of repairing its identity", () => {
    const value = structuredClone(request()) as unknown as Record<string, unknown>
    const profile = value.profile as Record<string, unknown>
    const skills = profile.skills as Array<Record<string, unknown>>
    const goals = profile.goals as Array<Record<string, unknown>>
    const preferences = profile.preferences as Array<Record<string, unknown>>
    skills[0].topic_id = "math.unknown"
    goals[0].topic_id = "math.unknown"
    for (const preference of preferences) {
      if (preference.topic_id === "math.algebra") preference.topic_id = "math.unknown"
    }

    expect(() => parseOnboardRequestV2(value, catalog())).toThrow(/supplied catalog/)
  })

  it.each([
    ["request UUID", { requestId: "request-1" }],
    ["explicit confidence", { topics: [{ ...baseTopic(), confidence: undefined }] }],
    ["normalized goal", { topics: [{ ...baseTopic(), goal: " goal with spaces " }] }],
    ["unit interval", { topics: [{ ...baseTopic(), progress: 1.1 }] }],
  ])("rejects invalid %s", (_label, override) => {
    const input = {
      requestId: REQUEST_ID,
      userId: USER_ID,
      nickname: "",
      grade: "大一",
      dislikes: [],
      topics: [baseTopic()],
      preferences: {},
      ...override,
    }

    expect(() => buildOnboardRequestV2(input as never, catalog())).toThrow(
      OnboardingContractError,
    )
  })
})

describe("onboard result v2", () => {
  it("accepts a matching authoritative result", () => {
    const submitted = request()
    const result = parseOnboardResultV2(
      {
        schema_version: "onboard_result_v2",
        status: "created",
        request_id: REQUEST_ID,
        user_id: USER_ID,
        summary: "学习画像已创建",
        skills_count: 2,
        goals_count: 2,
        preferences_count: 4,
      },
      {
        requestId: submitted.profile.request_id,
        userId: submitted.profile.user_id,
        skillsCount: submitted.profile.skills.length,
        goalsCount: submitted.profile.goals.length,
        preferencesCount: submitted.profile.preferences.length,
      },
    )

    expect(result.status).toBe("created")
    expect(Object.isFrozen(result)).toBe(true)
  })

  it.each([
    ["request identity", { request_id: "00000000-0000-4000-8000-000000000999" }],
    ["user identity", { user_id: "u_someone_else" }],
    ["inventory count", { skills_count: 1 }],
    ["terminal status", { status: "success" }],
  ])("rejects mismatched %s", (_label, override) => {
    expect(() =>
      parseOnboardResultV2(
        {
          schema_version: "onboard_result_v2",
          status: "replayed",
          request_id: REQUEST_ID,
          user_id: USER_ID,
          summary: "学习画像已存在",
          skills_count: 2,
          goals_count: 2,
          preferences_count: 4,
          ...override,
        },
        {
          requestId: REQUEST_ID,
          userId: USER_ID,
          skillsCount: 2,
          goalsCount: 2,
          preferencesCount: 4,
        },
      ),
    ).toThrow(OnboardingContractError)
  })
})

function baseTopic() {
  return {
    subject: "math",
    topic_id: "math.algebra",
    level: 0.5,
    confidence: 0.75,
    goal: "掌握代数推导",
    importance: 1,
    progress: 0.25,
  }
}
