import { describe, expect, it, vi } from "vitest"

import {
  fetchLearningGuidanceCatalog,
  freezeOnboardingAttempt,
  getOrCreateOnboardingAttempt,
  OnboardingAttemptError,
  OnboardingClientError,
  submitOnboardingV2,
  type KeyValueStorage,
} from "@/lib/onboarding-client"
import {
  buildOnboardRequestV2,
  parseLearningGuidanceCatalogV1,
  type LearningGuidanceCatalogV1,
} from "@/lib/onboarding-contracts"

const API_BASE_URL = "http://api.example.test"
const REQUEST_ID = "00000000-0000-4000-8000-000000000402"
const USER_ID = "u_onboarding_client"

class MemoryStorage implements KeyValueStorage {
  readonly values = new Map<string, string>()

  getItem(key: string): string | null {
    return this.values.get(key) ?? null
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value)
  }

  removeItem(key: string): void {
    this.values.delete(key)
  }
}

function catalogWire(): Record<string, unknown> {
  return {
    schema_version: "learning_guidance_catalog_v1",
    data_version: "2026.07.15",
    artifact_fingerprint: "b".repeat(64),
    subjects: [
      {
        subject_id: "python",
        title: "Python",
        topics: [{ topic_id: "python.basics", title: "Python 基础" }],
      },
    ],
  }
}

function catalog(): LearningGuidanceCatalogV1 {
  return parseLearningGuidanceCatalogV1(catalogWire())
}

function request(goal = "掌握 Python 基础") {
  return buildOnboardRequestV2(
    {
      requestId: REQUEST_ID,
      userId: USER_ID,
      nickname: "学习者",
      grade: "大一",
      dislikes: [],
      topics: [
        {
          subject: "python",
          topic_id: "python.basics",
          level: 0.5,
          confidence: 0.75,
          goal,
          importance: 1,
          progress: 0.25,
        },
      ],
      preferences: { prefer_examples: 0.8 },
    },
    catalog(),
  )
}

function successResult(status: "created" | "replayed" = "created") {
  return {
    schema_version: "onboard_result_v2",
    status,
    request_id: REQUEST_ID,
    user_id: USER_ID,
    summary: "学习画像已创建",
    skills_count: 1,
    goals_count: 1,
    preferences_count: 1,
  }
}

describe("onboarding HTTP client", () => {
  it("loads only the strict learning-guidance catalog endpoint", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(catalogWire()), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const fetchImpl = fetchMock as unknown as typeof fetch

    const value = await fetchLearningGuidanceCatalog({
      apiBaseUrl: API_BASE_URL,
      fetchImpl,
    })

    expect(value.subjects[0].topics[0].topic_id).toBe("python.basics")
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    expect(fetchImpl).toHaveBeenCalledWith(
      `${API_BASE_URL}/learning-guidance/catalog`,
      expect.objectContaining({ method: "GET" }),
    )
    expect(String(fetchMock.mock.calls[0][0])).not.toContain("/subjects")
  })

  it("submits the exact strict payload and binds the authoritative result identity", async () => {
    const payload = request()
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(JSON.stringify(successResult()), {
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      }),
    )
    const fetchImpl = fetchMock as unknown as typeof fetch

    const result = await submitOnboardingV2({
      apiBaseUrl: API_BASE_URL,
      catalog: catalog(),
      request: payload,
      fetchImpl,
    })

    expect(result.status).toBe("created")
    const [, init] = fetchMock.mock.calls[0]
    expect(JSON.parse(String(init?.body))).toEqual(payload)
    expect(init?.headers).toEqual({
      Accept: "application/json",
      "Content-Type": "application/json",
    })
  })

  it("rejects a successful response with a different request identity", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(
        JSON.stringify({
          ...successResult(),
          request_id: "00000000-0000-4000-8000-000000000999",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    ) as unknown as typeof fetch

    await expect(
      submitOnboardingV2({
        apiBaseUrl: API_BASE_URL,
        catalog: catalog(),
        request: request(),
        fetchImpl,
      }),
    ).rejects.toThrow(/identity or inventory/)
  })

  it("reports a typed HTTP failure without exposing the raw server body", async () => {
    const privateBody = "private-provider-body-canary"
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ detail: privateBody }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    ) as unknown as typeof fetch

    const error = await fetchLearningGuidanceCatalog({
      apiBaseUrl: API_BASE_URL,
      fetchImpl,
    }).catch((caught: unknown) => caught)

    expect(error).toBeInstanceOf(OnboardingClientError)
    expect((error as OnboardingClientError).code).toBe("onboarding_http_failed")
    expect((error as OnboardingClientError).status).toBe(503)
    expect((error as Error).message).not.toContain(privateBody)
    expect((error as OnboardingClientError).remoteCode).toBeNull()
  })

  it.each([
    ["trailing slash", `${API_BASE_URL}/`],
    ["credentials", "https://user:password@example.test"],
    ["relative URL", "/api"],
  ])("rejects invalid API configuration: %s", async (_label, apiBaseUrl) => {
    const fetchImpl = vi.fn() as unknown as typeof fetch
    await expect(
      fetchLearningGuidanceCatalog({ apiBaseUrl, fetchImpl }),
    ).rejects.toMatchObject({ code: "onboarding_client_configuration_invalid" })
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it("rejects non-JSON success instead of treating it as a catalog", async () => {
    const fetchImpl = vi.fn(async () =>
      new Response("ok", { status: 200, headers: { "Content-Type": "text/plain" } }),
    ) as unknown as typeof fetch

    await expect(
      fetchLearningGuidanceCatalog({ apiBaseUrl: API_BASE_URL, fetchImpl }),
    ).rejects.toMatchObject({ code: "onboarding_response_content_type_invalid" })
  })
})

describe("onboarding idempotency attempt", () => {
  it("creates one stable per-user request_id and reuses it", () => {
    const storage = new MemoryStorage()
    const uuidFactory = vi.fn(() => REQUEST_ID)

    const first = getOrCreateOnboardingAttempt({ storage, userId: USER_ID, uuidFactory })
    const second = getOrCreateOnboardingAttempt({ storage, userId: USER_ID, uuidFactory })

    expect(first.request_id).toBe(REQUEST_ID)
    expect(second).toEqual(first)
    expect(uuidFactory).toHaveBeenCalledTimes(1)
  })

  it("freezes the first payload and permits only byte-equivalent logical retries", () => {
    const storage = new MemoryStorage()
    getOrCreateOnboardingAttempt({ storage, userId: USER_ID, uuidFactory: () => REQUEST_ID })

    const first = freezeOnboardingAttempt({
      storage,
      userId: USER_ID,
      catalog: catalog(),
      request: request(),
    })
    const replay = freezeOnboardingAttempt({
      storage,
      userId: USER_ID,
      catalog: catalog(),
      request: request(),
    })

    expect(replay).toEqual(first)
    expect(Object.isFrozen(replay)).toBe(true)
    expect(() =>
      freezeOnboardingAttempt({
        storage,
        userId: USER_ID,
        catalog: catalog(),
        request: request("改写后的目标"),
      }),
    ).toThrow(OnboardingAttemptError)
  })

  it("sends an identical frozen body across transport retries", async () => {
    const storage = new MemoryStorage()
    getOrCreateOnboardingAttempt({ storage, userId: USER_ID, uuidFactory: () => REQUEST_ID })
    const frozen = freezeOnboardingAttempt({
      storage,
      userId: USER_ID,
      catalog: catalog(),
      request: request(),
    })
    const bodies: string[] = []
    const fetchImpl = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      bodies.push(String(init?.body))
      return new Response(JSON.stringify(successResult("replayed")), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }) as unknown as typeof fetch

    await submitOnboardingV2({
      apiBaseUrl: API_BASE_URL,
      catalog: catalog(),
      request: frozen,
      fetchImpl,
    })
    await submitOnboardingV2({
      apiBaseUrl: API_BASE_URL,
      catalog: catalog(),
      request: frozen,
      fetchImpl,
    })

    expect(bodies).toHaveLength(2)
    expect(bodies[1]).toBe(bodies[0])
    expect(JSON.parse(bodies[0]).profile.request_id).toBe(REQUEST_ID)
  })

  it("fails closed on a corrupted local attempt instead of rotating identity", () => {
    const storage = new MemoryStorage()
    storage.values.set(`a3_onboarding_attempt_v2:${USER_ID}`, "{not-json")
    const uuidFactory = vi.fn(() => REQUEST_ID)

    expect(() => getOrCreateOnboardingAttempt({ storage, userId: USER_ID, uuidFactory })).toThrow(
      OnboardingAttemptError,
    )
    expect(uuidFactory).not.toHaveBeenCalled()
  })
})
