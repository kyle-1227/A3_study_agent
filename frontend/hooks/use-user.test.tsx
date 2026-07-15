// @vitest-environment jsdom

import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { pushMock } = vi.hoisted(() => ({ pushMock: vi.fn() }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
}))

vi.mock("@/lib/public-config", () => ({
  requirePublicApiBaseUrl: () => "http://api.example.test",
}))

import { ONBOARDING_ATTEMPT_STORAGE_PREFIX } from "@/lib/onboarding-client"
import { useUser } from "@/hooks/use-user"

const STORED_USER_ID = "u_existing_user"
const GENERATED_UUID = "00000000-0000-4000-8000-000000000404"

describe("useUser onboarding identity", () => {
  beforeEach(() => {
    localStorage.clear()
    pushMock.mockReset()
    vi.stubGlobal("crypto", { randomUUID: vi.fn(() => GENERATED_UUID) })
  })

  it("verifies the backend profile even when a local completion marker exists", async () => {
    localStorage.setItem("a3_user_id", STORED_USER_ID)
    localStorage.setItem("a3_onboarding_completed", "true")
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ detail: "Profile not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    )
    vi.stubGlobal("fetch", fetchMock)

    const { result } = renderHook(() => useUser())

    await waitFor(() => expect(result.current.isLoading).toBe(false))
    expect(result.current.hasProfile).toBe(false)
    expect(result.current.profileAvailability).toBe("missing")
    expect(fetchMock).toHaveBeenCalledWith(
      `http://api.example.test/profile/${STORED_USER_ID}`,
      { headers: { Accept: "application/json" } },
    )
  })

  it("keeps backend failures distinct from a missing profile", async () => {
    localStorage.setItem("a3_user_id", STORED_USER_ID)
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response("{}", {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    )

    const { result } = renderHook(() => useUser())

    await waitFor(() => expect(result.current.isLoading).toBe(false))
    expect(result.current.hasProfile).toBe(false)
    expect(result.current.profileAvailability).toBe("unavailable")
  })

  it("treats transport failure as unavailable rather than missing", async () => {
    localStorage.setItem("a3_user_id", STORED_USER_ID)
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("offline"))))

    const { result } = renderHook(() => useUser())

    await waitFor(() => expect(result.current.isLoading).toBe(false))
    expect(result.current.profileAvailability).toBe("unavailable")
  })

  it("uses a canonical browser UUID when starting a new onboarding identity", () => {
    vi.stubGlobal("fetch", vi.fn())
    const { result } = renderHook(() => useUser())

    act(() => result.current.startOnboarding())

    expect(localStorage.getItem("a3_user_id")).toBe(`u_${GENERATED_UUID}`)
    expect(pushMock).toHaveBeenCalledWith("/onboarding")
  })

  it("clears the per-user idempotency attempt together with local identity", async () => {
    localStorage.setItem("a3_user_id", STORED_USER_ID)
    localStorage.setItem("a3_nickname", "学习者")
    localStorage.setItem("a3_onboarding_completed", "true")
    localStorage.setItem(`${ONBOARDING_ATTEMPT_STORAGE_PREFIX}${STORED_USER_ID}`, "attempt")
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }),
      ),
    )
    const { result } = renderHook(() => useUser())
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    act(() => result.current.clearUser())

    expect(localStorage.getItem("a3_user_id")).toBeNull()
    expect(localStorage.getItem("a3_nickname")).toBeNull()
    expect(localStorage.getItem("a3_onboarding_completed")).toBeNull()
    expect(localStorage.getItem(`${ONBOARDING_ATTEMPT_STORAGE_PREFIX}${STORED_USER_ID}`)).toBeNull()
    expect(result.current.userId).toBeNull()
  })
})
