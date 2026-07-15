// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { replaceMock } = vi.hoisted(() => ({ replaceMock: vi.fn() }))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: replaceMock }),
}))

vi.mock("@/lib/public-config", () => ({
  requirePublicApiBaseUrl: () => "http://api.example.test",
}))

import OnboardingPage from "@/app/onboarding/page"

const REQUEST_ID = "00000000-0000-4000-8000-000000000403"
const USER_ID = "u_onboarding_page"

function catalogResponse(): Response {
  return new Response(
    JSON.stringify({
      schema_version: "learning_guidance_catalog_v1",
      data_version: "2026.07.15",
      artifact_fingerprint: "c".repeat(64),
      subjects: [
        {
          subject_id: "python",
          title: "Python",
          topics: [{ topic_id: "python.basics", title: "Python 基础" }],
        },
      ],
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  )
}

describe("onboarding V2 page", () => {
  beforeEach(() => {
    localStorage.clear()
    localStorage.setItem("a3_user_id", USER_ID)
    localStorage.setItem("a3_nickname", "学习者")
    replaceMock.mockReset()
    vi.stubGlobal("crypto", { randomUUID: vi.fn(() => REQUEST_ID) })
  })

  it("collects every required topic field and writes completion only after a bound result", async () => {
    let submitted: Record<string, unknown> | null = null
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith("/learning-guidance/catalog")) return catalogResponse()
      if (url.endsWith("/onboard")) {
        submitted = JSON.parse(String(init?.body)) as Record<string, unknown>
        const profile = submitted.profile as Record<string, unknown>
        return new Response(
          JSON.stringify({
            schema_version: "onboard_result_v2",
            status: "created",
            request_id: profile.request_id,
            user_id: profile.user_id,
            summary: "学习画像已创建",
            skills_count: (profile.skills as unknown[]).length,
            goals_count: (profile.goals as unknown[]).length,
            preferences_count: (profile.preferences as unknown[]).length,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        )
      }
      throw new Error("unexpected endpoint")
    })
    vi.stubGlobal("fetch", fetchMock)

    render(<OnboardingPage />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    fireEvent.click(screen.getByRole("button", { name: "大一" }))
    fireEvent.click(screen.getByRole("button", { name: /下一步/ }))
    expect(await screen.findByRole("button", { name: "Python 基础" })).toBeEnabled()
    expect(screen.queryByPlaceholderText(/自定义/)).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Python 基础" }))
    fireEvent.click(screen.getByRole("button", { name: /下一步/ }))
    expect(screen.getByText("当前掌握水平")).toBeInTheDocument()
    expect(screen.getByText("对这次自评的信心")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "中等" }))
    fireEvent.click(screen.getByRole("button", { name: "较确定" }))
    fireEvent.click(screen.getByRole("button", { name: /下一步/ }))

    fireEvent.change(screen.getByLabelText(/Python 基础 的目标/), {
      target: { value: "掌握 Python 基础语法" },
    })
    fireEvent.click(screen.getByRole("button", { name: "最高" }))
    fireEvent.click(screen.getByRole("button", { name: "未开始" }))
    fireEvent.click(screen.getAllByRole("button", { name: "很喜欢" })[0])
    fireEvent.click(screen.getByRole("button", { name: /完成并开始学习/ }))

    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith("/"))
    expect(localStorage.getItem("a3_onboarding_completed")).toBe("true")
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(submitted).not.toBeNull()
    expect(submitted).toMatchObject({
      schema_version: "onboard_v2",
      nickname: "学习者",
      grade: "大一",
      dislikes: [],
    })
    expect(submitted).not.toHaveProperty("subjects")
    expect(submitted).not.toHaveProperty("skill_levels")
    const submittedPayload = submitted as unknown as Record<string, unknown>
    const profile = submittedPayload.profile as Record<string, unknown>
    expect(profile.request_id).toBe(REQUEST_ID)
    expect(profile.user_id).toBe(USER_ID)
    expect(profile.skills).toEqual([
      {
        subject: "python",
        topic_id: "python.basics",
        level: 0.5,
        confidence: 0.75,
      },
    ])
    expect(profile.goals).toEqual([
      {
        subject: "python",
        topic_id: "python.basics",
        goal: "掌握 Python 基础语法",
        importance: 1,
        progress: 0,
      },
    ])
    expect(profile.preferences).toEqual([
      {
        subject: "python",
        topic_id: "python.basics",
        dimension: "prefer_examples",
        strength: 0.8,
      },
    ])
  })

  it("blocks topic selection when the strict catalog is unavailable", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ detail: "learning_guidance_runtime_unavailable" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    )
    vi.stubGlobal("fetch", fetchMock)

    render(<OnboardingPage />)

    expect(await screen.findByRole("alert")).toHaveTextContent("学习主题服务暂不可用")
    expect(screen.queryByText("Python 基础")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: /下一步/ })).toBeDisabled()
  })

  it("does not write the completion marker when the response identity drifts", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/learning-guidance/catalog")) return catalogResponse()
      const payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      const profile = payload.profile as Record<string, unknown>
      return new Response(
        JSON.stringify({
          schema_version: "onboard_result_v2",
          status: "created",
          request_id: "00000000-0000-4000-8000-000000000999",
          user_id: profile.user_id,
          summary: "错误身份",
          skills_count: 1,
          goals_count: 1,
          preferences_count: 0,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    })
    vi.stubGlobal("fetch", fetchMock)

    render(<OnboardingPage />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole("button", { name: "大一" }))
    fireEvent.click(screen.getByRole("button", { name: /下一步/ }))
    fireEvent.click(await screen.findByRole("button", { name: "Python 基础" }))
    fireEvent.click(screen.getByRole("button", { name: /下一步/ }))
    fireEvent.click(screen.getByRole("button", { name: "中等" }))
    fireEvent.click(screen.getByRole("button", { name: "较确定" }))
    fireEvent.click(screen.getByRole("button", { name: /下一步/ }))
    fireEvent.change(screen.getByLabelText(/Python 基础 的目标/), {
      target: { value: "掌握 Python 基础语法" },
    })
    fireEvent.click(screen.getByRole("button", { name: "最高" }))
    fireEvent.click(screen.getByRole("button", { name: "未开始" }))
    fireEvent.click(screen.getByRole("button", { name: /完成并开始学习/ }))

    expect(await screen.findByRole("alert")).toHaveTextContent("服务返回了不符合契约的数据")
    expect(localStorage.getItem("a3_onboarding_completed")).toBeNull()
    expect(replaceMock).not.toHaveBeenCalled()
  })
})
