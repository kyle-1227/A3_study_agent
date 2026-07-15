// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import { beforeAll, describe, expect, it, vi } from "vitest"

import { ChatArea, type Message } from "@/components/chat-area"
import { parseRecommendationFinalV1 } from "@/lib/recommendation-final"
import {
  availableRecommendationFinalWire,
  unavailableRecommendationFinalWire,
} from "@/test/recommendation-final-fixtures"

beforeAll(() => {
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    configurable: true,
    value: vi.fn(),
  })
  vi.stubGlobal(
    "requestAnimationFrame",
    (callback: FrameRequestCallback) => window.setTimeout(() => callback(0), 0),
  )
  vi.stubGlobal("cancelAnimationFrame", (handle: number) => window.clearTimeout(handle))
})

function recommendationMessage(
  wire: Record<string, unknown>,
): Message {
  const recommendationFinal = parseRecommendationFinalV1(wire)
  return {
    id: "assistant-recommendation",
    role: "assistant",
    content: "",
    threadId: recommendationFinal.thread_id,
    requestId: recommendationFinal.request_id,
    recommendationFinal,
  }
}

describe("ChatArea recommendation final", () => {
  it("renders a compact accessible recommendation list from the validated final", () => {
    render(
      <ChatArea
        messages={[recommendationMessage(availableRecommendationFinalWire())]}
        onSendMessage={vi.fn()}
      />,
    )

    expect(screen.getByRole("region", { name: "个性化资源推荐" })).toBeInTheDocument()
    expect(screen.getByRole("list", { name: "推荐资源列表" })).toBeInTheDocument()
    expect(screen.getByText("Python loops quiz")).toBeInTheDocument()
    expect(screen.getByLabelText("匹配度 75%")).toBeInTheDocument()
  })

  it("renders an explicit unavailable state without a fabricated recommendation", () => {
    render(
      <ChatArea
        messages={[recommendationMessage(unavailableRecommendationFinalWire())]}
        onSendMessage={vi.fn()}
      />,
    )

    expect(screen.getByText("暂不可用")).toBeInTheDocument()
    expect(screen.getByRole("status")).toHaveTextContent("当前没有可安全展示的推荐结果")
    expect(screen.queryByRole("list", { name: "推荐资源列表" })).not.toBeInTheDocument()
  })
})
