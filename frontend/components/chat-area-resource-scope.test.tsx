// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import { beforeAll, describe, expect, it, vi } from "vitest"

import { ChatArea } from "@/components/chat-area"

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

describe("ChatArea evidence-limited resources", () => {
  it("renders a friendly scope notice without exposing the internal warning code", () => {
    render(
      <ChatArea
        messages={[
          {
            id: "assistant-scope",
            role: "assistant",
            content: "",
            resourceScopeNotice: true,
          },
        ]}
        onSendMessage={vi.fn()}
      />,
    )

    expect(screen.getByRole("note")).toHaveTextContent("基础版：内容仅覆盖当前资料范围。")
    expect(screen.queryByText("evidence_scope_limited")).not.toBeInTheDocument()
  })
})
