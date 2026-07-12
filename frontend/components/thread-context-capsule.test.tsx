// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { ThreadContextCapsule } from "@/components/thread-context-capsule"
import { parseThreadContextWindowV3 } from "@/lib/thread-context-window-v3"

function contextWindow() {
  const empty = {
    retained_tokens: 0,
    lifetime_injected_tokens: 0,
    lifetime_unique_tokens: 0,
    injection_count: 0,
    repeat_injection_count: 0,
    active_item_count: 0,
  }
  return parseThreadContextWindowV3({
    schema_version: 3,
    thread_id: "thread-1",
    updated_at: "2026-07-13T00:00:00Z",
    updating: false,
    window_model: "deepseek-v4-pro",
    context_window_limit_tokens: 1000,
    retained_memory_tokens: 100,
    retained_ratio: 0.1,
    lifetime_injected_tokens: 180,
    lifetime_unique_tokens: 100,
    request_count: 2,
    injection_count: 3,
    repeat_injection_count: 1,
    injection_types: {
      profile: empty,
      memory: {
        retained_tokens: 100,
        lifetime_injected_tokens: 180,
        lifetime_unique_tokens: 100,
        injection_count: 3,
        repeat_injection_count: 1,
        active_item_count: 1,
      },
      evidence: empty,
      artifact: empty,
      rules: empty,
      curriculum: empty,
      trajectory: empty,
      pipeline: empty,
    },
    measurement: {
      last_tokenizer_mode: "estimated_mixed_v1",
      last_estimated: true,
      estimated_injection_count: 3,
    },
    memory_summary: { active_item_count: 1, active_unique_content_count: 1 },
    compaction: {
      status: "never",
      boundary_id: "",
      compacted_at: null,
      before_tokens: 0,
      after_tokens: 0,
    },
  })
}

describe("ThreadContextCapsule", () => {
  it("shows retained memory percentage and exact basis", async () => {
    render(<ThreadContextCapsule window={contextWindow()} closeSignal="thread-1:idle" />)
    const trigger = screen.getByRole("button", { name: "查看上下文记忆" })
    expect(trigger).toHaveTextContent("上下文记忆 10%")
    fireEvent.mouseEnter(trigger)
    fireEvent.focus(trigger)
    expect(await screen.findByRole("tooltip")).toHaveTextContent("当前保留 100 / 1.0k")
  })

  it("shows lifetime, injection types, and compaction without predictions", async () => {
    render(<ThreadContextCapsule window={contextWindow()} closeSignal="thread-1:idle" />)
    const trigger = screen.getByRole("button", { name: "查看上下文记忆" })
    fireEvent.click(trigger)
    expect(await screen.findByRole("dialog", { name: "上下文记忆详情" })).toBeInTheDocument()
    expect(screen.getByText("当前保留记忆")).toBeInTheDocument()
    expect(screen.getByText("会话累计")).toBeInTheDocument()
    expect(screen.getByText("注入类型")).toBeInTheDocument()
    expect(screen.getByText("压缩状态")).toBeInTheDocument()
    expect(screen.queryByText(/下一次调用|线程基线|输出预留/)).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "关闭上下文记忆详情" }))
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "上下文记忆详情" })).not.toBeInTheDocument()
    })
    expect(trigger).toHaveFocus()
  })

  it("closes when request identity changes", async () => {
    const window = contextWindow()
    const view = render(
      <ThreadContextCapsule window={window} closeSignal="thread-1:idle" />,
    )
    fireEvent.click(screen.getByRole("button", { name: "查看上下文记忆" }))
    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    view.rerender(
      <ThreadContextCapsule window={window} closeSignal="thread-1:running" />,
    )
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  })

  it("supports Escape dismissal", async () => {
    render(<ThreadContextCapsule window={contextWindow()} closeSignal="thread-1:idle" />)
    fireEvent.click(screen.getByRole("button", { name: "查看上下文记忆" }))
    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    fireEvent.keyDown(document, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  })
})
