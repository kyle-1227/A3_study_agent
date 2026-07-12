// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { ThreadContextCapsule } from "@/components/thread-context-capsule"
import { parseThreadContextWindowV2 } from "@/lib/observability-contracts"
import { threadContextWindowV2Payload } from "@/test/observability-fixtures"

describe("ThreadContextCapsule", () => {
  it("shows only the estimated percentage in its hover tooltip", async () => {
    render(
      <ThreadContextCapsule
        window={parseThreadContextWindowV2(threadContextWindowV2Payload())}
        closeSignal="thread-1:idle"
      />,
    )

    const trigger = screen.getByRole("button", {
      name: "查看线程背景信息窗口",
    })
    expect(trigger).toHaveTextContent("10%")
    fireEvent.mouseEnter(trigger)
    fireEvent.focus(trigger)

    expect(await screen.findByRole("tooltip")).toHaveTextContent("预计 10% 已用")
    expect(screen.queryByText("最近一次 LLM 调用")).not.toBeInTheDocument()
  })

  it("opens detailed sections and closes through the explicit retract control", async () => {
    render(
      <ThreadContextCapsule
        window={parseThreadContextWindowV2(threadContextWindowV2Payload())}
        closeSignal="thread-1:idle"
      />,
    )
    const trigger = screen.getByRole("button", {
      name: "查看线程背景信息窗口",
    })

    fireEvent.click(trigger)
    expect(await screen.findByRole("dialog", { name: "线程背景信息详情" })).toBeInTheDocument()
    expect(screen.getByText("下一次调用估算")).toBeInTheDocument()
    expect(screen.getByText("最近一次 LLM 调用")).toBeInTheDocument()
    expect(screen.getByText("线程背景库存")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "收回背景信息窗口" }))
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "线程背景信息详情" })).not.toBeInTheDocument()
    })
    expect(trigger).toHaveFocus()
  })

  it("closes when a new request or thread changes the close signal", async () => {
    const window = parseThreadContextWindowV2(threadContextWindowV2Payload())
    const view = render(
      <ThreadContextCapsule window={window} closeSignal="thread-1:idle" />,
    )
    fireEvent.click(
      screen.getByRole("button", { name: "查看线程背景信息窗口" }),
    )
    expect(await screen.findByRole("dialog", { name: "线程背景信息详情" })).toBeInTheDocument()

    view.rerender(
      <ThreadContextCapsule window={window} closeSignal="thread-1:running" />,
    )
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "线程背景信息详情" })).not.toBeInTheDocument()
    })
  })

  it("supports Escape and outside-click dismissal", async () => {
    render(
      <ThreadContextCapsule
        window={parseThreadContextWindowV2(threadContextWindowV2Payload())}
        closeSignal="thread-1:idle"
      />,
    )
    const trigger = screen.getByRole("button", {
      name: "查看线程背景信息窗口",
    })
    fireEvent.click(trigger)
    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    fireEvent.keyDown(document, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())

    fireEvent.click(trigger)
    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    fireEvent.pointerDown(document.body)
    fireEvent.click(document.body)
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
  })
})
