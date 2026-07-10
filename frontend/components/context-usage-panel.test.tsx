// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { ContextUsagePanel } from "@/components/context-usage-panel"
import {
  beginContextUsageUpdate,
  restoreContextUsageReport,
} from "@/lib/context-usage-state"
import {
  parseBackgroundContextWindow,
  parseContextUsageReport,
} from "@/lib/observability-contracts"
import {
  backgroundContextPayload,
  contextUsageReportPayload,
} from "@/test/observability-fixtures"

describe("ContextUsagePanel", () => {
  it("keeps the prior usage visible while the next request updates", () => {
    const report = parseContextUsageReport(contextUsageReportPayload())
    render(
      <ContextUsagePanel
        state={beginContextUsageUpdate(restoreContextUsageReport(report))}
        background={parseBackgroundContextWindow(backgroundContextPayload())}
      />,
    )
    expect(screen.getByText("12% 已用")).toBeInTheDocument()
    expect(screen.getByText("更新中")).toBeInTheDocument()
    expect(screen.getByText("最近调用 · fixture_node")).toBeInTheDocument()
  })
})
