// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { EvidenceProgressPanel } from "@/components/evidence-progress-panel"
import {
  emptyEvidenceProgressTimeline,
  parseEvidenceProgressEvent,
  reduceEvidenceProgress,
} from "@/lib/evidence-progress"

describe("EvidenceProgressPanel", () => {
  it("renders only the safe, validated progress projection", () => {
    const event = parseEvidenceProgressEvent({
      schema_version: "evidence_progress_v1",
      progress_id: `evidence-progress:v1:${"a".repeat(64)}`,
      request_id: "00000000-0000-4000-8000-000000000001",
      thread_id: "thread-1",
      lifecycle_key: "plan",
      phase_status: "completed",
      details: {
        stage: "evidence_orchestration.plan.accepted",
        requirement_count: 2,
        resource_count: 1,
        subject_count: 2,
        budget_max_rounds: 2,
        budget_max_tasks: 10,
      },
    })
    const timeline = reduceEvidenceProgress(emptyEvidenceProgressTimeline(), event)

    render(<EvidenceProgressPanel timeline={timeline} />)

    expect(screen.getByRole("region", { name: "证据补搜进度" })).toBeInTheDocument()
    expect(screen.getByText("证据需求规划完成")).toBeInTheDocument()
    expect(screen.getByText("2 项要求 · 1 类资源 · 2 个学科")).toBeInTheDocument()
    expect(screen.queryByText(/query|https?:\/\//i)).not.toBeInTheDocument()
  })
})
