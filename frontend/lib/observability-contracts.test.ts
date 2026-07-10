import { describe, expect, it } from "vitest"

import {
  ContractParseError,
  parseActivityEvent,
  parseActivityTimeline,
  parseBackgroundContextWindow,
  parseContextUsageReport,
  parseGraphManifest,
} from "@/lib/observability-contracts"
import {
  activityPayload,
  backgroundContextPayload,
  contextUsageReportPayload,
  graphManifestPayload,
  LATER,
} from "@/test/observability-fixtures"

describe("observability contracts", () => {
  it("parses backend graph labels and rejects dangling edges", () => {
    const manifest = parseGraphManifest(graphManifestPayload())
    expect(manifest.nodes.map((node) => node.label)).toEqual(["后端起点", "后端输出"])

    expect(() =>
      parseGraphManifest(
        graphManifestPayload({
          edges: [
            {
              edge_id: "invalid",
              source: "start_node",
              target: "missing_node",
              kind: "graph",
              conditional: false,
              label: "",
              workflow: "",
            },
          ],
        }),
      ),
    ).toThrow(ContractParseError)
  })

  it("keeps only allowlisted activity details", () => {
    const activity = parseActivityEvent(
      activityPayload({
        safe_details: {
          resource_id: "resource:v1:fixture",
          status_code: 200,
          raw_prompt: "must not cross the contract",
          api_key: "secret",
        },
      }),
    )
    expect(activity.safeDetails).toEqual({
      resource_id: "resource:v1:fixture",
      status_code: 200,
    })
  })

  it("degrades a corrupt stored timeline entry without discarding valid entries", () => {
    const parsed = parseActivityTimeline([
      activityPayload(),
      { schema_version: "activity_event_v1", activity_id: "broken" },
    ])
    expect(parsed.items).toHaveLength(1)
    expect(parsed.rejectedCount).toBe(1)
  })

  it("requires terminal activity timestamps", () => {
    expect(() =>
      parseActivityEvent(activityPayload({ status: "completed", updated_at: LATER })),
    ).toThrow(/completed_at/)
  })

  it("reconciles the exact provider input report", () => {
    const report = parseContextUsageReport(contextUsageReportPayload())
    expect(report.usedTokens).toBe(120)
    expect(report.mainCategories.map((item) => item.estimatedTokens)).toEqual([60, 40])

    expect(() =>
      parseContextUsageReport(contextUsageReportPayload({ used_tokens: 121 })),
    ).toThrow(/reconcile/)
  })

  it("accepts the versioned numeric background schema and rejects drift", () => {
    expect(parseBackgroundContextWindow(backgroundContextPayload()).schemaVersion).toBe(1)
    expect(() =>
      parseBackgroundContextWindow(backgroundContextPayload({ schema_version: 2 })),
    ).toThrow(/schema_version/)
  })
})
