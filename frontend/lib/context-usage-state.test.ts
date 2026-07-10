import { describe, expect, it } from "vitest"

import {
  applyContextUsageError,
  applyContextUsageReport,
  beginContextUsageUpdate,
  finishContextUsageUpdate,
  restoreContextUsageReport,
} from "@/lib/context-usage-state"
import {
  parseContextUsageReport,
  parseContextUsageReportError,
} from "@/lib/observability-contracts"
import { contextUsageReportPayload } from "@/test/observability-fixtures"

describe("context usage state", () => {
  it("preserves the prior thread window while the next request updates", () => {
    const report = parseContextUsageReport(contextUsageReportPayload())
    const updating = beginContextUsageUpdate(restoreContextUsageReport(report))
    expect(updating.report).toBe(report)
    expect(updating.updating).toBe(true)
    expect(finishContextUsageUpdate(updating)).toEqual({
      report,
      error: null,
      updating: false,
    })
  })

  it("preserves the prior report when a new report fails", () => {
    const report = parseContextUsageReport(contextUsageReportPayload())
    const error = parseContextUsageReportError({
      schema_version: "context_usage_report_error_v1",
      manifest_id: "llm_input_manifest:v1:failed",
      node_name: "fixture_node",
      llm_node: "fixture_llm_node",
      provider: "configured-provider",
      model: "configured-model",
      reason: "budget_unavailable",
      warning: "budget accounting failed",
      error_type: "ContextUsageError",
    })
    const failed = applyContextUsageError(
      beginContextUsageUpdate(restoreContextUsageReport(report)),
      error,
    )
    expect(failed.report).toBe(report)
    expect(failed.error).toBe(error)
    expect(failed.updating).toBe(false)
  })

  it("atomically replaces a completed report", () => {
    const first = parseContextUsageReport(contextUsageReportPayload())
    const second = parseContextUsageReport(
      contextUsageReportPayload({
        report_id: "context_usage:v1:second",
        node_name: "second_node",
      }),
    )
    expect(applyContextUsageReport(restoreContextUsageReport(first), second)).toEqual({
      report: second,
      error: null,
      updating: false,
    })
  })
})
