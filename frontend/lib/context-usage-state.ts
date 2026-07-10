import type {
  ContextUsageReport,
  ContextUsageReportError,
} from "@/lib/observability-contracts"

export interface ContextUsageState {
  report: ContextUsageReport | null
  error: ContextUsageReportError | null
  updating: boolean
}

export const EMPTY_CONTEXT_USAGE_STATE: ContextUsageState = {
  report: null,
  error: null,
  updating: false,
}

export function beginContextUsageUpdate(state: ContextUsageState): ContextUsageState {
  return { ...state, error: null, updating: true }
}

export function finishContextUsageUpdate(state: ContextUsageState): ContextUsageState {
  return state.updating ? { ...state, updating: false } : state
}

export function applyContextUsageReport(
  state: ContextUsageState,
  report: ContextUsageReport,
): ContextUsageState {
  if (state.report?.reportId === report.reportId && !state.updating && !state.error) {
    return state
  }
  return { report, error: null, updating: false }
}

export function applyContextUsageError(
  state: ContextUsageState,
  error: ContextUsageReportError,
): ContextUsageState {
  return { ...state, error, updating: false }
}

export function restoreContextUsageReport(
  report: ContextUsageReport | null,
): ContextUsageState {
  return { report, error: null, updating: false }
}
