export type StreamPhase = "idle" | "running" | "waiting" | "completed" | "failed"

export interface StreamLifecycleState {
  phase: StreamPhase
  terminalEvent: "" | "done" | "error" | "interrupt"
}

export const IDLE_STREAM_LIFECYCLE: StreamLifecycleState = {
  phase: "idle",
  terminalEvent: "",
}

export function beginStreamLifecycle(): StreamLifecycleState {
  return { phase: "running", terminalEvent: "" }
}

export function reduceStreamLifecycle(
  state: StreamLifecycleState,
  event: unknown,
): StreamLifecycleState {
  if (!isRecord(event) || typeof event.type !== "string") return state
  if (event.type === "done") return { phase: "completed", terminalEvent: "done" }
  if (event.type === "error") return { phase: "failed", terminalEvent: "error" }
  if (event.type === "interrupt") return { phase: "waiting", terminalEvent: "interrupt" }
  return state
}

export function streamPhaseIsTerminal(state: StreamLifecycleState): boolean {
  return state.phase === "waiting" || state.phase === "completed" || state.phase === "failed"
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value))
}
