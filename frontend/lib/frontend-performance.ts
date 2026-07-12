import type { FrontendPerformanceCapability } from "@/lib/observability-contracts"

export const FRONTEND_PERFORMANCE_SCHEMA_VERSION = "frontend_performance_v1" as const

export type FrontendTerminalStatus = "completed" | "failed" | "interrupted"
export type FrontendMilestoneName =
  | "submit_to_stream_context"
  | "submit_to_first_event"
  | "submit_to_first_token"
  | "submit_to_resource_final"
  | "submit_to_done"
  | "submit_to_interrupt"
  | "submit_to_error"

export interface FrontendPerformanceMilestone {
  name: FrontendMilestoneName
  duration_ms: number
  count?: number
  status?: FrontendTerminalStatus
}

export interface FrontendPerformanceBatch {
  schema_version: typeof FRONTEND_PERFORMANCE_SCHEMA_VERSION
  request_id: string
  thread_id: string
  trace_id: string
  milestones: FrontendPerformanceMilestone[]
}

export type FrontendPerformanceDelivery =
  | { status: "skipped" }
  | { status: "delivered" }
  | { status: "incomplete"; reason: string }

interface Clock {
  now(): number
}

const MILESTONE_BY_EVENT: Partial<Record<string, FrontendMilestoneName>> = {
  stream_context: "submit_to_stream_context",
  token: "submit_to_first_token",
  resource_final: "submit_to_resource_final",
}

const TERMINAL_BY_EVENT: Partial<Record<string, { name: FrontendMilestoneName; status: FrontendTerminalStatus }>> = {
  done: { name: "submit_to_done", status: "completed" },
  error: { name: "submit_to_error", status: "failed" },
  interrupt: { name: "submit_to_interrupt", status: "interrupted" },
}

export class FrontendPerformanceTracker {
  private readonly startedAt: number
  private readonly milestones = new Map<FrontendMilestoneName, FrontendPerformanceMilestone>()
  private capability: FrontendPerformanceCapability | null = null
  private requestId = ""
  private threadId = ""
  private terminal = false
  private deliveryStarted = false

  constructor(private readonly clock: Clock) {
    this.startedAt = clock.now()
  }

  recordEvent(type: unknown): void {
    if (typeof type !== "string") return
    if (!this.milestones.has("submit_to_first_event")) {
      this.record("submit_to_first_event")
    }
    const milestone = MILESTONE_BY_EVENT[type]
    if (milestone) this.record(milestone)
    const terminal = TERMINAL_BY_EVENT[type]
    if (terminal) {
      if (this.terminal) return
      this.record(terminal.name, terminal.status)
      this.terminal = true
    }
  }

  bind(
    capability: FrontendPerformanceCapability,
    {
      requestId,
      threadId,
    }: {
      requestId: string
      threadId: string
    },
  ): void {
    if (this.capability) {
      if (
        this.capability.traceId !== capability.traceId ||
        this.requestId !== requestId ||
        this.threadId !== threadId
      ) {
        throw new Error("frontend_performance_capability_binding_mismatch")
      }
      return
    }
    this.capability = capability
    this.requestId = requestId
    this.threadId = threadId
  }

  isTerminal(): boolean {
    return this.terminal
  }

  buildBatch(): FrontendPerformanceBatch | null {
    if (!this.capability || !this.requestId || !this.threadId || !this.terminal) {
      return null
    }
    const milestones = [...this.milestones.values()]
    if (milestones.length === 0 || milestones.length > 16) {
      throw new Error("frontend_performance_milestone_contract_invalid")
    }
    return {
      schema_version: FRONTEND_PERFORMANCE_SCHEMA_VERSION,
      request_id: this.requestId,
      thread_id: this.threadId,
      trace_id: this.capability.traceId,
      milestones,
    }
  }

  async deliver(fetchImpl: typeof fetch, apiBaseUrl: string): Promise<FrontendPerformanceDelivery> {
    if (this.deliveryStarted) return { status: "skipped" }
    const batch = this.buildBatch()
    if (!batch || !this.capability) return { status: "skipped" }
    this.deliveryStarted = true
    try {
      const response = await fetchImpl(`${apiBaseUrl}${this.capability.endpoint}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.capability.token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(batch),
      })
      if (response.status === 204) return { status: "delivered" }
      return { status: "incomplete", reason: `frontend_performance_http_${response.status}` }
    } catch {
      return { status: "incomplete", reason: "frontend_performance_delivery_failed" }
    }
  }

  private record(name: FrontendMilestoneName, status?: FrontendTerminalStatus): void {
    if (this.milestones.has(name)) return
    const duration = Math.max(0, this.clock.now() - this.startedAt)
    this.milestones.set(name, {
      name,
      duration_ms: Math.round(duration * 1000) / 1000,
      ...(status ? { status } : {}),
    })
  }
}
