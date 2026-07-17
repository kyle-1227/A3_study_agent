export const CONTEXT_INJECTION_TYPES = [
  "profile",
  "memory",
  "evidence",
  "artifact",
  "rules",
  "curriculum",
  "trajectory",
  "pipeline",
] as const

export type ContextInjectionType = (typeof CONTEXT_INJECTION_TYPES)[number]

export interface ContextInjectionTypeStatsV3 {
  retainedTokens: number
  lifetimeInjectedTokens: number
  lifetimeUniqueTokens: number
  injectionCount: number
  repeatInjectionCount: number
  activeItemCount: number
}

export interface ThreadContextWindowV3 {
  schemaVersion: 3
  threadId: string
  updatedAt: string
  updating: boolean
  windowModel: string
  contextWindowLimitTokens: number
  retainedMemoryTokens: number
  retainedRatio: number
  lifetimeInjectedTokens: number
  lifetimeUniqueTokens: number
  requestCount: number
  injectionCount: number
  repeatInjectionCount: number
  injectionTypes: Record<ContextInjectionType, ContextInjectionTypeStatsV3>
  measurement: {
    lastTokenizerMode: string
    lastEstimated: boolean
    estimatedInjectionCount: number
  }
  memorySummary: {
    activeItemCount: number
    activeUniqueContentCount: number
  }
  compaction: {
    status: "never" | "compacted"
    boundaryId: string
    compactedAt: string | null
    beforeTokens: number
    afterTokens: number
  }
}

export class ThreadContextWindowV3Error extends Error {
  constructor(message: string) {
    super(message)
    this.name = "ThreadContextWindowV3Error"
  }
}

export function markThreadContextWindowV3Updating(
  current: ThreadContextWindowV3 | null,
): ThreadContextWindowV3 | null {
  if (current === null || current.updating) return current
  return { ...current, updating: true }
}

export function finishThreadContextWindowV3Update(
  current: ThreadContextWindowV3 | null,
): ThreadContextWindowV3 | null {
  if (current === null || !current.updating) return current
  return { ...current, updating: false }
}

export function threadContextWindowV3ForSelection(
  current: ThreadContextWindowV3 | null,
  activeThreadId: string | null,
  selectedThreadId: string,
): ThreadContextWindowV3 | null {
  if (
    current === null ||
    activeThreadId !== selectedThreadId ||
    current.threadId !== selectedThreadId
  ) {
    return null
  }
  return current
}

export function parseThreadContextWindowV3(value: unknown): ThreadContextWindowV3 {
  const data = record(value, "thread_context_window_v3")
  exactKeys(data, [
    "schema_version",
    "thread_id",
    "updated_at",
    "updating",
    "window_model",
    "context_window_limit_tokens",
    "retained_memory_tokens",
    "retained_ratio",
    "lifetime_injected_tokens",
    "lifetime_unique_tokens",
    "request_count",
    "injection_count",
    "repeat_injection_count",
    "injection_types",
    "measurement",
    "memory_summary",
    "compaction",
  ])
  if (data.schema_version !== 3) fail("schema_version must equal 3")
  const limit = positiveInteger(data.context_window_limit_tokens, "context_window_limit_tokens")
  const retained = nonNegativeInteger(data.retained_memory_tokens, "retained_memory_tokens")
  const retainedRatio = nonNegativeNumber(data.retained_ratio, "retained_ratio")
  if (Math.abs(retainedRatio - retained / limit) > 1e-7) {
    fail("retained_ratio does not match retained tokens and window limit")
  }
  const injectionTypes = parseInjectionTypes(data.injection_types)
  const lifetimeInjectedTokens = nonNegativeInteger(
    data.lifetime_injected_tokens,
    "lifetime_injected_tokens",
  )
  const lifetimeUniqueTokens = nonNegativeInteger(
    data.lifetime_unique_tokens,
    "lifetime_unique_tokens",
  )
  const injectionCount = nonNegativeInteger(data.injection_count, "injection_count")
  if (sum(injectionTypes, "retainedTokens") !== retained) fail("retained type totals do not match")
  if (sum(injectionTypes, "lifetimeInjectedTokens") !== lifetimeInjectedTokens) {
    fail("lifetime type totals do not match")
  }
  if (sum(injectionTypes, "lifetimeUniqueTokens") !== lifetimeUniqueTokens) {
    fail("unique type totals do not match")
  }
  if (sum(injectionTypes, "injectionCount") !== injectionCount) {
    fail("injection type counts do not match")
  }

  const measurement = record(data.measurement, "measurement")
  exactKeys(measurement, [
    "last_tokenizer_mode",
    "last_estimated",
    "estimated_injection_count",
  ])
  const memorySummary = record(data.memory_summary, "memory_summary")
  exactKeys(memorySummary, ["active_item_count", "active_unique_content_count"])
  const compaction = record(data.compaction, "compaction")
  exactKeys(compaction, [
    "status",
    "boundary_id",
    "compacted_at",
    "before_tokens",
    "after_tokens",
  ])
  if (compaction.status !== "never" && compaction.status !== "compacted") {
    fail("compaction.status is invalid")
  }

  return {
    schemaVersion: 3,
    threadId: requiredString(data.thread_id, "thread_id"),
    updatedAt: timestamp(data.updated_at, "updated_at"),
    updating: booleanValue(data.updating, "updating"),
    windowModel: requiredString(data.window_model, "window_model"),
    contextWindowLimitTokens: limit,
    retainedMemoryTokens: retained,
    retainedRatio,
    lifetimeInjectedTokens,
    lifetimeUniqueTokens,
    requestCount: nonNegativeInteger(data.request_count, "request_count"),
    injectionCount,
    repeatInjectionCount: nonNegativeInteger(
      data.repeat_injection_count,
      "repeat_injection_count",
    ),
    injectionTypes,
    measurement: {
      lastTokenizerMode: optionalString(measurement.last_tokenizer_mode, "last_tokenizer_mode"),
      lastEstimated: booleanValue(measurement.last_estimated, "last_estimated"),
      estimatedInjectionCount: nonNegativeInteger(
        measurement.estimated_injection_count,
        "estimated_injection_count",
      ),
    },
    memorySummary: {
      activeItemCount: nonNegativeInteger(memorySummary.active_item_count, "active_item_count"),
      activeUniqueContentCount: nonNegativeInteger(
        memorySummary.active_unique_content_count,
        "active_unique_content_count",
      ),
    },
    compaction: {
      status: compaction.status,
      boundaryId: optionalString(compaction.boundary_id, "boundary_id"),
      compactedAt:
        compaction.compacted_at === null
          ? null
          : timestamp(compaction.compacted_at, "compacted_at"),
      beforeTokens: nonNegativeInteger(compaction.before_tokens, "before_tokens"),
      afterTokens: nonNegativeInteger(compaction.after_tokens, "after_tokens"),
    },
  }
}

function parseInjectionTypes(value: unknown): Record<ContextInjectionType, ContextInjectionTypeStatsV3> {
  const data = record(value, "injection_types")
  exactKeys(data, [...CONTEXT_INJECTION_TYPES])
  return Object.fromEntries(
    CONTEXT_INJECTION_TYPES.map((source) => {
      const stats = record(data[source], `injection_types.${source}`)
      exactKeys(stats, [
        "retained_tokens",
        "lifetime_injected_tokens",
        "lifetime_unique_tokens",
        "injection_count",
        "repeat_injection_count",
        "active_item_count",
      ])
      return [
        source,
        {
          retainedTokens: nonNegativeInteger(stats.retained_tokens, `${source}.retained_tokens`),
          lifetimeInjectedTokens: nonNegativeInteger(
            stats.lifetime_injected_tokens,
            `${source}.lifetime_injected_tokens`,
          ),
          lifetimeUniqueTokens: nonNegativeInteger(
            stats.lifetime_unique_tokens,
            `${source}.lifetime_unique_tokens`,
          ),
          injectionCount: nonNegativeInteger(stats.injection_count, `${source}.injection_count`),
          repeatInjectionCount: nonNegativeInteger(
            stats.repeat_injection_count,
            `${source}.repeat_injection_count`,
          ),
          activeItemCount: nonNegativeInteger(stats.active_item_count, `${source}.active_item_count`),
        },
      ]
    }),
  ) as Record<ContextInjectionType, ContextInjectionTypeStatsV3>
}

function sum(
  values: Record<ContextInjectionType, ContextInjectionTypeStatsV3>,
  key: keyof ContextInjectionTypeStatsV3,
): number {
  return CONTEXT_INJECTION_TYPES.reduce((total, source) => total + values[source][key], 0)
}

function record(value: unknown, field: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${field} must be an object`)
  return value as Record<string, unknown>
}

function exactKeys(data: Record<string, unknown>, expected: string[]): void {
  const allowed = new Set(expected)
  const extra = Object.keys(data).find((key) => !allowed.has(key))
  if (extra) fail(`unexpected field: ${extra}`)
  const missing = expected.find((key) => !(key in data))
  if (missing) fail(`missing field: ${missing}`)
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) fail(`${field} is required`)
  return value
}

function optionalString(value: unknown, field: string): string {
  if (typeof value !== "string") fail(`${field} must be a string`)
  return value
}

function booleanValue(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") fail(`${field} must be a boolean`)
  return value
}

function positiveInteger(value: unknown, field: string): number {
  const result = nonNegativeInteger(value, field)
  if (result === 0) fail(`${field} must be positive`)
  return result
}

function nonNegativeInteger(value: unknown, field: string): number {
  if (!Number.isInteger(value) || (value as number) < 0) fail(`${field} must be a non-negative integer`)
  return value as number
}

function nonNegativeNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    fail(`${field} must be a non-negative number`)
  }
  return value
}

function timestamp(value: unknown, field: string): string {
  const text = requiredString(value, field)
  if (!Number.isFinite(Date.parse(text)) || !/(?:Z|[+-]\d\d:\d\d)$/.test(text)) {
    fail(`${field} must be timezone-aware`)
  }
  return text
}

function fail(message: string): never {
  throw new ThreadContextWindowV3Error(message)
}
