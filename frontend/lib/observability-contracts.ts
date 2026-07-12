export const ACTIVITY_STATUSES = [
  "queued",
  "running",
  "waiting",
  "completed",
  "retrying",
  "interrupted",
  "failed",
  "skipped",
] as const

export const ACTIVITY_KINDS = [
  "stream",
  "node",
  "llm",
  "tool",
  "retrieval",
  "context",
  "review",
  "interrupt",
  "artifact",
  "retry",
] as const

export const CONTEXT_MAIN_CATEGORIES = [
  "system_prompt",
  "tool_definitions",
  "rules",
  "skills",
  "subagent_definitions",
  "conversation",
  "unclassified",
] as const

export type ActivityStatus = (typeof ACTIVITY_STATUSES)[number]
export type ActivityKind = (typeof ACTIVITY_KINDS)[number]
export type ContextMainCategory = (typeof CONTEXT_MAIN_CATEGORIES)[number]
export type ContextWarningLevel = "ok" | "warning" | "critical" | "overflow"

export type SafeActivityDetail = string | number | boolean | null

export interface GraphManifestNode {
  nodeId: string
  label: string
  description: string
  kind: string
  group: string
  parent: string
  workflow: string
  order: number
  stageRank: number
  visible: boolean
  logical: boolean
  activityRunning: string
  activityCompleted: string
}

export interface GraphManifestEdge {
  edgeId: string
  source: string
  target: string
  kind: "graph" | "logical"
  conditional: boolean
  label: string
  workflow: string
}

export interface GraphCapabilityMetadata {
  resourceTypes: string[]
  contextPolicyMode: string
  checkpointerEnabled: boolean
  checkpointerType: string
  physicalNodeCount: number
  logicalNodeCount: number
}

export interface GraphManifest {
  schemaVersion: "graph_manifest_v1"
  graphVersion: string
  generatedAt: string
  nodes: GraphManifestNode[]
  edges: GraphManifestEdge[]
  capabilityMetadata: GraphCapabilityMetadata
}

export interface GraphManifestUnavailable {
  schemaVersion: "graph_manifest_error_v1"
  error: "graph_manifest_unavailable"
  reason: string
  errorType: string
}

export interface StreamContext {
  schemaVersion: "stream_context_v1"
  requestId: string
  threadId: string
  graphVersion: string
}

export interface FrontendPerformanceCapability {
  schemaVersion: "frontend_performance_capability_v1"
  endpoint: string
  traceId: string
  token: string
  expiresAt: string
}

export interface GraphManifestRef {
  schemaVersion: "graph_manifest_ref_v1"
  graphVersion: string
  endpoint: string
}

export interface ActivityEvent {
  schemaVersion: "activity_event_v1"
  activityId: string
  sequence: number
  threadId: string
  requestId: string
  kind: ActivityKind
  status: ActivityStatus
  node: string
  parent: string
  title: string
  summary: string
  tool: string
  model: string
  startedAt: string
  updatedAt: string
  completedAt: string
  durationMs?: number
  safeDetails: Record<string, SafeActivityDetail>
}

export interface ContextUsageCategory {
  category: string
  estimatedTokens: number
  segmentCount: number
  messageCount: number
}

export interface ContextUsageSegment {
  segmentId: string
  fingerprint: string
  messageIndex: number
  role: string
  mainCategory: ContextMainCategory
  detailedCategory: string
  charCount: number
  estimatedTokens: number
  provenance: Record<string, string>
}

export interface ContextUsageReport {
  schemaVersion: "context_usage_report_v1"
  reportId: string
  manifestId: string
  createdAt: string
  requestId: string
  threadId: string
  nodeName: string
  llmNode: string
  provider: string
  model: string
  inputEstimatedTokens: number
  outputReservedTokens: number
  usedTokens: number
  maxContextTokens: number
  availableTokens: number
  usedRatio: number
  warningLevel: ContextWarningLevel
  estimated: boolean
  tokenizerMode: string
  messageCount: number
  schemaSizeChars?: number
  mainCategories: ContextUsageCategory[]
  detailedCategories: ContextUsageCategory[]
  overlapRollups: ContextUsageCategory[]
  segments: ContextUsageSegment[]
  unclassifiedTokens: number
  reconciliationOk: boolean
  reconciliationWarnings: string[]
}

export interface ContextUsageReportError {
  schemaVersion: "context_usage_report_error_v1"
  manifestId: string
  nodeName: string
  llmNode: string
  provider: string
  model: string
  reason: string
  warning: string
  errorType: string
}

export interface BackgroundContextWindow {
  schemaVersion: 1
  threadId: string
  updatedAt: string
  lastManifestId: string
  usedTokens: number
  maxContextTokens: number
  usedRatio: number
  messageCount: number
  sectionCount: number
  sectionNames: string[]
  conversationSummaryPresent: boolean
  selectedMemoryCount: number
  artifactSummaryCount: number
  evidenceSummaryCount: number
  ceBlockPresent: boolean
  structuredContractPresent: boolean
  manifestCount: number
  influenceEntryCount: number
  influenceTokenEstimate: number
  influenceSourceNodeCount: number
  workspacePresent: boolean
  workspaceActiveSubject: string
  workspaceEvidenceSummaryCount: number
  workspaceGapCount: number
  workspaceArtifactCount: number
  workspaceUpdatedAt: string
}

export interface ContextSectionEstimate {
  section: string
  source: string
  itemCount: number
  messageCount: number
  charCount: number
  estimatedTokens: number
  known: boolean
}

export interface NextCallContextEstimate {
  basis: "known_next_node" | "thread_baseline"
  confidence: "high" | "medium" | "low"
  estimated: boolean
  estimatedAt: string
  targetNode: string
  unknownSections: string[]
  sections: ContextSectionEstimate[]
  estimatedInputTokens: number
  estimatedOutputReservedTokens: number
  estimatedUsedTokens: number
  maxContextTokens: number
  usedRatio: number
  tokenizerMode: string
  stateFingerprint: string
  reusedManifestStatistics: boolean
  knownSectionRatio: number
}

export interface LastLLMCallUsage {
  present: boolean
  reportId: string
  manifestId: string
  createdAt: string
  nodeName: string
  llmNode: string
  model: string
  inputEstimatedTokens: number
  outputReservedTokens: number
  usedTokens: number
  maxContextTokens: number
  usedRatio: number
  estimated: boolean
  tokenizerMode: string
  sections: ContextSectionEstimate[]
}

export interface BackgroundInventory {
  conversationSummaryPresent: boolean
  selectedMemoryCount: number
  evidenceSummaryCount: number
  artifactSummaryCount: number
  workspacePresent: boolean
  workspaceActiveSubject: string
  workspaceEvidenceSummaryCount: number
  workspaceGapCount: number
  workspaceArtifactCount: number
  workspaceUpdatedAt: string
  manifestCount: number
  influenceEntryCount: number
}

export interface ThreadContextWindowV2 {
  schemaVersion: 2
  contract: "thread_context_window_v2"
  threadId: string
  updatedAt: string
  nextCallContextEstimate: NextCallContextEstimate
  lastLlmCallUsage: LastLLMCallUsage
  backgroundInventory: BackgroundInventory
}

export interface ParsedCollection<T> {
  items: T[]
  rejectedCount: number
}

export class ContractParseError extends Error {
  readonly contract: string
  readonly reason: string

  constructor(contract: string, reason: string) {
    super(`${contract}: ${reason}`)
    this.name = "ContractParseError"
    this.contract = contract
    this.reason = reason
  }
}

const SAFE_ACTIVITY_DETAIL_KEYS = new Set([
  "error_type",
  "finish_reason",
  "interrupt_type",
  "item_count",
  "manifest_id",
  "max_retries",
  "output_mode",
  "report_id",
  "resource_id",
  "resource_type",
  "retry_count",
  "status_code",
  "trace_call_id",
  "trace_seq",
  "warning_level",
])

const SAFE_PROVENANCE_KEYS = new Set([
  "source",
  "source_type",
  "source_id",
  "overlap",
  "source_node",
  "purpose",
  "reason",
  "segment_count",
  "message_role",
])

export function parseGraphManifest(value: unknown): GraphManifest {
  const data = record(value, "graph_manifest_v1")
  literal(data, "schema_version", "graph_manifest_v1", "graph_manifest_v1")
  const graphVersion = requiredString(data, "graph_version", "graph_manifest_v1")
  if (!graphVersion.startsWith("graph:v1:")) {
    fail("graph_manifest_v1", "graph_version prefix is invalid")
  }
  const nodes = requiredArray(data, "nodes", "graph_manifest_v1").map((item) =>
    parseGraphNode(item),
  )
  const nodeIds = new Set(nodes.map((node) => node.nodeId))
  if (nodeIds.size !== nodes.length) {
    fail("graph_manifest_v1", "duplicate node_id")
  }
  const edges = requiredArray(data, "edges", "graph_manifest_v1").map((item) =>
    parseGraphEdge(item),
  )
  for (const edge of edges) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
      fail("graph_manifest_v1", "edge references an unknown node")
    }
  }
  return {
    schemaVersion: "graph_manifest_v1",
    graphVersion,
    generatedAt: utcTimestamp(data, "generated_at", "graph_manifest_v1"),
    nodes,
    edges,
    capabilityMetadata: parseCapabilityMetadata(data.capability_metadata),
  }
}

export function parseGraphManifestUnavailable(value: unknown): GraphManifestUnavailable {
  const envelope = record(value, "graph_manifest_error_v1")
  const data = isRecord(envelope.detail) ? envelope.detail : envelope
  literal(data, "schema_version", "graph_manifest_error_v1", "graph_manifest_error_v1")
  literal(data, "error", "graph_manifest_unavailable", "graph_manifest_error_v1")
  return {
    schemaVersion: "graph_manifest_error_v1",
    error: "graph_manifest_unavailable",
    reason: requiredString(data, "reason", "graph_manifest_error_v1"),
    errorType: requiredString(data, "error_type", "graph_manifest_error_v1"),
  }
}

export function parseStreamContext(value: unknown): StreamContext {
  const data = record(value, "stream_context_v1")
  literal(data, "schema_version", "stream_context_v1", "stream_context_v1")
  return {
    schemaVersion: "stream_context_v1",
    requestId: requiredString(data, "request_id", "stream_context_v1"),
    threadId: requiredString(data, "thread_id", "stream_context_v1"),
    graphVersion: versionedGraphId(data, "graph_version", "stream_context_v1"),
  }
}

export function parseFrontendPerformanceCapability(value: unknown): FrontendPerformanceCapability {
  const data = record(value, "frontend_performance_capability_v1")
  literal(
    data,
    "schema_version",
    "frontend_performance_capability_v1",
    "frontend_performance_capability_v1",
  )
  if (data.enabled !== true) {
    fail("frontend_performance_capability_v1", "enabled must be true")
  }
  const endpoint = requiredString(data, "endpoint", "frontend_performance_capability_v1")
  if (!/^\/[a-z0-9/_-]{1,119}$/.test(endpoint)) {
    fail("frontend_performance_capability_v1", "endpoint is invalid")
  }
  const traceId = requiredString(data, "trace_id", "frontend_performance_capability_v1")
  if (!/^trace:v1:[a-f0-9]{64}$/.test(traceId)) {
    fail("frontend_performance_capability_v1", "trace_id is invalid")
  }
  const token = requiredString(data, "token", "frontend_performance_capability_v1")
  if (token.length > 2048 || !/^[A-Za-z0-9._-]+$/.test(token)) {
    fail("frontend_performance_capability_v1", "token is invalid")
  }
  const expiresAt = utcTimestamp(data, "expires_at", "frontend_performance_capability_v1")
  return {
    schemaVersion: "frontend_performance_capability_v1",
    endpoint,
    traceId,
    token,
    expiresAt,
  }
}

export function parseGraphManifestRef(value: unknown): GraphManifestRef {
  const data = record(value, "graph_manifest_ref_v1")
  literal(data, "schema_version", "graph_manifest_ref_v1", "graph_manifest_ref_v1")
  const endpoint = requiredString(data, "endpoint", "graph_manifest_ref_v1")
  if (!endpoint.startsWith("/") || endpoint.startsWith("//") || endpoint.includes("?")) {
    fail("graph_manifest_ref_v1", "endpoint must be a query-free relative API path")
  }
  return {
    schemaVersion: "graph_manifest_ref_v1",
    graphVersion: versionedGraphId(data, "graph_version", "graph_manifest_ref_v1"),
    endpoint,
  }
}

export function parseActivityEvent(value: unknown): ActivityEvent {
  const data = record(value, "activity_event_v1")
  literal(data, "schema_version", "activity_event_v1", "activity_event_v1")
  const activityId = requiredString(data, "activity_id", "activity_event_v1")
  if (!activityId.startsWith("activity:v1:")) {
    fail("activity_event_v1", "activity_id prefix is invalid")
  }
  const status = enumValue(data, "status", ACTIVITY_STATUSES, "activity_event_v1")
  const completedAt = optionalString(data, "completed_at", "activity_event_v1")
  if (["completed", "interrupted", "failed", "skipped"].includes(status) && !completedAt) {
    fail("activity_event_v1", "terminal activity requires completed_at")
  }
  if (completedAt) validateUtcTimestamp(completedAt, "activity_event_v1", "completed_at")
  return {
    schemaVersion: "activity_event_v1",
    activityId,
    sequence: integer(data, "sequence", "activity_event_v1", 1),
    threadId: requiredString(data, "thread_id", "activity_event_v1"),
    requestId: requiredString(data, "request_id", "activity_event_v1"),
    kind: enumValue(data, "kind", ACTIVITY_KINDS, "activity_event_v1"),
    status,
    node: optionalString(data, "node", "activity_event_v1"),
    parent: optionalString(data, "parent", "activity_event_v1"),
    title: requiredString(data, "title", "activity_event_v1"),
    summary: optionalString(data, "summary", "activity_event_v1"),
    tool: optionalString(data, "tool", "activity_event_v1"),
    model: optionalString(data, "model", "activity_event_v1"),
    startedAt: utcTimestamp(data, "started_at", "activity_event_v1"),
    updatedAt: utcTimestamp(data, "updated_at", "activity_event_v1"),
    completedAt,
    durationMs: optionalNonNegativeNumber(data, "duration_ms", "activity_event_v1"),
    safeDetails: parseSafeActivityDetails(data.safe_details),
  }
}

export function parseActivityTimeline(value: unknown): ParsedCollection<ActivityEvent> {
  if (!Array.isArray(value)) return { items: [], rejectedCount: value == null ? 0 : 1 }
  const items: ActivityEvent[] = []
  let rejectedCount = 0
  for (const entry of value) {
    try {
      items.push(parseActivityEvent(entry))
    } catch (error) {
      if (!(error instanceof ContractParseError)) throw error
      rejectedCount += 1
    }
  }
  return { items, rejectedCount }
}

export function parseContextUsageReport(value: unknown): ContextUsageReport {
  const data = record(value, "context_usage_report_v1")
  literal(data, "schema_version", "context_usage_report_v1", "context_usage_report_v1")
  const mainCategories = requiredArray(
    data,
    "main_categories",
    "context_usage_report_v1",
  ).map((item) => parseUsageCategory(item))
  const segments = requiredArray(data, "segments", "context_usage_report_v1").map(
    (item) => parseUsageSegment(item),
  )
  const inputEstimatedTokens = integer(
    data,
    "input_estimated_tokens",
    "context_usage_report_v1",
    0,
  )
  if (sumTokens(mainCategories) !== inputEstimatedTokens) {
    fail("context_usage_report_v1", "main categories do not reconcile")
  }
  if (sumTokens(segments) !== inputEstimatedTokens) {
    fail("context_usage_report_v1", "segments do not reconcile")
  }
  const outputReservedTokens = integer(
    data,
    "output_reserved_tokens",
    "context_usage_report_v1",
    0,
  )
  const usedTokens = integer(data, "used_tokens", "context_usage_report_v1", 0)
  if (usedTokens !== inputEstimatedTokens + outputReservedTokens) {
    fail("context_usage_report_v1", "used_tokens does not reconcile")
  }
  const maxContextTokens = integer(
    data,
    "max_context_tokens",
    "context_usage_report_v1",
    1,
  )
  const availableTokens = integer(
    data,
    "available_tokens",
    "context_usage_report_v1",
    0,
  )
  if (availableTokens !== Math.max(maxContextTokens - usedTokens, 0)) {
    fail("context_usage_report_v1", "available_tokens does not reconcile")
  }
  if (data.reconciliation_ok !== true) {
    fail("context_usage_report_v1", "reconciliation_ok must be true")
  }
  const reportId = requiredString(data, "report_id", "context_usage_report_v1")
  if (!reportId.startsWith("context_usage:v1:")) {
    fail("context_usage_report_v1", "report_id prefix is invalid")
  }
  const warningLevel = enumValue(
    data,
    "warning_level",
    ["ok", "warning", "critical", "overflow"] as const,
    "context_usage_report_v1",
  )
  const schemaSizeChars = optionalNonNegativeNumber(
    data,
    "schema_size_chars",
    "context_usage_report_v1",
  )
  return {
    schemaVersion: "context_usage_report_v1",
    reportId,
    manifestId: requiredString(data, "manifest_id", "context_usage_report_v1"),
    createdAt: utcTimestamp(data, "created_at", "context_usage_report_v1"),
    requestId: optionalString(data, "request_id", "context_usage_report_v1"),
    threadId: optionalString(data, "thread_id", "context_usage_report_v1"),
    nodeName: requiredString(data, "node_name", "context_usage_report_v1"),
    llmNode: requiredString(data, "llm_node", "context_usage_report_v1"),
    provider: optionalString(data, "provider", "context_usage_report_v1"),
    model: requiredString(data, "model", "context_usage_report_v1"),
    inputEstimatedTokens,
    outputReservedTokens,
    usedTokens,
    maxContextTokens,
    availableTokens,
    usedRatio: nonNegativeNumber(data, "used_ratio", "context_usage_report_v1"),
    warningLevel,
    estimated: booleanValue(data, "estimated", "context_usage_report_v1"),
    tokenizerMode: requiredString(data, "tokenizer_mode", "context_usage_report_v1"),
    messageCount: integer(data, "message_count", "context_usage_report_v1", 0),
    schemaSizeChars,
    mainCategories,
    detailedCategories: requiredArray(
      data,
      "detailed_categories",
      "context_usage_report_v1",
    ).map((item) => parseUsageCategory(item)),
    overlapRollups: requiredArray(
      data,
      "overlap_rollups",
      "context_usage_report_v1",
    ).map((item) => parseUsageCategory(item)),
    segments,
    unclassifiedTokens: integer(
      data,
      "unclassified_tokens",
      "context_usage_report_v1",
      0,
    ),
    reconciliationOk: booleanValue(
      data,
      "reconciliation_ok",
      "context_usage_report_v1",
    ),
    reconciliationWarnings: stringArray(
      data,
      "reconciliation_warnings",
      "context_usage_report_v1",
    ),
  }
}

export function parseContextUsageReportError(value: unknown): ContextUsageReportError {
  const data = record(value, "context_usage_report_error_v1")
  literal(
    data,
    "schema_version",
    "context_usage_report_error_v1",
    "context_usage_report_error_v1",
  )
  return {
    schemaVersion: "context_usage_report_error_v1",
    manifestId: optionalString(data, "manifest_id", "context_usage_report_error_v1"),
    nodeName: optionalString(data, "node_name", "context_usage_report_error_v1"),
    llmNode: optionalString(data, "llm_node", "context_usage_report_error_v1"),
    provider: optionalString(data, "provider", "context_usage_report_error_v1"),
    model: optionalString(data, "model", "context_usage_report_error_v1"),
    reason: requiredString(data, "reason", "context_usage_report_error_v1"),
    warning: requiredString(data, "warning", "context_usage_report_error_v1"),
    errorType: requiredString(data, "error_type", "context_usage_report_error_v1"),
  }
}

export function parseBackgroundContextWindow(value: unknown): BackgroundContextWindow {
  const data = record(value, "background_context_window_v1")
  if (data.schema_version !== 1) {
    fail("background_context_window_v1", "schema_version must equal 1")
  }
  return {
    schemaVersion: 1,
    threadId: requiredString(data, "thread_id", "background_context_window_v1"),
    updatedAt: utcTimestamp(data, "updated_at", "background_context_window_v1"),
    lastManifestId: requiredString(
      data,
      "last_manifest_id",
      "background_context_window_v1",
    ),
    usedTokens: integer(data, "used_tokens", "background_context_window_v1", 0),
    maxContextTokens: integer(
      data,
      "max_context_tokens",
      "background_context_window_v1",
      0,
    ),
    usedRatio: nonNegativeNumber(data, "used_ratio", "background_context_window_v1"),
    messageCount: integer(data, "message_count", "background_context_window_v1", 0),
    sectionCount: integer(data, "section_count", "background_context_window_v1", 0),
    sectionNames: stringArray(data, "section_names", "background_context_window_v1"),
    conversationSummaryPresent: booleanValue(
      data,
      "conversation_summary_present",
      "background_context_window_v1",
    ),
    selectedMemoryCount: integer(
      data,
      "selected_memory_count",
      "background_context_window_v1",
      0,
    ),
    artifactSummaryCount: integer(
      data,
      "artifact_summary_count",
      "background_context_window_v1",
      0,
    ),
    evidenceSummaryCount: integer(
      data,
      "evidence_summary_count",
      "background_context_window_v1",
      0,
    ),
    ceBlockPresent: booleanValue(data, "ce_block_present", "background_context_window_v1"),
    structuredContractPresent: booleanValue(
      data,
      "structured_contract_present",
      "background_context_window_v1",
    ),
    manifestCount: integer(data, "manifest_count", "background_context_window_v1", 0),
    influenceEntryCount: integer(
      data,
      "influence_entry_count",
      "background_context_window_v1",
      0,
    ),
    influenceTokenEstimate: integer(
      data,
      "influence_token_estimate",
      "background_context_window_v1",
      0,
    ),
    influenceSourceNodeCount: integer(
      data,
      "influence_source_node_count",
      "background_context_window_v1",
      0,
    ),
    workspacePresent: booleanValue(
      data,
      "workspace_present",
      "background_context_window_v1",
    ),
    workspaceActiveSubject: optionalString(
      data,
      "workspace_active_subject",
      "background_context_window_v1",
    ),
    workspaceEvidenceSummaryCount: integer(
      data,
      "workspace_evidence_summary_count",
      "background_context_window_v1",
      0,
    ),
    workspaceGapCount: integer(
      data,
      "workspace_gap_count",
      "background_context_window_v1",
      0,
    ),
    workspaceArtifactCount: integer(
      data,
      "workspace_artifact_count",
      "background_context_window_v1",
      0,
    ),
    workspaceUpdatedAt: optionalString(
      data,
      "workspace_updated_at",
      "background_context_window_v1",
    ),
  }
}

export function parseThreadContextWindowV2(value: unknown): ThreadContextWindowV2 {
  const contract = "thread_context_window_v2"
  const data = record(value, contract)
  if (data.schema_version !== 2) fail(contract, "schema_version must equal 2")
  literal(data, "contract", "thread_context_window_v2", contract)
  return {
    schemaVersion: 2,
    contract: "thread_context_window_v2",
    threadId: requiredString(data, "thread_id", contract),
    updatedAt: utcTimestamp(data, "updated_at", contract),
    nextCallContextEstimate: parseNextCallContextEstimate(
      data.next_call_context_estimate,
    ),
    lastLlmCallUsage: parseLastLlmCallUsage(data.last_llm_call_usage),
    backgroundInventory: parseBackgroundInventory(data.background_inventory),
  }
}

function parseNextCallContextEstimate(value: unknown): NextCallContextEstimate {
  const contract = "next_call_context_estimate_v1"
  const data = record(value, contract)
  const stateFingerprint = requiredString(data, "state_fingerprint", contract)
  if (!stateFingerprint.startsWith("thread_context:v2:")) {
    fail(contract, "state_fingerprint prefix is invalid")
  }
  return {
    basis: enumValue(
      data,
      "basis",
      ["known_next_node", "thread_baseline"] as const,
      contract,
    ),
    confidence: enumValue(
      data,
      "confidence",
      ["high", "medium", "low"] as const,
      contract,
    ),
    estimated: booleanValue(data, "estimated", contract),
    estimatedAt: utcTimestamp(data, "estimated_at", contract),
    targetNode: optionalString(data, "target_node", contract),
    unknownSections: stringArray(data, "unknown_sections", contract),
    sections: parseContextSectionEstimates(data.sections, contract),
    estimatedInputTokens: integer(data, "estimated_input_tokens", contract, 0),
    estimatedOutputReservedTokens: integer(
      data,
      "estimated_output_reserved_tokens",
      contract,
      0,
    ),
    estimatedUsedTokens: integer(data, "estimated_used_tokens", contract, 0),
    maxContextTokens: integer(data, "max_context_tokens", contract, 0),
    usedRatio: nonNegativeNumber(data, "used_ratio", contract),
    tokenizerMode: requiredString(data, "tokenizer_mode", contract),
    stateFingerprint,
    reusedManifestStatistics: booleanValue(
      data,
      "reused_manifest_statistics",
      contract,
    ),
    knownSectionRatio: boundedRatio(data, "known_section_ratio", contract),
  }
}

function parseLastLlmCallUsage(value: unknown): LastLLMCallUsage {
  const contract = "last_llm_call_usage_v1"
  const data = record(value, contract)
  const present = booleanValue(data, "present", contract)
  return {
    present,
    reportId: optionalString(data, "report_id", contract),
    manifestId: optionalString(data, "manifest_id", contract),
    createdAt: present ? utcTimestamp(data, "created_at", contract) : "",
    nodeName: optionalString(data, "node_name", contract),
    llmNode: optionalString(data, "llm_node", contract),
    model: optionalString(data, "model", contract),
    inputEstimatedTokens: integer(data, "input_estimated_tokens", contract, 0),
    outputReservedTokens: integer(data, "output_reserved_tokens", contract, 0),
    usedTokens: integer(data, "used_tokens", contract, 0),
    maxContextTokens: integer(data, "max_context_tokens", contract, 0),
    usedRatio: nonNegativeNumber(data, "used_ratio", contract),
    estimated: booleanValue(data, "estimated", contract),
    tokenizerMode: optionalString(data, "tokenizer_mode", contract),
    sections: parseContextSectionEstimates(data.sections, contract),
  }
}

function parseBackgroundInventory(value: unknown): BackgroundInventory {
  const contract = "background_inventory_v1"
  const data = record(value, contract)
  return {
    conversationSummaryPresent: booleanValue(
      data,
      "conversation_summary_present",
      contract,
    ),
    selectedMemoryCount: integer(data, "selected_memory_count", contract, 0),
    evidenceSummaryCount: integer(data, "evidence_summary_count", contract, 0),
    artifactSummaryCount: integer(data, "artifact_summary_count", contract, 0),
    workspacePresent: booleanValue(data, "workspace_present", contract),
    workspaceActiveSubject: optionalString(
      data,
      "workspace_active_subject",
      contract,
    ),
    workspaceEvidenceSummaryCount: integer(
      data,
      "workspace_evidence_summary_count",
      contract,
      0,
    ),
    workspaceGapCount: integer(data, "workspace_gap_count", contract, 0),
    workspaceArtifactCount: integer(
      data,
      "workspace_artifact_count",
      contract,
      0,
    ),
    workspaceUpdatedAt: optionalString(data, "workspace_updated_at", contract),
    manifestCount: integer(data, "manifest_count", contract, 0),
    influenceEntryCount: integer(data, "influence_entry_count", contract, 0),
  }
}

function parseContextSectionEstimates(
  value: unknown,
  parentContract: string,
): ContextSectionEstimate[] {
  if (!Array.isArray(value) || value.length > 16) {
    fail(parentContract, "sections must be an array with at most 16 items")
  }
  return value.map((item) => {
    const contract = "context_section_estimate_v1"
    const data = record(item, contract)
    return {
      section: requiredString(data, "section", contract),
      source: requiredString(data, "source", contract),
      itemCount: integer(data, "item_count", contract, 0),
      messageCount: integer(data, "message_count", contract, 0),
      charCount: integer(data, "char_count", contract, 0),
      estimatedTokens: integer(data, "estimated_tokens", contract, 0),
      known: booleanValue(data, "known", contract),
    }
  })
}

function parseGraphNode(value: unknown): GraphManifestNode {
  const data = record(value, "graph_manifest_node")
  return {
    nodeId: requiredString(data, "node_id", "graph_manifest_node"),
    label: requiredString(data, "label", "graph_manifest_node"),
    description: optionalString(data, "description", "graph_manifest_node"),
    kind: requiredString(data, "kind", "graph_manifest_node"),
    group: requiredString(data, "group", "graph_manifest_node"),
    parent: optionalString(data, "parent", "graph_manifest_node"),
    workflow: optionalString(data, "workflow", "graph_manifest_node"),
    order: integer(data, "order", "graph_manifest_node", 0),
    stageRank: integer(data, "stage_rank", "graph_manifest_node", 0),
    visible: booleanValue(data, "visible", "graph_manifest_node"),
    logical: booleanValue(data, "logical", "graph_manifest_node"),
    activityRunning: optionalString(data, "activity_running", "graph_manifest_node"),
    activityCompleted: optionalString(data, "activity_completed", "graph_manifest_node"),
  }
}

function parseGraphEdge(value: unknown): GraphManifestEdge {
  const data = record(value, "graph_manifest_edge")
  return {
    edgeId: requiredString(data, "edge_id", "graph_manifest_edge"),
    source: requiredString(data, "source", "graph_manifest_edge"),
    target: requiredString(data, "target", "graph_manifest_edge"),
    kind: enumValue(data, "kind", ["graph", "logical"] as const, "graph_manifest_edge"),
    conditional: booleanValue(data, "conditional", "graph_manifest_edge"),
    label: optionalString(data, "label", "graph_manifest_edge"),
    workflow: optionalString(data, "workflow", "graph_manifest_edge"),
  }
}

function parseCapabilityMetadata(value: unknown): GraphCapabilityMetadata {
  const data = record(value, "graph_capability_metadata")
  return {
    resourceTypes: stringArray(data, "resource_types", "graph_capability_metadata"),
    contextPolicyMode: requiredString(
      data,
      "context_policy_mode",
      "graph_capability_metadata",
    ),
    checkpointerEnabled: booleanValue(
      data,
      "checkpointer_enabled",
      "graph_capability_metadata",
    ),
    checkpointerType: requiredString(
      data,
      "checkpointer_type",
      "graph_capability_metadata",
    ),
    physicalNodeCount: integer(
      data,
      "physical_node_count",
      "graph_capability_metadata",
      0,
    ),
    logicalNodeCount: integer(
      data,
      "logical_node_count",
      "graph_capability_metadata",
      0,
    ),
  }
}

function parseUsageCategory(value: unknown): ContextUsageCategory {
  const data = record(value, "context_usage_category")
  return {
    category: requiredString(data, "category", "context_usage_category"),
    estimatedTokens: integer(data, "estimated_tokens", "context_usage_category", 0),
    segmentCount: integer(data, "segment_count", "context_usage_category", 0),
    messageCount: integer(data, "message_count", "context_usage_category", 0),
  }
}

function parseUsageSegment(value: unknown): ContextUsageSegment {
  const data = record(value, "context_usage_segment")
  return {
    segmentId: requiredString(data, "segment_id", "context_usage_segment"),
    fingerprint: requiredString(data, "fingerprint", "context_usage_segment"),
    messageIndex: integer(data, "message_index", "context_usage_segment", 0),
    role: requiredString(data, "role", "context_usage_segment"),
    mainCategory: enumValue(
      data,
      "main_category",
      CONTEXT_MAIN_CATEGORIES,
      "context_usage_segment",
    ),
    detailedCategory: requiredString(
      data,
      "detailed_category",
      "context_usage_segment",
    ),
    charCount: integer(data, "char_count", "context_usage_segment", 0),
    estimatedTokens: integer(data, "estimated_tokens", "context_usage_segment", 0),
    provenance: parseSafeProvenance(data.provenance),
  }
}

function parseSafeActivityDetails(value: unknown): Record<string, SafeActivityDetail> {
  if (!isRecord(value)) return {}
  const result: Record<string, SafeActivityDetail> = {}
  for (const key of [...SAFE_ACTIVITY_DETAIL_KEYS].sort()) {
    const item = value[key]
    if (
      typeof item === "string" ||
      typeof item === "number" ||
      typeof item === "boolean" ||
      item === null
    ) {
      result[key] = item
    }
  }
  return result
}

function parseSafeProvenance(value: unknown): Record<string, string> {
  if (!isRecord(value)) return {}
  const result: Record<string, string> = {}
  for (const key of [...SAFE_PROVENANCE_KEYS].sort()) {
    if (typeof value[key] === "string") result[key] = value[key]
  }
  return result
}

function sumTokens(values: Array<{ estimatedTokens: number }>): number {
  return values.reduce((total, item) => total + item.estimatedTokens, 0)
}

function record(value: unknown, contract: string): Record<string, unknown> {
  if (!isRecord(value)) fail(contract, "expected an object")
  return value
}

function requiredArray(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): unknown[] {
  const value = data[key]
  if (!Array.isArray(value)) fail(contract, `${key} must be an array`)
  return value
}

function stringArray(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): string[] {
  const values = requiredArray(data, key, contract)
  if (!values.every((item) => typeof item === "string")) {
    fail(contract, `${key} must contain strings`)
  }
  return values as string[]
}

function requiredString(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): string {
  const value = data[key]
  if (typeof value !== "string" || !value.trim()) {
    fail(contract, `${key} is required`)
  }
  return value
}

function optionalString(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): string {
  const value = data[key]
  if (value === undefined || value === null || value === "") return ""
  if (typeof value !== "string") fail(contract, `${key} must be a string`)
  return value
}

function booleanValue(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): boolean {
  const value = data[key]
  if (typeof value !== "boolean") fail(contract, `${key} must be a boolean`)
  return value
}

function integer(
  data: Record<string, unknown>,
  key: string,
  contract: string,
  minimum: number,
): number {
  const value = data[key]
  if (!Number.isInteger(value) || (value as number) < minimum) {
    fail(contract, `${key} must be an integer >= ${minimum}`)
  }
  return value as number
}

function nonNegativeNumber(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): number {
  const value = data[key]
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    fail(contract, `${key} must be a non-negative number`)
  }
  return value
}

function boundedRatio(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): number {
  const value = nonNegativeNumber(data, key, contract)
  if (value > 1) fail(contract, `${key} must be <= 1`)
  return value
}

function optionalNonNegativeNumber(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): number | undefined {
  const value = data[key]
  if (value === undefined || value === null) return undefined
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    fail(contract, `${key} must be a non-negative number when present`)
  }
  return value
}

function literal<T extends string>(
  data: Record<string, unknown>,
  key: string,
  expected: T,
  contract: string,
): T {
  if (data[key] !== expected) fail(contract, `${key} must equal ${expected}`)
  return expected
}

function enumValue<const T extends readonly string[]>(
  data: Record<string, unknown>,
  key: string,
  values: T,
  contract: string,
): T[number] {
  const value = data[key]
  if (typeof value !== "string" || !values.includes(value)) {
    fail(contract, `${key} is invalid`)
  }
  return value as T[number]
}

function utcTimestamp(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): string {
  const value = requiredString(data, key, contract)
  validateUtcTimestamp(value, contract, key)
  return value
}

function validateUtcTimestamp(value: string, contract: string, key: string): void {
  if (!/(?:Z|\+00:00)$/.test(value) || Number.isNaN(Date.parse(value))) {
    fail(contract, `${key} must be a UTC ISO-8601 timestamp`)
  }
}

function versionedGraphId(
  data: Record<string, unknown>,
  key: string,
  contract: string,
): string {
  const value = requiredString(data, key, contract)
  if (!value.startsWith("graph:v1:")) fail(contract, `${key} prefix is invalid`)
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value))
}

function fail(contract: string, reason: string): never {
  throw new ContractParseError(contract, reason)
}
