export const EVIDENCE_PROGRESS_SCHEMA_VERSION = "evidence_progress_v1" as const

export const EVIDENCE_PROGRESS_STAGES = [
  "evidence_orchestration.plan.accepted",
  "evidence_orchestration.round.started",
  "evidence_orchestration.source.completed",
  "evidence_orchestration.source.empty",
  "evidence_orchestration.source.failed",
  "evidence_orchestration.round.merged",
  "evidence_orchestration.coverage.judged",
  "evidence_orchestration.progress.evaluated",
  "evidence_orchestration.route.decided",
  "evidence_orchestration.resource.assigned",
  "evidence_orchestration.terminal",
  "evidence_orchestration.failed",
] as const

export type EvidenceProgressStage = (typeof EVIDENCE_PROGRESS_STAGES)[number]
export type EvidenceProgressPhaseStatus = "running" | "completed" | "failed"

interface EvidencePlanAcceptedDetails {
  stage: "evidence_orchestration.plan.accepted"
  requirement_count: number
  resource_count: number
  subject_count: number
  budget_max_rounds: number
  budget_max_tasks: number
}

interface EvidenceRoundStartedDetails {
  stage: "evidence_orchestration.round.started"
  round_index: number
  task_count: number
  local_task_count: number
  web_task_count: number
  budget_used_tasks: number
  budget_remaining_tasks: number
}

interface EvidenceSourceCompletedDetails {
  stage: "evidence_orchestration.source.completed"
  round_index: number
  source: "local" | "web"
  status: "completed"
  task_count: number
  candidate_count: number
  latency_ms: number
}

interface EvidenceSourceEmptyDetails {
  stage: "evidence_orchestration.source.empty"
  round_index: number
  source: "local" | "web"
  status: "empty"
  task_count: number
  latency_ms: number
  reason_code: string
}

interface EvidenceSourceFailedDetails {
  stage: "evidence_orchestration.source.failed"
  round_index: number
  source: "local" | "web"
  status: "failed"
  task_count: number
  latency_ms: number
  reason_code: string
  error_type: string
}

interface EvidenceRoundMergedDetails {
  stage: "evidence_orchestration.round.merged"
  round_index: number
  local_candidate_count: number
  web_candidate_count: number
  deduplicated_count: number
  ledger_count: number
}

interface EvidenceCoverageJudgedDetails {
  stage: "evidence_orchestration.coverage.judged"
  round_index: number
  requirement_count: number
  complete_count: number
  partial_count: number
  missing_count: number
  accepted_evidence_count: number
}

interface EvidenceProgressEvaluatedDetails {
  stage: "evidence_orchestration.progress.evaluated"
  round_index: number
  previous_complete_count: number
  current_complete_count: number
  previous_partial_count: number
  current_partial_count: number
  previous_missing_count: number
  current_missing_count: number
  new_accepted_evidence_count: number
  progressed: boolean
  consecutive_no_progress_rounds: number
}

interface EvidenceRouteDecidedDetails {
  stage: "evidence_orchestration.route.decided"
  round_index: number
  status: "repair" | "terminal"
  reason_code: string
  next_local_task_count: number
  next_web_task_count: number
  budget_remaining_rounds: number
  budget_remaining_tasks: number
}

interface EvidenceResourceAssignedDetails {
  stage: "evidence_orchestration.resource.assigned"
  round_index: number
  resource_type: string
  status: "ready" | "fallback" | "blocked"
  requirement_count: number
  assigned_evidence_count: number
  missing_requirement_count: number
}

interface EvidenceTerminalDetails {
  stage: "evidence_orchestration.terminal"
  orchestration_fingerprint: string
  status:
    | "sufficient"
    | "partial_resources_ready"
    | "insufficient_max_rounds"
    | "insufficient_no_progress"
    | "insufficient_empty_sources"
    | "blocked_insufficient_evidence"
  rounds_completed: number
  ready_resource_count: number
  blocked_resource_count: number
  total_search_tasks: number
  ledger_count: number
  reason_code: string
}

interface EvidenceFailedDetails {
  stage: "evidence_orchestration.failed"
  status: "failed"
  round_index: number
  source: "orchestration" | "local" | "web" | "judge" | "assignment"
  error_type: string
  reason_code: string
  budget_used_tasks: number
  budget_remaining_tasks: number
}

export type EvidenceProgressDetails =
  | EvidencePlanAcceptedDetails
  | EvidenceRoundStartedDetails
  | EvidenceSourceCompletedDetails
  | EvidenceSourceEmptyDetails
  | EvidenceSourceFailedDetails
  | EvidenceRoundMergedDetails
  | EvidenceCoverageJudgedDetails
  | EvidenceProgressEvaluatedDetails
  | EvidenceRouteDecidedDetails
  | EvidenceResourceAssignedDetails
  | EvidenceTerminalDetails
  | EvidenceFailedDetails

export interface EvidenceProgressEventV1 {
  schemaVersion: typeof EVIDENCE_PROGRESS_SCHEMA_VERSION
  progressId: string
  requestId: string
  threadId: string
  lifecycleKey: string
  phaseStatus: EvidenceProgressPhaseStatus
  details: EvidenceProgressDetails
}

export interface EvidenceProgressTimeline {
  requestId: string
  threadId: string
  order: string[]
  byId: Record<string, EvidenceProgressEventV1>
  terminal: EvidenceProgressEventV1 | null
  aborted: boolean
}

export class EvidenceProgressContractError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "EvidenceProgressContractError"
  }
}

const STAGES = new Set<string>(EVIDENCE_PROGRESS_STAGES)
const PROGRESS_ID = /^evidence-progress:v1:[a-f0-9]{64}$/
const LIFECYCLE_KEY = /^[a-z0-9:._-]{1,160}$/
const SAFE_CODE = /^[a-z][a-z0-9_.-]{0,79}$/
const SAFE_ERROR_TYPE = /^[A-Za-z][A-Za-z0-9_.]{0,79}$/
const SAFE_RESOURCE_TYPE = /^[a-z][a-z0-9_]{0,39}$/
const SHA256 = /^[a-f0-9]{64}$/
const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const SENSITIVE = /(?:https?:\/\/|www\.|bearer\s+|sk-(?:or-v1-)?[A-Za-z0-9_-]{8,}|(?:authorization|api[_-]?key|cookie|x-api-key)\s*[:=])/i

const DETAIL_KEYS: Record<EvidenceProgressStage, readonly string[]> = {
  "evidence_orchestration.plan.accepted": [
    "stage",
    "requirement_count",
    "resource_count",
    "subject_count",
    "budget_max_rounds",
    "budget_max_tasks",
  ],
  "evidence_orchestration.round.started": [
    "stage",
    "round_index",
    "task_count",
    "local_task_count",
    "web_task_count",
    "budget_used_tasks",
    "budget_remaining_tasks",
  ],
  "evidence_orchestration.source.completed": [
    "stage",
    "round_index",
    "source",
    "status",
    "task_count",
    "candidate_count",
    "latency_ms",
  ],
  "evidence_orchestration.source.empty": [
    "stage",
    "round_index",
    "source",
    "status",
    "task_count",
    "latency_ms",
    "reason_code",
  ],
  "evidence_orchestration.source.failed": [
    "stage",
    "round_index",
    "source",
    "status",
    "task_count",
    "latency_ms",
    "reason_code",
    "error_type",
  ],
  "evidence_orchestration.round.merged": [
    "stage",
    "round_index",
    "local_candidate_count",
    "web_candidate_count",
    "deduplicated_count",
    "ledger_count",
  ],
  "evidence_orchestration.coverage.judged": [
    "stage",
    "round_index",
    "requirement_count",
    "complete_count",
    "partial_count",
    "missing_count",
    "accepted_evidence_count",
  ],
  "evidence_orchestration.progress.evaluated": [
    "stage",
    "round_index",
    "previous_complete_count",
    "current_complete_count",
    "previous_partial_count",
    "current_partial_count",
    "previous_missing_count",
    "current_missing_count",
    "new_accepted_evidence_count",
    "progressed",
    "consecutive_no_progress_rounds",
  ],
  "evidence_orchestration.route.decided": [
    "stage",
    "round_index",
    "status",
    "reason_code",
    "next_local_task_count",
    "next_web_task_count",
    "budget_remaining_rounds",
    "budget_remaining_tasks",
  ],
  "evidence_orchestration.resource.assigned": [
    "stage",
    "round_index",
    "resource_type",
    "status",
    "requirement_count",
    "assigned_evidence_count",
    "missing_requirement_count",
  ],
  "evidence_orchestration.terminal": [
    "stage",
    "orchestration_fingerprint",
    "status",
    "rounds_completed",
    "ready_resource_count",
    "blocked_resource_count",
    "total_search_tasks",
    "ledger_count",
    "reason_code",
  ],
  "evidence_orchestration.failed": [
    "stage",
    "status",
    "round_index",
    "source",
    "error_type",
    "reason_code",
    "budget_used_tasks",
    "budget_remaining_tasks",
  ],
}

export function emptyEvidenceProgressTimeline(): EvidenceProgressTimeline {
  return { requestId: "", threadId: "", order: [], byId: {}, terminal: null, aborted: false }
}

export function parseEvidenceProgressEvent(value: unknown): EvidenceProgressEventV1 {
  const data = record(value, "evidence_progress_v1")
  exactKeys(data, [
    "schema_version",
    "progress_id",
    "request_id",
    "thread_id",
    "lifecycle_key",
    "phase_status",
    "details",
  ])
  if (data.schema_version !== EVIDENCE_PROGRESS_SCHEMA_VERSION) fail("schema_version is invalid")
  const progressId = requiredString(data.progress_id, "progress_id", 100)
  if (!PROGRESS_ID.test(progressId)) fail("progress_id is invalid")
  const requestId = requiredString(data.request_id, "request_id", 160)
  if (!CANONICAL_UUID.test(requestId)) fail("request_id must use canonical UUID text")
  const threadId = requiredString(data.thread_id, "thread_id", 160)
  if (threadId !== threadId.trim()) fail("thread_id must be stripped")
  const lifecycleKey = requiredString(data.lifecycle_key, "lifecycle_key", 160)
  if (!LIFECYCLE_KEY.test(lifecycleKey)) fail("lifecycle_key is invalid")
  const phaseStatus = data.phase_status
  if (phaseStatus !== "running" && phaseStatus !== "completed" && phaseStatus !== "failed") {
    fail("phase_status is invalid")
  }
  const details = parseDetails(data.details)
  if (lifecycleKey !== lifecycleKeyFor(details)) fail("lifecycle_key does not match details")
  if (phaseStatus !== phaseStatusFor(details.stage)) fail("phase_status does not match stage")
  return {
    schemaVersion: EVIDENCE_PROGRESS_SCHEMA_VERSION,
    progressId,
    requestId,
    threadId,
    lifecycleKey,
    phaseStatus,
    details,
  }
}

export function reduceEvidenceProgress(
  state: EvidenceProgressTimeline,
  event: EvidenceProgressEventV1,
): EvidenceProgressTimeline {
  if (state.requestId && (event.requestId !== state.requestId || event.threadId !== state.threadId)) {
    throw new EvidenceProgressContractError("evidence progress identity changed within one timeline")
  }
  if (state.terminal || state.aborted) {
    const existing = state.byId[event.progressId]
    if (existing && JSON.stringify(existing) === JSON.stringify(event)) return state
    throw new EvidenceProgressContractError("evidence progress arrived after terminal")
  }

  const existing = state.byId[event.progressId]
  const lifecycleExisting = Object.values(state.byId).find(
    (item) => item.lifecycleKey === event.lifecycleKey,
  )
  if (lifecycleExisting && lifecycleExisting.progressId !== event.progressId) {
    throw new EvidenceProgressContractError(
      "lifecycle_key was rebound to a different progress_id",
    )
  }
  const stage = event.details.stage
  if (stage === "evidence_orchestration.round.merged") {
    if (!existing || existing.details.stage !== "evidence_orchestration.round.started") {
      throw new EvidenceProgressContractError("round merge requires its matching start")
    }
  } else if (existing) {
    if (JSON.stringify(existing) === JSON.stringify(event)) return state
    throw new EvidenceProgressContractError("progress_id was reused by a different lifecycle event")
  }
  validateEvidenceTransition(state, event, existing)

  let byId = { ...state.byId, [event.progressId]: event }
  let terminal: EvidenceProgressEventV1 | null = null
  if (stage === "evidence_orchestration.terminal") {
    if (Object.values(byId).some((item) => item.phaseStatus === "running")) {
      throw new EvidenceProgressContractError("successful terminal cannot leave a running lifecycle")
    }
    terminal = event
  } else if (stage === "evidence_orchestration.failed") {
    byId = Object.fromEntries(
      Object.entries(byId).map(([id, item]) => [
        id,
        item.phaseStatus === "running" ? { ...item, phaseStatus: "failed" as const } : item,
      ]),
    )
    terminal = event
  }

  return {
    requestId: state.requestId || event.requestId,
    threadId: state.threadId || event.threadId,
    order: existing ? state.order : [...state.order, event.progressId],
    byId,
    terminal,
    aborted: false,
  }
}

function validateEvidenceTransition(
  state: EvidenceProgressTimeline,
  event: EvidenceProgressEventV1,
  existing: EvidenceProgressEventV1 | undefined,
): void {
  const details = event.details
  if (details.stage === "evidence_orchestration.failed") return
  if (details.stage === "evidence_orchestration.plan.accepted") {
    if (state.order.length !== 0) {
      throw new EvidenceProgressContractError("plan.accepted must be the first progress event")
    }
    return
  }

  const plan = findStage(state, "evidence_orchestration.plan.accepted")
  if (!plan) {
    throw new EvidenceProgressContractError("evidence progress requires plan.accepted first")
  }

  switch (details.stage) {
    case "evidence_orchestration.round.started": {
      if (Object.values(state.byId).some((item) => item.phaseStatus === "running")) {
        throw new EvidenceProgressContractError("a new round cannot start while another is running")
      }
      const rounds = roundEvents(state)
      const expectedRound = rounds.length === 0 ? 0 : Math.max(...rounds.map(roundOf)) + 1
      if (details.round_index !== expectedRound) {
        throw new EvidenceProgressContractError("evidence rounds must be contiguous from round 0")
      }
      if (details.round_index > plan.details.budget_max_rounds) {
        throw new EvidenceProgressContractError("evidence round exceeds the accepted round budget")
      }
      if (
        details.budget_used_tasks + details.budget_remaining_tasks !==
        plan.details.budget_max_tasks
      ) {
        throw new EvidenceProgressContractError("round task budget does not match the accepted plan")
      }
      if (details.round_index > 0) {
        const route = findRoundStage(
          state,
          "evidence_orchestration.route.decided",
          details.round_index - 1,
        )
        if (!route || route.details.status !== "repair") {
          throw new EvidenceProgressContractError("supplement round requires a prior repair route")
        }
        if (
          details.local_task_count !== route.details.next_local_task_count ||
          details.web_task_count !== route.details.next_web_task_count
        ) {
          throw new EvidenceProgressContractError("supplement round tasks do not match the repair route")
        }
      }
      return
    }
    case "evidence_orchestration.source.completed":
    case "evidence_orchestration.source.empty":
    case "evidence_orchestration.source.failed": {
      const round = findRoundLifecycle(state, details.round_index)
      if (!round || round.details.stage !== "evidence_orchestration.round.started") {
        throw new EvidenceProgressContractError("source progress requires a running matching round")
      }
      const expectedTasks =
        details.source === "local"
          ? round.details.local_task_count
          : round.details.web_task_count
      if (details.task_count !== expectedTasks) {
        throw new EvidenceProgressContractError("source task count does not match the round plan")
      }
      return
    }
    case "evidence_orchestration.round.merged": {
      if (!existing || existing.details.stage !== "evidence_orchestration.round.started") {
        throw new EvidenceProgressContractError("round merge requires its matching start")
      }
      for (const sourceName of ["local", "web"] as const) {
        const expectedTasks =
          sourceName === "local"
            ? existing.details.local_task_count
            : existing.details.web_task_count
        const sourceEvent = findSource(state, details.round_index, sourceName)
        if (expectedTasks > 0 && !sourceEvent) {
          throw new EvidenceProgressContractError("round merge requires every scheduled source result")
        }
        if (
          sourceEvent &&
          (sourceEvent.details.stage === "evidence_orchestration.source.failed" ||
            sourceEvent.details.task_count !== expectedTasks)
        ) {
          throw new EvidenceProgressContractError("round merge received an invalid source result")
        }
      }
      return
    }
    case "evidence_orchestration.coverage.judged": {
      const round = findRoundLifecycle(state, details.round_index)
      if (!round || round.details.stage !== "evidence_orchestration.round.merged") {
        throw new EvidenceProgressContractError("coverage judgment requires a merged round")
      }
      if (details.requirement_count !== plan.details.requirement_count) {
        throw new EvidenceProgressContractError("coverage requirement count changed from the plan")
      }
      return
    }
    case "evidence_orchestration.progress.evaluated":
      if (
        !findRoundStage(
          state,
          "evidence_orchestration.coverage.judged",
          details.round_index,
        )
      ) {
        throw new EvidenceProgressContractError("progress evaluation requires coverage judgment")
      }
      return
    case "evidence_orchestration.route.decided":
      if (
        !findRoundStage(
          state,
          "evidence_orchestration.progress.evaluated",
          details.round_index,
        )
      ) {
        throw new EvidenceProgressContractError("route decision requires progress evaluation")
      }
      return
    case "evidence_orchestration.resource.assigned": {
      const route = latestRoute(state)
      if (
        !route ||
        route.details.status !== "terminal" ||
        route.details.round_index !== details.round_index
      ) {
        throw new EvidenceProgressContractError("resource assignment requires a terminal route")
      }
      return
    }
    case "evidence_orchestration.terminal": {
      const route = latestRoute(state)
      if (!route || route.details.status !== "terminal") {
        throw new EvidenceProgressContractError("terminal progress requires a terminal route")
      }
      const assignments = Object.values(state.byId).filter(
        (item): item is EvidenceProgressEventV1 & { details: EvidenceResourceAssignedDetails } =>
          item.details.stage === "evidence_orchestration.resource.assigned",
      )
      const ready = assignments.filter((item) => item.details.status === "ready").length
      const blocked = assignments.filter((item) => item.details.status === "blocked").length
      if (
        assignments.length !== plan.details.resource_count ||
        ready !== details.ready_resource_count ||
        blocked !== details.blocked_resource_count
      ) {
        throw new EvidenceProgressContractError("terminal resource counts do not match assignments")
      }
      if (details.rounds_completed !== roundEvents(state).length) {
        throw new EvidenceProgressContractError("terminal round count does not match completed rounds")
      }
      const lastRound = findRoundLifecycle(state, route.details.round_index)
      if (
        !lastRound ||
        lastRound.details.stage !== "evidence_orchestration.round.merged" ||
        lastRound.details.ledger_count !== details.ledger_count
      ) {
        throw new EvidenceProgressContractError("terminal ledger count does not match the last round")
      }
    }
  }
}

function findStage<T extends EvidenceProgressStage>(
  state: EvidenceProgressTimeline,
  stage: T,
): (EvidenceProgressEventV1 & { details: Extract<EvidenceProgressDetails, { stage: T }> }) | undefined {
  return Object.values(state.byId).find(
    (item) => item.details.stage === stage,
  ) as
    | (EvidenceProgressEventV1 & {
        details: Extract<EvidenceProgressDetails, { stage: T }>
      })
    | undefined
}

function findRoundStage<T extends EvidenceProgressStage>(
  state: EvidenceProgressTimeline,
  stage: T,
  roundIndex: number,
): (EvidenceProgressEventV1 & { details: Extract<EvidenceProgressDetails, { stage: T }> }) | undefined {
  return Object.values(state.byId).find(
    (item) =>
      item.details.stage === stage &&
      "round_index" in item.details &&
      item.details.round_index === roundIndex,
  ) as
    | (EvidenceProgressEventV1 & {
        details: Extract<EvidenceProgressDetails, { stage: T }>
      })
    | undefined
}

function findRoundLifecycle(
  state: EvidenceProgressTimeline,
  roundIndex: number,
): EvidenceProgressEventV1 | undefined {
  return Object.values(state.byId).find(
    (item) => item.lifecycleKey === `round:${roundIndex}`,
  )
}

function findSource(
  state: EvidenceProgressTimeline,
  roundIndex: number,
  sourceName: "local" | "web",
): EvidenceProgressEventV1 & {
  details: EvidenceSourceCompletedDetails | EvidenceSourceEmptyDetails | EvidenceSourceFailedDetails
} | undefined {
  return Object.values(state.byId).find(
    (item) => item.lifecycleKey === `round:${roundIndex}:source:${sourceName}`,
  ) as
    | (EvidenceProgressEventV1 & {
        details:
          | EvidenceSourceCompletedDetails
          | EvidenceSourceEmptyDetails
          | EvidenceSourceFailedDetails
      })
    | undefined
}

function roundEvents(state: EvidenceProgressTimeline): EvidenceProgressEventV1[] {
  return Object.values(state.byId).filter((item) => /^round:\d+$/.test(item.lifecycleKey))
}

function roundOf(event: EvidenceProgressEventV1): number {
  const match = /^round:(\d+)$/.exec(event.lifecycleKey)
  if (!match) throw new EvidenceProgressContractError("round lifecycle key is invalid")
  return Number(match[1])
}

function latestRoute(
  state: EvidenceProgressTimeline,
): (EvidenceProgressEventV1 & { details: EvidenceRouteDecidedDetails }) | undefined {
  const routes = Object.values(state.byId).filter(
    (item): item is EvidenceProgressEventV1 & { details: EvidenceRouteDecidedDetails } =>
      item.details.stage === "evidence_orchestration.route.decided",
  )
  return routes.sort((left, right) => right.details.round_index - left.details.round_index)[0]
}

export function abortEvidenceProgress(
  state: EvidenceProgressTimeline,
): EvidenceProgressTimeline {
  if (state.order.length === 0 || state.terminal || state.aborted) return state
  return {
    ...state,
    aborted: true,
    byId: Object.fromEntries(
      Object.entries(state.byId).map(([id, item]) => [
        id,
        item.phaseStatus === "running" ? { ...item, phaseStatus: "failed" as const } : item,
      ]),
    ),
  }
}

export function evidenceProgressItems(state: EvidenceProgressTimeline): EvidenceProgressEventV1[] {
  return state.order.map((id) => state.byId[id]).filter(Boolean)
}

export function evidenceProgressTitle(event: EvidenceProgressEventV1): string {
  const details = event.details
  const stage = details.stage
  if (stage === "evidence_orchestration.plan.accepted") return "证据需求规划完成"
  if (stage === "evidence_orchestration.round.started") return `${roundLabel(number(details.round_index))}开始`
  if (
    stage === "evidence_orchestration.source.completed" ||
    stage === "evidence_orchestration.source.empty" ||
    stage === "evidence_orchestration.source.failed"
  ) {
    return `${roundLabel(number(details.round_index))} · ${details.source === "local" ? "本地检索" : "网络检索"}`
  }
  if (stage === "evidence_orchestration.round.merged") return `${roundLabel(number(details.round_index))}结果已合并`
  if (stage === "evidence_orchestration.coverage.judged") return `${roundLabel(number(details.round_index))}覆盖判断完成`
  if (stage === "evidence_orchestration.progress.evaluated") return `${roundLabel(number(details.round_index))}进展已评估`
  if (stage === "evidence_orchestration.route.decided") return details.status === "repair" ? "证据不足，继续补搜" : "证据检索结束"
  if (stage === "evidence_orchestration.resource.assigned") return `${String(details.resource_type)}：${details.status === "ready" ? "证据就绪" : "证据不足"}`
  if (stage === "evidence_orchestration.terminal") return "证据闭环已终止"
  return "证据闭环失败"
}

export function evidenceProgressSummary(event: EvidenceProgressEventV1): string {
  const details = event.details
  switch (details.stage) {
    case "evidence_orchestration.plan.accepted":
      return `${number(details.requirement_count)} 项要求 · ${number(details.resource_count)} 类资源 · ${number(details.subject_count)} 个学科`
    case "evidence_orchestration.round.started":
      return `本地 ${number(details.local_task_count)} / 网络 ${number(details.web_task_count)} · 剩余任务 ${number(details.budget_remaining_tasks)}`
    case "evidence_orchestration.source.completed":
      return `完成 ${number(details.task_count)} 个任务 · 获得 ${number(details.candidate_count)} 条候选`
    case "evidence_orchestration.source.empty":
    case "evidence_orchestration.source.failed":
      return `任务 ${number(details.task_count)} · ${String(details.reason_code)}`
    case "evidence_orchestration.round.merged":
      return `本地 ${number(details.local_candidate_count)} / 网络 ${number(details.web_candidate_count)} · 去重后 ${number(details.deduplicated_count)}`
    case "evidence_orchestration.coverage.judged":
      return `完整 ${number(details.complete_count)} · 部分 ${number(details.partial_count)} · 缺失 ${number(details.missing_count)}`
    case "evidence_orchestration.progress.evaluated":
      return `${details.progressed === true ? "有可测进展" : "无可测进展"} · 新增有效证据 ${number(details.new_accepted_evidence_count)}`
    case "evidence_orchestration.route.decided":
      return `${String(details.reason_code)} · 剩余轮次 ${number(details.budget_remaining_rounds)} · 剩余任务 ${number(details.budget_remaining_tasks)}`
    case "evidence_orchestration.resource.assigned":
      return `已分配 ${number(details.assigned_evidence_count)} 条 · 缺失要求 ${number(details.missing_requirement_count)}`
    case "evidence_orchestration.terminal":
      return `就绪资源 ${number(details.ready_resource_count)} · 阻断资源 ${number(details.blocked_resource_count)} · ${String(details.reason_code)}`
    case "evidence_orchestration.failed":
      return `${String(details.reason_code)} · ${String(details.error_type)}`
  }
}

function parseDetails(value: unknown): EvidenceProgressEventV1["details"] {
  const details = record(value, "details")
  const stageValue = details.stage
  if (typeof stageValue !== "string" || !STAGES.has(stageValue)) fail("details.stage is invalid")
  const stage = stageValue as EvidenceProgressStage
  exactKeys(details, [...DETAIL_KEYS[stage]])
  for (const [key, item] of Object.entries(details)) {
    if (typeof item === "string" && SENSITIVE.test(item)) fail(`${key} contains forbidden content`)
  }
  validateDetailValues(stage, details)
  return details as unknown as EvidenceProgressEventV1["details"]
}

function validateDetailValues(stage: EvidenceProgressStage, details: Record<string, unknown>): void {
  switch (stage) {
    case "evidence_orchestration.plan.accepted":
      counts(details, [
        "requirement_count",
        "resource_count",
        "subject_count",
        "budget_max_rounds",
        "budget_max_tasks",
      ])
      return
    case "evidence_orchestration.round.started": {
      roundIndex(details)
      counts(details, [
        "task_count",
        "local_task_count",
        "web_task_count",
        "budget_used_tasks",
        "budget_remaining_tasks",
      ])
      if (
        integer(details, "local_task_count") + integer(details, "web_task_count") !==
        integer(details, "task_count")
      ) {
        fail("round task partition is invalid")
      }
      return
    }
    case "evidence_orchestration.source.completed":
      source(details)
      exactString(details, "status", ["completed"])
      roundIndex(details)
      counts(details, ["task_count", "candidate_count"])
      latency(details)
      return
    case "evidence_orchestration.source.empty":
      source(details)
      exactString(details, "status", ["empty"])
      roundIndex(details)
      counts(details, ["task_count"])
      latency(details)
      safeCode(details, "reason_code")
      return
    case "evidence_orchestration.source.failed":
      source(details)
      exactString(details, "status", ["failed"])
      roundIndex(details)
      counts(details, ["task_count"])
      latency(details)
      safeCode(details, "reason_code")
      safeErrorType(details, "error_type")
      return
    case "evidence_orchestration.round.merged":
      roundIndex(details)
      counts(details, [
        "local_candidate_count",
        "web_candidate_count",
        "deduplicated_count",
        "ledger_count",
      ])
      return
    case "evidence_orchestration.coverage.judged":
      roundIndex(details)
      counts(details, [
        "requirement_count",
        "complete_count",
        "partial_count",
        "missing_count",
        "accepted_evidence_count",
      ])
      if (
        integer(details, "complete_count") +
          integer(details, "partial_count") +
          integer(details, "missing_count") !==
        integer(details, "requirement_count")
      ) {
        fail("coverage partition is invalid")
      }
      return
    case "evidence_orchestration.progress.evaluated": {
      roundIndex(details)
      counts(details, [
        "previous_complete_count",
        "current_complete_count",
        "previous_partial_count",
        "current_partial_count",
        "previous_missing_count",
        "current_missing_count",
        "new_accepted_evidence_count",
        "consecutive_no_progress_rounds",
      ])
      const progressed = strictBoolean(details, "progressed")
      const measurable =
        integer(details, "current_missing_count") < integer(details, "previous_missing_count") ||
        integer(details, "current_complete_count") > integer(details, "previous_complete_count") ||
        integer(details, "new_accepted_evidence_count") > 0
      if (progressed !== measurable) fail("progressed does not match measurable progress")
      return
    }
    case "evidence_orchestration.route.decided": {
      roundIndex(details)
      const status = exactString(details, "status", ["repair", "terminal"])
      safeCode(details, "reason_code")
      counts(details, [
        "next_local_task_count",
        "next_web_task_count",
        "budget_remaining_rounds",
        "budget_remaining_tasks",
      ])
      const tasks =
        integer(details, "next_local_task_count") + integer(details, "next_web_task_count")
      if ((status === "terminal" && tasks !== 0) || (status === "repair" && tasks === 0)) {
        fail("route task contract is invalid")
      }
      return
    }
    case "evidence_orchestration.resource.assigned": {
      roundIndex(details)
      const status = exactString(details, "status", ["ready", "fallback", "blocked"])
      const resourceType = requiredString(details.resource_type, "resource_type", 40)
      if (!SAFE_RESOURCE_TYPE.test(resourceType)) fail("resource_type is invalid")
      counts(details, [
        "requirement_count",
        "assigned_evidence_count",
        "missing_requirement_count",
      ])
      const missing = integer(details, "missing_requirement_count")
      if (missing > integer(details, "requirement_count")) {
        fail("missing requirements exceed requirement_count")
      }
      if (
        (status === "ready" && missing !== 0) ||
        (status === "fallback" && (missing === 0 || integer(details, "assigned_evidence_count") === 0)) ||
        (status === "blocked" && missing === 0)
      ) {
        fail("resource readiness contract is invalid")
      }
      return
    }
    case "evidence_orchestration.terminal": {
      const fingerprint = requiredString(
        details.orchestration_fingerprint,
        "orchestration_fingerprint",
        64,
      )
      if (!SHA256.test(fingerprint)) fail("orchestration_fingerprint is invalid")
      exactString(details, "status", [
        "sufficient",
        "partial_resources_ready",
        "insufficient_max_rounds",
        "insufficient_no_progress",
        "insufficient_empty_sources",
        "blocked_insufficient_evidence",
      ])
      counts(details, [
        "rounds_completed",
        "ready_resource_count",
        "blocked_resource_count",
        "total_search_tasks",
        "ledger_count",
      ])
      safeCode(details, "reason_code")
      return
    }
    case "evidence_orchestration.failed":
      exactString(details, "status", ["failed"])
      exactString(details, "source", [
        "orchestration",
        "local",
        "web",
        "judge",
        "assignment",
      ])
      roundIndex(details)
      safeErrorType(details, "error_type")
      safeCode(details, "reason_code")
      counts(details, ["budget_used_tasks", "budget_remaining_tasks"])
  }
}

function lifecycleKeyFor(details: EvidenceProgressEventV1["details"]): string {
  const stage = details.stage
  if (stage === "evidence_orchestration.plan.accepted") return "plan"
  if (stage === "evidence_orchestration.round.started" || stage === "evidence_orchestration.round.merged") return `round:${number(details.round_index)}`
  if (
    stage === "evidence_orchestration.source.completed" ||
    stage === "evidence_orchestration.source.empty" ||
    stage === "evidence_orchestration.source.failed"
  ) {
    return `round:${number(details.round_index)}:source:${String(details.source)}`
  }
  if (stage === "evidence_orchestration.coverage.judged") return `round:${number(details.round_index)}:coverage`
  if (stage === "evidence_orchestration.progress.evaluated") return `round:${number(details.round_index)}:progress`
  if (stage === "evidence_orchestration.route.decided") return `round:${number(details.round_index)}:route`
  if (stage === "evidence_orchestration.resource.assigned") return `resource:${String(details.resource_type)}`
  return "terminal"
}

function phaseStatusFor(stage: EvidenceProgressStage): EvidenceProgressPhaseStatus {
  if (stage === "evidence_orchestration.round.started") return "running"
  if (stage === "evidence_orchestration.source.failed" || stage === "evidence_orchestration.failed") return "failed"
  return "completed"
}

function roundLabel(index: number): string {
  return index === 0 ? "初始检索" : `第 ${index} 次补搜`
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

function requiredString(value: unknown, field: string, maxLength: number): string {
  if (typeof value !== "string" || value.length === 0) fail(`${field} is required`)
  if (value.length > maxLength) fail(`${field} exceeds ${maxLength} characters`)
  return value
}

function counts(data: Record<string, unknown>, fields: readonly string[]): void {
  for (const field of fields) integer(data, field)
}

function roundIndex(data: Record<string, unknown>): number {
  return integer(data, "round_index", 100)
}

function latency(data: Record<string, unknown>): number {
  return integer(data, "latency_ms", 86_400_000)
}

function integer(
  data: Record<string, unknown>,
  field: string,
  maximum = 100_000,
): number {
  const value = data[field]
  if (!Number.isInteger(value) || (value as number) < 0 || (value as number) > maximum) {
    fail(`${field} must be a bounded non-negative integer`)
  }
  return value as number
}

function strictBoolean(data: Record<string, unknown>, field: string): boolean {
  const value = data[field]
  if (typeof value !== "boolean") fail(`${field} must be a boolean`)
  return value
}

function exactString<T extends string>(
  data: Record<string, unknown>,
  field: string,
  values: readonly T[],
): T {
  const value = data[field]
  if (typeof value !== "string" || !values.includes(value as T)) {
    fail(`${field} is invalid`)
  }
  return value as T
}

function source(data: Record<string, unknown>): "local" | "web" {
  return exactString(data, "source", ["local", "web"])
}

function safeCode(data: Record<string, unknown>, field: string): string {
  const value = requiredString(data[field], field, 80)
  if (!SAFE_CODE.test(value)) fail(`${field} is invalid`)
  return value
}

function safeErrorType(data: Record<string, unknown>, field: string): string {
  const value = requiredString(data[field], field, 80)
  if (!SAFE_ERROR_TYPE.test(value)) fail(`${field} is invalid`)
  return value
}

function number(value: unknown): number {
  if (!Number.isInteger(value)) fail("validated evidence progress integer is invalid")
  return value as number
}

function fail(message: string): never {
  throw new EvidenceProgressContractError(message)
}
