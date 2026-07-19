import { describe, expect, it } from "vitest"

import {
  EvidenceProgressContractError,
  emptyEvidenceProgressTimeline,
  parseEvidenceProgressEvent,
  reduceEvidenceProgress,
  type EvidenceProgressEventV1,
} from "@/lib/evidence-progress"

const REQUEST_ID = "aaaaaaaa-0000-4000-8000-000000000001"
const THREAD_ID = "thread-1"
const SHA = "a".repeat(64)

function rawProgress(
  details: object,
  lifecycleKey: string,
  phaseStatus: "running" | "completed" | "failed",
  idCharacter: string,
): Record<string, unknown> {
  return {
    schema_version: "evidence_progress_v1",
    progress_id: `evidence-progress:v1:${idCharacter.repeat(64)}`,
    request_id: REQUEST_ID,
    thread_id: THREAD_ID,
    lifecycle_key: lifecycleKey,
    phase_status: phaseStatus,
    details,
  }
}

function progress(
  details: object,
  lifecycleKey: string,
  phaseStatus: "running" | "completed" | "failed",
  idCharacter: string,
): EvidenceProgressEventV1 {
  return parseEvidenceProgressEvent(
    rawProgress(details, lifecycleKey, phaseStatus, idCharacter),
  )
}

const plan = () =>
  progress(
    {
      stage: "evidence_orchestration.plan.accepted",
      requirement_count: 2,
      resource_count: 2,
      subject_count: 1,
      budget_max_rounds: 2,
      budget_max_tasks: 10,
    },
    "plan",
    "completed",
    "1",
  )

const roundStarted = (roundIndex = 0, idCharacter = "2") =>
  progress(
    {
      stage: "evidence_orchestration.round.started",
      round_index: roundIndex,
      task_count: roundIndex === 0 ? 2 : 1,
      local_task_count: roundIndex === 0 ? 1 : 0,
      web_task_count: 1,
      budget_used_tasks: roundIndex === 0 ? 2 : 3,
      budget_remaining_tasks: roundIndex === 0 ? 8 : 7,
    },
    `round:${roundIndex}`,
    "running",
    idCharacter,
  )

const sourceCompleted = (
  source: "local" | "web",
  roundIndex: number,
  taskCount: number,
  idCharacter: string,
) =>
  progress(
    {
      stage: "evidence_orchestration.source.completed",
      round_index: roundIndex,
      source,
      status: "completed",
      task_count: taskCount,
      candidate_count: taskCount,
      latency_ms: 12,
    },
    `round:${roundIndex}:source:${source}`,
    "completed",
    idCharacter,
  )

const sourceEmpty = (
  source: "local" | "web",
  roundIndex: number,
  taskCount: number,
  idCharacter: string,
) =>
  progress(
    {
      stage: "evidence_orchestration.source.empty",
      round_index: roundIndex,
      source,
      status: "empty",
      task_count: taskCount,
      latency_ms: 8,
      reason_code: "no_candidates",
    },
    `round:${roundIndex}:source:${source}`,
    "completed",
    idCharacter,
  )

const roundMerged = (roundIndex = 0, idCharacter = "2") =>
  progress(
    {
      stage: "evidence_orchestration.round.merged",
      round_index: roundIndex,
      local_candidate_count: roundIndex === 0 ? 1 : 0,
      web_candidate_count: 1,
      deduplicated_count: 0,
      ledger_count: roundIndex === 0 ? 2 : 3,
    },
    `round:${roundIndex}`,
    "completed",
    idCharacter,
  )

const coverage = (roundIndex = 0, idCharacter = "5") =>
  progress(
    {
      stage: "evidence_orchestration.coverage.judged",
      round_index: roundIndex,
      requirement_count: 2,
      complete_count: 1,
      partial_count: 0,
      missing_count: 1,
      accepted_evidence_count: roundIndex === 0 ? 2 : 3,
    },
    `round:${roundIndex}:coverage`,
    "completed",
    idCharacter,
  )

const evaluated = (roundIndex = 0, idCharacter = "6") =>
  progress(
    {
      stage: "evidence_orchestration.progress.evaluated",
      round_index: roundIndex,
      previous_complete_count: roundIndex === 0 ? 0 : 1,
      current_complete_count: roundIndex === 0 ? 1 : 1,
      previous_partial_count: 0,
      current_partial_count: 0,
      previous_missing_count: roundIndex === 0 ? 2 : 1,
      current_missing_count: 1,
      new_accepted_evidence_count: roundIndex === 0 ? 2 : 1,
      progressed: true,
      consecutive_no_progress_rounds: 0,
    },
    `round:${roundIndex}:progress`,
    "completed",
    idCharacter,
  )

const route = (
  status: "repair" | "terminal",
  roundIndex = 0,
  idCharacter = "7",
) =>
  progress(
    {
      stage: "evidence_orchestration.route.decided",
      round_index: roundIndex,
      status,
      reason_code: status === "repair" ? "targeted_required_gap_repair" : "budget_complete",
      next_local_task_count: 0,
      next_web_task_count: status === "repair" ? 1 : 0,
      budget_remaining_rounds: 2 - roundIndex,
      budget_remaining_tasks: roundIndex === 0 ? 8 : 7,
    },
    `round:${roundIndex}:route`,
    "completed",
    idCharacter,
  )

function reduce(...events: EvidenceProgressEventV1[]) {
  return events.reduce(reduceEvidenceProgress, emptyEvidenceProgressTimeline())
}

describe("parseEvidenceProgressEvent", () => {
  it("strictly parses every public stage variant", () => {
    const variants = [
      plan(),
      roundStarted(),
      sourceCompleted("local", 0, 1, "3"),
      sourceEmpty("web", 0, 1, "4"),
      progress(
        {
          stage: "evidence_orchestration.source.failed",
          round_index: 0,
          source: "web",
          status: "failed",
          task_count: 1,
          latency_ms: 5,
          reason_code: "provider_failed",
          error_type: "ProviderError",
        },
        "round:0:source:web",
        "failed",
        "5",
      ),
      roundMerged(),
      coverage(),
      evaluated(),
      route("repair"),
      progress(
        {
          stage: "evidence_orchestration.resource.assigned",
          round_index: 0,
          resource_type: "quiz",
          status: "ready",
          requirement_count: 1,
          assigned_evidence_count: 2,
          missing_requirement_count: 0,
        },
        "resource:quiz",
        "completed",
        "8",
      ),
      progress(
        {
          stage: "evidence_orchestration.resource.assigned",
          round_index: 0,
          resource_type: "review_doc",
          status: "fallback",
          requirement_count: 1,
          assigned_evidence_count: 1,
          missing_requirement_count: 1,
        },
        "resource:review_doc",
        "completed",
        "b",
      ),
      progress(
        {
          stage: "evidence_orchestration.terminal",
          orchestration_fingerprint: SHA,
          status: "partial_resources_ready",
          rounds_completed: 1,
          ready_resource_count: 1,
          blocked_resource_count: 1,
          total_search_tasks: 2,
          ledger_count: 2,
          reason_code: "budget_complete",
        },
        "terminal",
        "completed",
        "9",
      ),
      progress(
        {
          stage: "evidence_orchestration.failed",
          status: "failed",
          round_index: 0,
          source: "judge",
          error_type: "JudgeError",
          reason_code: "judge_failed",
          budget_used_tasks: 2,
          budget_remaining_tasks: 8,
        },
        "terminal",
        "failed",
        "a",
      ),
    ]
    expect(variants.map((item) => item.details.stage)).toHaveLength(13)
  })

  it("rejects extra, sensitive, drifted, and non-canonical values", () => {
    const base = rawProgress(plan().details, "plan", "completed", "1")
    expect(() => parseEvidenceProgressEvent({ ...base, query: "private" })).toThrow(
      EvidenceProgressContractError,
    )
    expect(() =>
      parseEvidenceProgressEvent({
        ...base,
        request_id: REQUEST_ID.toUpperCase(),
      }),
    ).toThrow("canonical UUID")
    expect(() =>
      parseEvidenceProgressEvent({
        ...base,
        details: { ...plan().details, subject_count: true },
      }),
    ).toThrow("subject_count")
    expect(() =>
      parseEvidenceProgressEvent({
        ...base,
        details: { ...plan().details, stage: "evidence_orchestration.plan.accepted", subject_count: "https://private.example" },
      }),
    ).toThrow("forbidden content")
  })

  it("enforces stage-specific status and business invariants", () => {
    const invalidStatus = rawProgress(
      { ...sourceEmpty("web", 0, 1, "4").details, status: "completed" },
      "round:0:source:web",
      "completed",
      "4",
    )
    expect(() => parseEvidenceProgressEvent(invalidStatus)).toThrow("status is invalid")

    const invalidProgress = rawProgress(
      { ...evaluated().details, progressed: false },
      "round:0:progress",
      "completed",
      "6",
    )
    expect(() => parseEvidenceProgressEvent(invalidProgress)).toThrow(
      "progressed does not match",
    )

    const invalidResource = rawProgress(
      {
        stage: "evidence_orchestration.resource.assigned",
        round_index: 0,
        resource_type: "quiz",
        status: "ready",
        requirement_count: 1,
        assigned_evidence_count: 0,
        missing_requirement_count: 1,
      },
      "resource:quiz",
      "completed",
      "8",
    )
    expect(() => parseEvidenceProgressEvent(invalidResource)).toThrow(
      "resource readiness contract",
    )

    const invalidFallback = rawProgress(
      {
        stage: "evidence_orchestration.resource.assigned",
        round_index: 0,
        resource_type: "review_doc",
        status: "fallback",
        requirement_count: 1,
        assigned_evidence_count: 0,
        missing_requirement_count: 1,
      },
      "resource:review_doc",
      "completed",
      "b",
    )
    expect(() => parseEvidenceProgressEvent(invalidFallback)).toThrow(
      "resource readiness contract",
    )
  })
})

describe("reduceEvidenceProgress", () => {
  it("closes a complete initial-search lifecycle with resource assignments", () => {
    const assignmentReady = progress(
      {
        stage: "evidence_orchestration.resource.assigned",
        round_index: 0,
        resource_type: "quiz",
        status: "ready",
        requirement_count: 1,
        assigned_evidence_count: 2,
        missing_requirement_count: 0,
      },
      "resource:quiz",
      "completed",
      "8",
    )
    const assignmentBlocked = progress(
      {
        stage: "evidence_orchestration.resource.assigned",
        round_index: 0,
        resource_type: "mindmap",
        status: "blocked",
        requirement_count: 1,
        assigned_evidence_count: 0,
        missing_requirement_count: 1,
      },
      "resource:mindmap",
      "completed",
      "9",
    )
    const terminal = progress(
      {
        stage: "evidence_orchestration.terminal",
        orchestration_fingerprint: SHA,
        status: "partial_resources_ready",
        rounds_completed: 1,
        ready_resource_count: 1,
        blocked_resource_count: 1,
        total_search_tasks: 2,
        ledger_count: 2,
        reason_code: "budget_complete",
      },
      "terminal",
      "completed",
      "a",
    )

    const state = reduce(
      plan(),
      roundStarted(),
      sourceCompleted("local", 0, 1, "3"),
      sourceEmpty("web", 0, 1, "4"),
      roundMerged(),
      coverage(),
      evaluated(),
      route("terminal"),
      assignmentReady,
      assignmentBlocked,
      terminal,
    )

    expect(state.terminal?.details.stage).toBe("evidence_orchestration.terminal")
    expect(Object.values(state.byId).some((item) => item.phaseStatus === "running")).toBe(false)
    expect(state.order).toHaveLength(10)
  })

  it("keeps fallback assignments deliverable without counting them as blocked", () => {
    const assignmentReady = progress(
      {
        stage: "evidence_orchestration.resource.assigned",
        round_index: 0,
        resource_type: "quiz",
        status: "ready",
        requirement_count: 1,
        assigned_evidence_count: 2,
        missing_requirement_count: 0,
      },
      "resource:quiz",
      "completed",
      "8",
    )
    const assignmentFallback = progress(
      {
        stage: "evidence_orchestration.resource.assigned",
        round_index: 0,
        resource_type: "mindmap",
        status: "fallback",
        requirement_count: 1,
        assigned_evidence_count: 1,
        missing_requirement_count: 1,
      },
      "resource:mindmap",
      "completed",
      "9",
    )
    const terminal = progress(
      {
        stage: "evidence_orchestration.terminal",
        orchestration_fingerprint: SHA,
        status: "partial_resources_ready",
        rounds_completed: 1,
        ready_resource_count: 1,
        blocked_resource_count: 0,
        total_search_tasks: 2,
        ledger_count: 2,
        reason_code: "budget_complete",
      },
      "terminal",
      "completed",
      "a",
    )

    const state = reduce(
      plan(),
      roundStarted(),
      sourceCompleted("local", 0, 1, "3"),
      sourceEmpty("web", 0, 1, "4"),
      roundMerged(),
      coverage(),
      evaluated(),
      route("terminal"),
      assignmentReady,
      assignmentFallback,
      terminal,
    )

    expect(state.terminal).toBe(terminal)
    const fallbackDetails = state.byId[assignmentFallback.progressId].details
    expect(fallbackDetails.stage).toBe("evidence_orchestration.resource.assigned")
    if (fallbackDetails.stage !== "evidence_orchestration.resource.assigned") {
      throw new Error("fallback assignment projection drifted")
    }
    expect(fallbackDetails.status).toBe("fallback")
  })

  it("requires contiguous repair rounds and stable round lifecycle identity", () => {
    let state = reduce(
      plan(),
      roundStarted(),
      sourceCompleted("local", 0, 1, "3"),
      sourceCompleted("web", 0, 1, "4"),
      roundMerged(),
      coverage(),
      evaluated(),
      route("repair"),
    )
    state = reduceEvidenceProgress(state, roundStarted(1, "8"))
    expect(state.byId[roundStarted(1, "8").progressId].phaseStatus).toBe("running")

    const rebound = { ...roundMerged(1, "9"), lifecycleKey: "round:1" }
    expect(() => reduceEvidenceProgress(state, rebound)).toThrow(
      "lifecycle_key was rebound",
    )
  })

  it("rejects merge before scheduled sources and terminal count drift", () => {
    const started = reduce(plan(), roundStarted())
    expect(() => reduceEvidenceProgress(started, roundMerged())).toThrow(
      "scheduled source result",
    )

    const beforeAssignments = reduce(
      plan(),
      roundStarted(),
      sourceCompleted("local", 0, 1, "3"),
      sourceCompleted("web", 0, 1, "4"),
      roundMerged(),
      coverage(),
      evaluated(),
      route("terminal"),
    )
    const terminal = progress(
      {
        stage: "evidence_orchestration.terminal",
        orchestration_fingerprint: SHA,
        status: "sufficient",
        rounds_completed: 1,
        ready_resource_count: 2,
        blocked_resource_count: 0,
        total_search_tasks: 2,
        ledger_count: 2,
        reason_code: "complete",
      },
      "terminal",
      "completed",
      "a",
    )
    expect(() => reduceEvidenceProgress(beforeAssignments, terminal)).toThrow(
      "resource counts",
    )
  })

  it("allows an explicit overall failure to close a running round", () => {
    const failed = progress(
      {
        stage: "evidence_orchestration.failed",
        status: "failed",
        round_index: 0,
        source: "web",
        error_type: "ProviderError",
        reason_code: "provider_failed",
        budget_used_tasks: 2,
        budget_remaining_tasks: 8,
      },
      "terminal",
      "failed",
      "a",
    )
    const state = reduce(plan(), roundStarted(), failed)
    expect(state.terminal).toBe(failed)
    expect(Object.values(state.byId).some((item) => item.phaseStatus === "running")).toBe(false)
  })
})
