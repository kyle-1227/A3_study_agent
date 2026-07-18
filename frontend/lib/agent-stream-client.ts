import {
  parseAgentStreamEvent,
  type AgentStreamEventV2,
} from "@/lib/agent-stream-contracts"
import { SSEParser, type SSEFrame } from "@/lib/sse-parser"

export type AgentStreamReplay = (
  streamId: string,
  lastEventId: string,
  signal?: AbortSignal,
) => Promise<ReadableStream<Uint8Array>>

export type AgentStreamReplayRecoveryReason = "transport" | "expired"

export class AgentStreamReplayRecoveryError extends Error {
  readonly reason: AgentStreamReplayRecoveryReason

  constructor(reason: AgentStreamReplayRecoveryReason) {
    super(`agent stream replay ${reason}`)
    this.name = "AgentStreamReplayRecoveryError"
    this.reason = reason
  }
}

export interface AgentStreamRecoveryIdentity {
  userId: string
  streamId: string
  requestId: string
  threadId: string
  lastEventId: string
}

export type AgentStreamTerminalStatus = "completed" | "failed" | "stopped"

export interface AgentStreamStatusRecoveryResult
  extends AgentStreamRecoveryIdentity {
  status: AgentStreamTerminalStatus
}

export type AgentStreamStatusRecovery = (
  identity: AgentStreamRecoveryIdentity,
  signal?: AbortSignal,
) => Promise<AgentStreamStatusRecoveryResult>

export function validateAgentStreamThreadStatusIdentity(
  statusThreadId: unknown,
  requestedThreadId: string,
  recoveryIdentity?: AgentStreamRecoveryIdentity,
): void {
  if (
    !requestedThreadId ||
    statusThreadId !== requestedThreadId ||
    (recoveryIdentity !== undefined &&
      recoveryIdentity.threadId !== requestedThreadId)
  ) {
    throw new Error("thread status identity mismatch")
  }
}

export function validateAgentStreamRecoveryIdentity(
  observed: {
    readonly userId: unknown
    readonly threadId: unknown
    readonly requestId: unknown
  },
  expected: Pick<
    AgentStreamRecoveryIdentity,
    "userId" | "threadId" | "requestId"
  >,
): void {
  if (
    !expected.userId ||
    !expected.threadId ||
    !expected.requestId ||
    observed.userId !== expected.userId ||
    observed.threadId !== expected.threadId ||
    observed.requestId !== expected.requestId
  ) {
    throw new Error("thread status recovery identity mismatch")
  }
}

export function classifyAgentStreamThreadStatusRecovery(
  statusPayload: unknown,
  identity: AgentStreamRecoveryIdentity,
  currentUserId: unknown,
  latestActivityRequestId: string,
): AgentStreamStatusRecoveryResult {
  if (
    !statusPayload ||
    typeof statusPayload !== "object" ||
    Array.isArray(statusPayload)
  ) {
    throw new Error("thread status recovery contract is invalid")
  }
  const status = statusPayload as Record<string, unknown>
  if (status.schema_version !== "run_control_v1") {
    throw new Error("thread status recovery schema is invalid")
  }
  validateAgentStreamThreadStatusIdentity(
    status.thread_id,
    identity.threadId,
    identity,
  )
  validateAgentStreamRecoveryIdentity(
    {
      userId: currentUserId,
      threadId: status.thread_id,
      requestId: latestActivityRequestId,
    },
    identity,
  )
  if (
    status.run_status !== "completed" &&
    status.run_status !== "failed" &&
    status.run_status !== "stopped"
  ) {
    if (typeof status.run_status !== "string") {
      throw new Error("thread status recovery run status is invalid")
    }
    throw new Error("thread status recovery is not terminal")
  }
  return {
    ...identity,
    status: status.run_status,
  }
}

export interface ConsumeAgentStreamOptions {
  initialBody: ReadableStream<Uint8Array>
  onEvent: (event: AgentStreamEventV2) => void
  reconnect: AgentStreamReplay
  recoverStatus?: AgentStreamStatusRecovery
  recoveryUserId?: string | null
  signal?: AbortSignal
}

export async function consumeAgentStreamV2({
  initialBody,
  onEvent,
  reconnect,
  recoverStatus,
  recoveryUserId,
  signal,
}: ConsumeAgentStreamOptions): Promise<void> {
  let body = initialBody
  let streamId = ""
  let lastEventId = ""
  let retryMs: number | null = null
  let streamDone = false
  let requestId = ""
  let threadId = ""
  let lastSequence = 0
  let statusRecoveryAttempted = false
  const seenEvents = new Map<number, string>()

  const dispatchFrames = (frames: SSEFrame[]) => {
    for (const frame of frames) {
      const event = parseAgentStreamEvent(JSON.parse(frame.data))
      if (frame.event !== event.type) {
        throw new Error("SSE event field does not match payload type")
      }
      if (frame.id !== event.eventId) {
        throw new Error("SSE id does not match payload event_id")
      }
      if (!streamId) {
        if (event.type !== "stream_start" || event.sequence !== 1) {
          throw new Error("first stream event must be stream_start sequence 1")
        }
        streamId = event.streamId
        requestId = event.requestId
        threadId = event.threadId
      } else if (
        event.streamId !== streamId ||
        event.requestId !== requestId ||
        event.threadId !== threadId
      ) {
        throw new Error("stream event identity changed")
      }
      const serialized = JSON.stringify(event)
      if (event.sequence <= lastSequence) {
        if (seenEvents.get(event.sequence) !== serialized) {
          throw new Error("replayed stream sequence conflicts with the original event")
        }
        continue
      }
      if (event.sequence !== lastSequence + 1) {
        throw new Error(
          `stream sequence gap: expected ${lastSequence + 1}, received ${event.sequence}`,
        )
      }
      lastSequence = event.sequence
      seenEvents.set(event.sequence, serialized)
      lastEventId = event.eventId
      if (frame.retry !== undefined) retryMs = frame.retry
      onEvent(event)
      if (event.type === "stream_done") streamDone = true
    }
  }

  while (!streamDone) {
    const parser = new SSEParser()
    const reader = body.getReader()
    let transportDisconnected = false
    try {
      while (true) {
        let result: ReadableStreamReadResult<Uint8Array>
        try {
          result = await reader.read()
        } catch (error) {
          if (signal?.aborted) throw error
          transportDisconnected = true
          break
        }
        if (result.done) break
        dispatchFrames(parser.feed(result.value))
      }
      if (!transportDisconnected) dispatchFrames(parser.finish())
    } finally {
      reader.releaseLock()
    }
    if (streamDone) return
    if (!streamId || !lastEventId || retryMs === null) {
      throw new Error("stream ended before replay identity was established")
    }
    await waitForReconnect(retryMs, signal)
    try {
      body = await reconnect(streamId, lastEventId, signal)
    } catch (error) {
      if (
        signal?.aborted ||
        !(error instanceof AgentStreamReplayRecoveryError) ||
        !recoverStatus ||
        statusRecoveryAttempted
      ) {
        throw error
      }
      statusRecoveryAttempted = true
      if (
        typeof recoveryUserId !== "string" ||
        recoveryUserId.length === 0 ||
        recoveryUserId.trim() !== recoveryUserId
      ) {
        throw new Error("agent stream status recovery user identity is unavailable")
      }
      const identity: AgentStreamRecoveryIdentity = {
        userId: recoveryUserId,
        streamId,
        requestId,
        threadId,
        lastEventId,
      }
      const recovery = await recoverStatus(identity, signal)
      validateStatusRecovery(recovery, identity)
      return
    }
  }
}

function validateStatusRecovery(
  recovery: AgentStreamStatusRecoveryResult,
  identity: AgentStreamRecoveryIdentity,
): void {
  if (
    !recovery ||
    typeof recovery !== "object" ||
    (recovery.status !== "completed" &&
      recovery.status !== "failed" &&
      recovery.status !== "stopped")
  ) {
    throw new Error("thread status recovery contract is invalid")
  }
  validateAgentStreamRecoveryIdentity(recovery, identity)
  if (
    recovery.streamId !== identity.streamId ||
    recovery.lastEventId !== identity.lastEventId
  ) {
    throw new Error("thread status recovery identity mismatch")
  }
}

function waitForReconnect(delayMs: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(new DOMException("Aborted", "AbortError"))
  return new Promise((resolve, reject) => {
    let timeout: ReturnType<typeof globalThis.setTimeout>
    const handleAbort = () => {
      globalThis.clearTimeout(timeout)
      signal?.removeEventListener("abort", handleAbort)
      reject(new DOMException("Aborted", "AbortError"))
    }
    timeout = globalThis.setTimeout(() => {
      signal?.removeEventListener("abort", handleAbort)
      resolve()
    }, delayMs)
    signal?.addEventListener("abort", handleAbort, { once: true })
  })
}
