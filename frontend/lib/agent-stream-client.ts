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

export interface ConsumeAgentStreamOptions {
  initialBody: ReadableStream<Uint8Array>
  onEvent: (event: AgentStreamEventV2) => void
  reconnect: AgentStreamReplay
  signal?: AbortSignal
}

export async function consumeAgentStreamV2({
  initialBody,
  onEvent,
  reconnect,
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
    body = await reconnect(streamId, lastEventId, signal)
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
