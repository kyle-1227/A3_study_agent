"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { LeftSidebar } from "@/components/left-sidebar"
import { useUser } from "@/hooks/use-user"
import { RightPanel, type LogEntry } from "@/components/right-panel"
import {
  ChatArea,
  type Message,
  type ResourceGenerationStatus,
  type ResourceGenerationStep,
} from "@/components/chat-area"
import { PlanReview } from "@/components/plan-review"
import {
  isCompletedWithoutResourceDiagnostic,
  mergeResourceFinalIntoMessage,
  parseResourceFinalEvent,
  resourceFinalDedupeKey,
  resourceFinalOutcome,
  resourceMessageIdFromDedupeKey,
  type ResourceFinalEvent,
} from "@/lib/resource-final"
import { mergeActivityTimeline } from "@/lib/activity-reducer"
import {
  attachQAFinalToMessages,
  parseQAFinalEvent,
  qaFinalDedupeKey,
  type QAFinalEventV1,
} from "@/lib/qa-final"
import {
  applyContextUsageError,
  applyContextUsageReport,
  beginContextUsageUpdate,
  EMPTY_CONTEXT_USAGE_STATE,
  finishContextUsageUpdate,
  restoreContextUsageReport,
} from "@/lib/context-usage-state"
import {
  attachActivityToAssistantMessage,
  restoreActivitiesToMessages,
} from "@/lib/message-activity"
import {
  ContractParseError,
  parseActivityEvent,
  parseActivityTimeline,
  parseBackgroundContextWindow,
  parseContextUsageReport,
  parseContextUsageReportError,
  parseGraphManifest,
  parseGraphManifestRef,
  parseGraphManifestUnavailable,
  parseFrontendPerformanceCapability,
  parseStreamContext,
  parseThreadContextWindowV2,
  type ActivityEvent,
  type BackgroundContextWindow,
  type ContextUsageReportError,
  type GraphManifest,
  type GraphManifestUnavailable,
  type ThreadContextWindowV2,
} from "@/lib/observability-contracts"
import { FrontendPerformanceTracker } from "@/lib/frontend-performance"
import { requirePublicApiBaseUrl } from "@/lib/public-config"
import {
  beginStreamLifecycle,
  IDLE_STREAM_LIFECYCLE,
  reduceStreamLifecycle,
  type StreamLifecycleState,
} from "@/lib/stream-lifecycle"
import { mergeSafeFailureContent } from "@/lib/assistant-failure"
import {
  type AgentStreamEventV2,
} from "@/lib/agent-stream-contracts"
import { consumeAgentStreamV2 } from "@/lib/agent-stream-client"
import {
  parseThreadContextWindowV3,
  type ThreadContextWindowV3,
} from "@/lib/thread-context-window-v3"
import {
  LiveTurnSequenceError,
  reduceLiveTurn,
  type LiveTurnState,
} from "@/lib/live-turn"

const API_BASE_URL = requirePublicApiBaseUrl()

const A3_CHAT_HISTORY_KEY = "a3_chat_history"
const A3_CURRENT_CHAT_ID_KEY = "a3_current_chat_id"
const A3_CURRENT_THREAD_ID_KEY = "a3_current_thread_id"
const A3_MESSAGES_KEY_PREFIX = "a3_messages:"

type ChatHistoryItem = {
  id: string
  threadId: string
  title: string
  updatedAt?: number
}

type MemoryConfirmationState = {
  question: string
  reason?: string
  selectedMemoryCount?: number
}

type ProfileCompletionField = {
  key: string
  label: string
  required?: boolean
  max_chars?: number
}

type ProfileCompletionState = {
  title: string
  fields: ProfileCompletionField[]
}

const initialChatHistory: ChatHistoryItem[] = []

function isBrowser(): boolean {
  return typeof window !== "undefined"
}

function readJSON<T>(key: string, fallback: T): T {
  if (!isBrowser()) return fallback
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : fallback
  } catch {
    return fallback
  }
}

function writeJSON(key: string, value: unknown) {
  if (!isBrowser()) return
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Ignore storage quota / serialization issues; chat should keep working in memory.
  }
}

function removeStorageItem(key: string) {
  if (!isBrowser()) return
  localStorage.removeItem(key)
}

function messageStorageKey(threadId: string): string {
  return `${A3_MESSAGES_KEY_PREFIX}${threadId}`
}

function normalizeChatHistory(raw: unknown): ChatHistoryItem[] {
  if (!Array.isArray(raw)) return []
  return raw
    .map((item: any) => {
      const threadId = typeof item?.threadId === "string" ? item.threadId : typeof item?.id === "string" ? item.id : ""
      const title = typeof item?.title === "string" && item.title.trim() ? item.title : "新对话"
      return threadId
        ? {
            id: threadId,
            threadId,
            title,
            updatedAt: typeof item?.updatedAt === "number" ? item.updatedAt : undefined,
          }
        : null
    })
    .filter(Boolean) as ChatHistoryItem[]
}

function normalizeMessages(raw: unknown): Message[] {
  if (!Array.isArray(raw)) return []
  return raw
    .filter((item: any) => {
      return (
        item &&
        typeof item.id === "string" &&
        (item.role === "user" || item.role === "assistant") &&
        typeof item.content === "string"
      )
    })
    .map((item: Message) => {
      if (!item.activities) return item
      const parsed = parseActivityTimeline(item.activities)
      return { ...item, activities: parsed.items }
    })
}

function makeChatTitle(content: string): string {
  const compact = content.trim().replace(/\s+/g, " ")
  if (!compact) return "新对话"
  return compact.slice(0, 30) + (compact.length > 30 ? "..." : "")
}

function timestamp(): string {
  return new Date().toLocaleTimeString("en-GB", { hour12: false })
}

function contractFailureReason(error: unknown): string {
  if (error instanceof ContractParseError) return error.reason
  return error instanceof Error ? error.message : "unknown_contract_error"
}

function graphManifestFailure(error: unknown): GraphManifestUnavailable {
  return {
    schemaVersion: "graph_manifest_error_v1",
    error: "graph_manifest_unavailable",
    reason: contractFailureReason(error).slice(0, 160) || "graph_manifest_contract_invalid",
    errorType: error instanceof Error ? error.name.slice(0, 120) : "ContractError",
  }
}

function contextUsageContractFailure(error: unknown): ContextUsageReportError {
  return {
    schemaVersion: "context_usage_report_error_v1",
    manifestId: "",
    nodeName: "",
    llmNode: "",
    provider: "",
    model: "",
    reason: "context_usage_report_contract_invalid",
    warning: contractFailureReason(error).slice(0, 200),
    errorType: error instanceof Error ? error.name.slice(0, 120) : "ContractError",
  }
}

function mapProfileCompletionRequest(data: any): ProfileCompletionState | null {
  if (!data || typeof data !== "object") return null
  const fields = Array.isArray(data.fields)
    ? data.fields
        .filter((field: any) => field && typeof field.key === "string")
        .map((field: any) => ({
          key: field.key,
          label: typeof field.label === "string" && field.label.trim() ? field.label : field.key,
          required: field.required === true,
          max_chars: typeof field.max_chars === "number" ? field.max_chars : undefined,
        }))
    : []
  if (fields.length === 0) return null
  return {
    title:
      typeof data.title === "string" && data.title.trim()
        ? data.title
        : "生成学习计划前需要补充学习信息",
    fields,
  }
}

function getAuthHeaders(): Record<string, string> {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("demo_access_token")
    if (token) return { "X-Access-Token": token }
  }
  return {}
}

function createInitialResourceStatus(): ResourceGenerationStatus {
  return {
    state: "running",
    summary: "正在解析学习需求，准备调度多智能体生成个性化学习资源。",
    steps: [],
    tokenUsage: { input: 0, output: 0, total: 0 },
  }
}

function isResourceActivity(activity: ActivityEvent): boolean {
  return (
    activity.parent === "resource_worker" ||
    typeof activity.safeDetails.resource_type === "string"
  )
}

function ProfileCompletionDialog({
  request,
  isSubmitting,
  onSubmit,
}: {
  request: ProfileCompletionState
  isSubmitting: boolean
  onSubmit: (completion: Record<string, string>) => void
}) {
  const [values, setValues] = useState<Record<string, string>>({})
  const requiredMissing = request.fields.some((field) => field.required && !values[field.key]?.trim())

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(36,48,39,0.22)] px-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="profile-completion-title"
    >
      <form
        className="w-full max-w-[560px] rounded-2xl border border-border bg-card p-5 text-card-foreground shadow-[0_12px_30px_rgba(36,48,39,0.12)]"
        onSubmit={(event) => {
          event.preventDefault()
          onSubmit(values)
        }}
      >
        <div className="mb-4">
          <p id="profile-completion-title" className="text-base font-semibold text-foreground">
            {request.title}
          </p>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            请补充必要学习信息，提交后会从当前节点继续生成学习计划。
          </p>
        </div>
        <div className="max-h-[56vh] space-y-3 overflow-y-auto pr-1">
          {request.fields.map((field) => (
            <label key={field.key} className="block">
              <span className="mb-1 flex items-center justify-between gap-2 text-sm font-medium text-foreground">
                <span>{field.label}</span>
                <span className="text-xs text-muted-foreground">{field.required ? "必填" : "可选"}</span>
              </span>
              <textarea
                value={values[field.key] || ""}
                onChange={(event) =>
                  setValues((prev) => ({
                    ...prev,
                    [field.key]: event.target.value.slice(0, field.max_chars || 512),
                  }))
                }
                required={field.required}
                rows={2}
                className="w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition focus:border-primary"
              />
            </label>
          ))}
        </div>
        <div className="mt-5 flex justify-end">
          <button
            type="submit"
            className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-[var(--primary-deep)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={isSubmitting || requiredMissing}
          >
            继续生成
          </button>
        </div>
      </form>
    </div>
  )
}

export default function Home() {
  const [chatHistory, setChatHistory] = useState(initialChatHistory)
  const [selectedChatId, setSelectedChatId] = useState<string | undefined>()
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [storageReady, setStorageReady] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([
    { type: "info", message: "[INFO] 系统已初始化。", ts: "--:--:--" },
  ])
  const [activityTimeline, setActivityTimeline] = useState<ActivityEvent[]>([])
  const [tokenUsage, setTokenUsage] = useState({ input: 0, output: 0, total: 0 })
  const [contextUsageState, setContextUsageState] = useState(EMPTY_CONTEXT_USAGE_STATE)
  const [backgroundContextWindow, setBackgroundContextWindow] = useState<BackgroundContextWindow | null>(null)
  const [threadContextWindowV2, setThreadContextWindowV2] = useState<ThreadContextWindowV2 | null>(null)
  const [threadContextWindowV3, setThreadContextWindowV3] = useState<ThreadContextWindowV3 | null>(null)
  const [graphManifest, setGraphManifest] = useState<GraphManifest | null>(null)
  const [graphManifestError, setGraphManifestError] = useState<GraphManifestUnavailable | null>(null)
  const [graphManifestLoading, setGraphManifestLoading] = useState(false)
  const [currentRequestId, setCurrentRequestId] = useState("")
  const [canContinue, setCanContinue] = useState(false)
  const [stopPending, setStopPending] = useState(false)
  const [liveTurn, setLiveTurn] = useState<LiveTurnState | null>(null)

  // HIL state
  const [isInterrupted, setIsInterrupted] = useState(false)
  const [interruptDraft, setInterruptDraft] = useState("")
  const [isResuming, setIsResuming] = useState(false)
  const [memoryConfirmation, setMemoryConfirmation] = useState<MemoryConfirmationState | null>(null)
  const [isMemoryConfirming, setIsMemoryConfirming] = useState(false)
  const [profileCompletion, setProfileCompletion] = useState<ProfileCompletionState | null>(null)
  const [isProfileCompleting, setIsProfileCompleting] = useState(false)
  const threadIdRef = useRef<string | null>(null)
  const router = useRouter()
  const { userId, nickname, hasProfile, isLoading: userLoading, startOnboarding } = useUser()

  // Redirect to onboarding if user exists but has no profile
  useEffect(() => {
    if (userLoading || !storageReady) return
    if (userId && !hasProfile) {
      router.push("/onboarding")
    }
  }, [userId, hasProfile, userLoading, storageReady, router])
  const assistantMessageIdRef = useRef<string>("")
  const pendingChatTitleRef = useRef<string>("")
  const streamHadErrorRef = useRef(false)
  const streamLifecycleRef = useRef<StreamLifecycleState>(IDLE_STREAM_LIFECYCLE)
  const graphManifestVersionRef = useRef("")
  const resourceFinalDedupeRef = useRef<Set<string>>(new Set())
  const qaFinalDedupeRef = useRef<Set<string>>(new Set())
  const requestIdRef = useRef("")
  const abortControllerRef = useRef<AbortController | null>(null)
  const stopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const frontendPerformanceTrackerRef = useRef<FrontendPerformanceTracker | null>(null)
  const liveTurnRef = useRef<LiveTurnState | null>(null)

  const clearStopTimeout = useCallback(() => {
    if (stopTimeoutRef.current) {
      clearTimeout(stopTimeoutRef.current)
      stopTimeoutRef.current = null
    }
  }, [])

  const setActiveThreadId = useCallback((threadId: string | null) => {
    threadIdRef.current = threadId
    setCurrentThreadId(threadId)
    if (!threadId) {
      removeStorageItem(A3_CURRENT_THREAD_ID_KEY)
      return
    }
    if (isBrowser()) localStorage.setItem(A3_CURRENT_THREAD_ID_KEY, threadId)
  }, [])

  const beginFrontendPerformanceTracking = useCallback(() => {
    const tracker = new FrontendPerformanceTracker(window.performance)
    frontendPerformanceTrackerRef.current = tracker
    return tracker
  }, [])

  const deliverFrontendPerformance = useCallback((tracker: FrontendPerformanceTracker) => {
    void tracker
      .deliver(fetch, API_BASE_URL)
      .then((result) => {
        if (result.status !== "incomplete") return
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[PERF] Browser performance sample incomplete: ${result.reason}`,
            ts: timestamp(),
          },
        ])
      })
      .catch((error: unknown) => {
        const errorType = error instanceof Error ? error.name : "PerformanceDeliveryError"
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[PERF] Browser performance sample incomplete: ${errorType}`,
            ts: timestamp(),
          },
        ])
      })
  }, [])

  const setActiveRequestId = useCallback((requestId: string) => {
    requestIdRef.current = requestId
    setCurrentRequestId(requestId)
  }, [])

  const beginRequestStream = useCallback((requestId: string) => {
    liveTurnRef.current = null
    setLiveTurn(null)
    setThreadContextWindowV3((current) =>
      current ? { ...current, updating: true } : null,
    )
    setActiveRequestId(requestId)
  }, [setActiveRequestId])

  useEffect(() => {
    const storedHistory = normalizeChatHistory(readJSON<unknown>(A3_CHAT_HISTORY_KEY, []))
    const storedThreadId = isBrowser() ? localStorage.getItem(A3_CURRENT_THREAD_ID_KEY) : null
    const storedChatId = isBrowser() ? localStorage.getItem(A3_CURRENT_CHAT_ID_KEY) : null
    const activeThreadId = storedThreadId || storedChatId || storedHistory[0]?.threadId || null

    setChatHistory(storedHistory)
    setSelectedChatId(activeThreadId ?? undefined)
    threadIdRef.current = activeThreadId
    setCurrentThreadId(activeThreadId)
    setMessages(activeThreadId ? normalizeMessages(readJSON<unknown>(messageStorageKey(activeThreadId), [])) : [])
    setStorageReady(true)
  }, [])

  // Fix hydration: update initial log timestamp on client only
  useEffect(() => {
    setLogs((prev) => {
      if (prev.length === 1 && prev[0].ts === "--:--:--") {
        return [{ ...prev[0], ts: timestamp() }]
      }
      return prev
    })
  }, [])

  useEffect(() => {
    if (!storageReady) return
    if (chatHistory.length > 0) {
      writeJSON(A3_CHAT_HISTORY_KEY, chatHistory)
    } else {
      removeStorageItem(A3_CHAT_HISTORY_KEY)
    }
  }, [chatHistory, storageReady])

  useEffect(() => {
    if (!storageReady) return
    if (selectedChatId) {
      localStorage.setItem(A3_CURRENT_CHAT_ID_KEY, selectedChatId)
    } else {
      removeStorageItem(A3_CURRENT_CHAT_ID_KEY)
    }
  }, [selectedChatId, storageReady])

  useEffect(() => {
    if (!storageReady) return
    if (currentThreadId) {
      localStorage.setItem(A3_CURRENT_THREAD_ID_KEY, currentThreadId)
      writeJSON(messageStorageKey(currentThreadId), messages)
    } else {
      removeStorageItem(A3_CURRENT_THREAD_ID_KEY)
    }
  }, [currentThreadId, messages, storageReady])

  const updateAssistantResourceStatus = useCallback((
    messageId: string,
    updater: (status: ResourceGenerationStatus) => ResourceGenerationStatus,
  ) => {
    if (!messageId) return

    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId && msg.role === "assistant" && msg.resourceStatus
          ? { ...msg, resourceStatus: updater(msg.resourceStatus) }
          : msg
      )
    )
  }, [])

  const ensureAssistantResourceStatus = useCallback((
    messageId: string,
    updater: (status: ResourceGenerationStatus) => ResourceGenerationStatus,
  ) => {
    if (!messageId) return
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId && msg.role === "assistant"
          ? { ...msg, resourceStatus: updater(msg.resourceStatus ?? createInitialResourceStatus()) }
          : msg
      ),
    )
  }, [])

  const attachQAFinalToAssistant = useCallback((event: QAFinalEventV1, source: "stream" | "restore") => {
    if (source === "stream") {
      if (threadIdRef.current && event.threadId !== threadIdRef.current) {
        throw new ContractParseError("qa_final_binding", "thread_id does not match active stream")
      }
      if (requestIdRef.current && event.requestId !== requestIdRef.current) {
        throw new ContractParseError("qa_final_binding", "request_id does not match active stream")
      }
    } else if (threadIdRef.current && event.threadId !== threadIdRef.current) {
      throw new ContractParseError("qa_final_binding", "thread_id does not match restored thread")
    }
    const dedupeKey = qaFinalDedupeKey(event)
    if (qaFinalDedupeRef.current.has(dedupeKey)) return false
    qaFinalDedupeRef.current.add(dedupeKey)
    setMessages((current) => {
      try {
        return attachQAFinalToMessages(
          current,
          event,
          source === "stream" ? assistantMessageIdRef.current : "",
        ).messages
      } catch (error) {
        qaFinalDedupeRef.current.delete(dedupeKey)
        queueMicrotask(() => {
          setLogs((logs) => [
            ...logs,
            {
              type: "warning",
              message: `[CONTRACT] QA final binding rejected: ${contractFailureReason(error)}`,
              ts: timestamp(),
            },
          ])
        })
        return current
      }
    })
    return true
  }, [])

  const attachResourceFinalToAssistant = useCallback((
    event: ResourceFinalEvent,
    source: "stream" | "restore",
  ) => {
    if (source === "stream") {
      if (threadIdRef.current && event.thread_id !== threadIdRef.current) {
        throw new ContractParseError(
          "resource_final_binding",
          "thread_id does not match active stream",
        )
      }
      if (requestIdRef.current && event.request_id !== requestIdRef.current) {
        throw new ContractParseError(
          "resource_final_binding",
          "request_id does not match active stream",
        )
      }
    } else if (threadIdRef.current && event.thread_id !== threadIdRef.current) {
      throw new ContractParseError(
        "resource_final_binding",
        "thread_id does not match restored thread",
      )
    }
    const dedupeKey = resourceFinalDedupeKey(event)
    if (resourceFinalDedupeRef.current.has(dedupeKey)) {
      return { attached: false, dedupeKey }
    }
    resourceFinalDedupeRef.current.add(dedupeKey)

    const existingAssistantId = source === "stream" ? assistantMessageIdRef.current : ""
    const messageId = existingAssistantId || resourceMessageIdFromDedupeKey(dedupeKey)
    if (!existingAssistantId) assistantMessageIdRef.current = messageId

    setMessages((prev) => {
      if (prev.some((msg) => msg.resourceFinalDedupeKey === dedupeKey)) return prev
      const requestTarget = prev.find(
        (message) =>
          message.role === "assistant" &&
          message.requestId === event.request_id &&
          message.threadId === event.thread_id,
      )
      const targetMessageId = requestTarget?.id || messageId
      const baseMessage: Message = requestTarget || {
        id: messageId,
        role: "assistant",
        content: "",
        requestId: typeof event.request_id === "string" ? event.request_id : undefined,
        threadId: typeof event.thread_id === "string" ? event.thread_id : undefined,
        resourceStatus: createInitialResourceStatus(),
      }
      let foundTarget = false
      const nextMessages = prev.map((msg) => {
        if (msg.id !== targetMessageId) return msg
        foundTarget = true
        return mergeResourceFinalIntoMessage(msg, event, API_BASE_URL)
      })
      if (foundTarget) return nextMessages
      return [...nextMessages, mergeResourceFinalIntoMessage(baseMessage, event, API_BASE_URL)]
    })

    return { attached: true, dedupeKey, messageId }
  }, [])

  const handleNewChat = useCallback(() => {
    setSelectedChatId(undefined)
    setMessages([])
    setActivityTimeline([])
    setLogs([{ type: "info", message: "[INFO] 已开始新对话。", ts: timestamp() }])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setContextUsageState(EMPTY_CONTEXT_USAGE_STATE)
    setBackgroundContextWindow(null)
    setThreadContextWindowV2(null)
    setThreadContextWindowV3(null)
    liveTurnRef.current = null
    setLiveTurn(null)
    setActiveRequestId("")
    setCanContinue(false)
    setStopPending(false)
    setIsInterrupted(false)
    setInterruptDraft("")
    setMemoryConfirmation(null)
    setProfileCompletion(null)
    resourceFinalDedupeRef.current.clear()
    qaFinalDedupeRef.current.clear()
    assistantMessageIdRef.current = ""
    setActiveThreadId(null)
    pendingChatTitleRef.current = ""
  }, [setActiveThreadId])

  const handleSelectChat = useCallback((id: string) => {
    const chat = chatHistory.find((item) => item.id === id || item.threadId === id)
    const threadId = chat?.threadId || id
    setSelectedChatId(threadId)
    setMessages(normalizeMessages(readJSON<unknown>(messageStorageKey(threadId), [])))
    setActivityTimeline([])
    setContextUsageState(EMPTY_CONTEXT_USAGE_STATE)
    setBackgroundContextWindow(null)
    setThreadContextWindowV2(null)
    setThreadContextWindowV3(null)
    liveTurnRef.current = null
    setLiveTurn(null)
    setActiveRequestId("")
    setCanContinue(false)
    setStopPending(false)
    setIsInterrupted(false)
    setInterruptDraft("")
    setMemoryConfirmation(null)
    setProfileCompletion(null)
    resourceFinalDedupeRef.current.clear()
    qaFinalDedupeRef.current.clear()
    assistantMessageIdRef.current = ""
    setActiveThreadId(threadId)
    setLogs((prev) => [
      ...prev,
      { type: "info", message: `[INFO] Restored chat thread: ${threadId.slice(0, 8)}...`, ts: timestamp() },
    ])
  }, [chatHistory, setActiveThreadId])

  const handleClearChatHistory = useCallback(() => {
    if (isBrowser()) {
      localStorage.removeItem(A3_CHAT_HISTORY_KEY)
      localStorage.removeItem(A3_CURRENT_CHAT_ID_KEY)
      localStorage.removeItem(A3_CURRENT_THREAD_ID_KEY)
      Object.keys(localStorage)
        .filter((key) => key.startsWith(A3_MESSAGES_KEY_PREFIX))
        .forEach((key) => localStorage.removeItem(key))
    }
    setChatHistory([])
    setSelectedChatId(undefined)
    setMessages([])
    setActivityTimeline([])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setContextUsageState(EMPTY_CONTEXT_USAGE_STATE)
    setBackgroundContextWindow(null)
    setThreadContextWindowV2(null)
    setThreadContextWindowV3(null)
    liveTurnRef.current = null
    setLiveTurn(null)
    setActiveRequestId("")
    setCanContinue(false)
    setStopPending(false)
    setIsInterrupted(false)
    setInterruptDraft("")
    setMemoryConfirmation(null)
    setProfileCompletion(null)
    resourceFinalDedupeRef.current.clear()
    qaFinalDedupeRef.current.clear()
    assistantMessageIdRef.current = ""
    setActiveThreadId(null)
    pendingChatTitleRef.current = ""
    setLogs([{ type: "info", message: "[INFO] 对话历史已清空。", ts: timestamp() }])
  }, [setActiveThreadId])

  const handleClearChat = useCallback(async (id: string) => {
    const chat = chatHistory.find((item) => item.id === id || item.threadId === id)
    const threadId = chat?.threadId || id
    const nextHistory = chatHistory.filter((item) => item.id !== id && item.threadId !== threadId)

    if (isBrowser()) {
      localStorage.removeItem(messageStorageKey(threadId))
      if (selectedChatId === id || selectedChatId === threadId) {
        localStorage.removeItem(A3_CURRENT_CHAT_ID_KEY)
        localStorage.removeItem(A3_CURRENT_THREAD_ID_KEY)
      }
    }

    setChatHistory(nextHistory)
    if (selectedChatId === id || selectedChatId === threadId) {
      setSelectedChatId(undefined)
      setMessages([])
      setActivityTimeline([])
      setTokenUsage({ input: 0, output: 0, total: 0 })
      setContextUsageState(EMPTY_CONTEXT_USAGE_STATE)
      setBackgroundContextWindow(null)
      setThreadContextWindowV2(null)
      setThreadContextWindowV3(null)
      liveTurnRef.current = null
      setLiveTurn(null)
    setActiveRequestId("")
      setIsInterrupted(false)
      setInterruptDraft("")
      setMemoryConfirmation(null)
      setProfileCompletion(null)
    resourceFinalDedupeRef.current.clear()
    qaFinalDedupeRef.current.clear()
      assistantMessageIdRef.current = ""
      setActiveThreadId(null)
      pendingChatTitleRef.current = ""
    }

    setLogs((prev) => [
      ...prev,
      { type: "info", message: `[INFO] 已清除此对话：${threadId.slice(0, 8)}...`, ts: timestamp() },
    ])

    try {
      const response = await fetch(`${API_BASE_URL}/dev/threads/${encodeURIComponent(threadId)}/memory/clear`, {
        method: "POST",
        headers: { ...getAuthHeaders() },
      })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      setLogs((prev) => [
        ...prev,
        { type: "info", message: `[INFO] 已清理后端记忆：${threadId.slice(0, 8)}...`, ts: timestamp() },
      ])
    } catch (error: any) {
      setLogs((prev) => [
        ...prev,
        {
          type: "warning",
          message: `[WARN] 本地对话已清除，后端记忆清理未完成：${error.message}`,
          ts: timestamp(),
        },
      ])
    }
  }, [chatHistory, selectedChatId, setActiveThreadId])

  const fetchGraphManifest = useCallback(async (
    expectedVersion = "",
    endpoint = "/graph/manifest",
  ) => {
    setGraphManifestLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: { ...getAuthHeaders() },
      })
      const payload: unknown = await response.json()
      if (!response.ok) {
        const unavailable = parseGraphManifestUnavailable(payload)
        graphManifestVersionRef.current = ""
        setGraphManifest(null)
        setGraphManifestError(unavailable)
        return
      }
      const manifest = parseGraphManifest(payload)
      if (expectedVersion && manifest.graphVersion !== expectedVersion) {
        throw new ContractParseError(
          "graph_manifest_v1",
          "graph_version does not match the stream reference",
        )
      }
      graphManifestVersionRef.current = manifest.graphVersion
      setGraphManifest((current) =>
        current?.graphVersion === manifest.graphVersion ? current : manifest,
      )
      setGraphManifestError(null)
    } catch (error) {
      graphManifestVersionRef.current = ""
      setGraphManifest(null)
      setGraphManifestError(graphManifestFailure(error))
      setLogs((current) => [
        ...current,
        {
          type: "warning",
          message: `[CONTRACT] Graph Manifest rejected: ${contractFailureReason(error)}`,
          ts: timestamp(),
        },
      ])
    } finally {
      setGraphManifestLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchGraphManifest()
  }, [fetchGraphManifest])

  /** Process a single SSE data payload shared between /stream and /resume */
  const processSSEEvent = useCallback((data: any) => {
    const asstId = assistantMessageIdRef.current
    const performanceTracker = frontendPerformanceTrackerRef.current
    const performanceWasTerminal = performanceTracker?.isTerminal() ?? false
    performanceTracker?.recordEvent(data.type)
    if (!performanceWasTerminal && performanceTracker?.isTerminal()) {
      deliverFrontendPerformance(performanceTracker)
    }
    streamLifecycleRef.current = reduceStreamLifecycle(streamLifecycleRef.current, data)
    if (["done", "error", "interrupt"].includes(streamLifecycleRef.current.terminalEvent)) {
      setIsLoading(false)
      setContextUsageState((current) => finishContextUsageUpdate(current))
    }

    if (data.type === "thread_id") {
      const threadId = data.thread_id
      setActiveThreadId(threadId)
      setSelectedChatId(threadId)
      setChatHistory((prev) => {
        const existing = prev.find((item) => item.id === threadId || item.threadId === threadId)
        const nextItem: ChatHistoryItem = {
          id: threadId,
          threadId,
          title: existing?.title || makeChatTitle(pendingChatTitleRef.current),
          updatedAt: Date.now(),
        }
        return [nextItem, ...prev.filter((item) => item.id !== threadId && item.threadId !== threadId)]
      })
      setLogs((prev) => [
        ...prev,
        { type: "info", message: `[INFO] Thread: ${threadId.slice(0, 8)}...`, ts: timestamp() },
      ])
      return
    }

    if (data.type === "run_status") {
      const runStatus = typeof data.run_status === "string" ? data.run_status : ""
      const pendingInterruptType = typeof data.pending_interrupt_type === "string" ? data.pending_interrupt_type : ""
      if (data.thread_id) setActiveThreadId(data.thread_id)

      if (runStatus === "stopping") {
        setStopPending(true)
        updateAssistantResourceStatus(asstId, (status) => ({
          ...status,
          state: "stopping",
          summary: "Stop requested. Waiting for the next safe checkpoint.",
          waitingForReview: false,
        }))
        setLogs((prev) => [
          ...prev,
          { type: "warning", message: "[RUN] Stop requested; waiting for safe checkpoint.", ts: timestamp() },
        ])
        return
      }

      if (runStatus === "stopped") {
        clearStopTimeout()
        setIsLoading(false)
        setStopPending(false)
        setCanContinue(true)
        updateAssistantResourceStatus(asstId, (status) => ({
          ...status,
          state: "stopped",
          summary: "Stopped at a safe checkpoint. You can continue this run.",
          waitingForReview: false,
          error: undefined,
        }))
        setLogs((prev) => [
          ...prev,
          { type: "warning", message: "[RUN] Stopped at safe checkpoint; resume is available.", ts: timestamp() },
        ])
        return
      }

      if (runStatus === "continuing" || runStatus === "running") {
        setStopPending(false)
        setCanContinue(false)
        updateAssistantResourceStatus(asstId, (status) => ({
          ...status,
          state: "running",
          summary: runStatus === "continuing" ? "Continuing from saved checkpoint." : status.summary,
          waitingForReview: false,
        }))
        return
      }

      if (runStatus === "not_resumable") {
        setIsLoading(false)
        setCanContinue(false)
        setStopPending(false)
        updateAssistantResourceStatus(asstId, (status) => ({
          ...status,
          state: "stopped",
          summary: data.message || "This thread is not resumable from a saved stop checkpoint.",
          waitingForReview: false,
        }))
        setLogs((prev) => [
          ...prev,
          {
            type: "warning",
            message: `[RUN] Not resumable${pendingInterruptType ? ` (${pendingInterruptType})` : ""}.`,
            ts: timestamp(),
          },
        ])
        return
      }

      if (runStatus === "completed") {
        setCanContinue(false)
        setStopPending(false)
        clearStopTimeout()
        return
      }
    }

    if (data.type === "thread_context_window_v3") {
      try {
        setThreadContextWindowV3(
          parseThreadContextWindowV3(data.thread_context_window_v3),
        )
      } catch (error) {
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Thread context v3 rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
      }
      return
    }

    if (data.type === "context_usage_report") {
      try {
        const report = parseContextUsageReport(data)
        setContextUsageState((current) => applyContextUsageReport(current, report))
        updateAssistantResourceStatus(asstId, (status) => ({ ...status, contextUsage: report }))
        setLogs((current) => [
          ...current,
          {
            type: "context",
            message: `[CONTEXT] ${report.nodeName}: ${Math.round(report.usedRatio * 100)}% used, ${report.availableTokens} available`,
            ts: timestamp(),
          },
        ])
      } catch (error) {
        setContextUsageState((current) =>
          applyContextUsageError(current, contextUsageContractFailure(error)),
        )
      }
      if (data.thread_context_window_v2) {
        try {
          setThreadContextWindowV2(
            parseThreadContextWindowV2(data.thread_context_window_v2),
          )
        } catch (error) {
          setLogs((current) => [
            ...current,
            {
              type: "warning",
              message: `[CONTRACT] Thread context v2 rejected: ${contractFailureReason(error)}`,
              ts: timestamp(),
            },
          ])
        }
      }
      return
    }

    if (data.type === "stream_context") {
      try {
        const streamContext = parseStreamContext(data)
        setActiveRequestId(streamContext.requestId)
        setActiveThreadId(streamContext.threadId)
        setMessages((current) =>
          current.map((message) =>
            message.id === asstId && message.role === "assistant"
              ? {
                  ...message,
                  requestId: streamContext.requestId,
                  threadId: streamContext.threadId,
                }
              : message,
          ),
        )
        if (graphManifestVersionRef.current !== streamContext.graphVersion) {
          void fetchGraphManifest(streamContext.graphVersion)
        }
        if (data.performance_telemetry !== undefined) {
          try {
            performanceTracker?.bind(
              parseFrontendPerformanceCapability(data.performance_telemetry),
              {
                requestId: streamContext.requestId,
                threadId: streamContext.threadId,
              },
            )
          } catch (error) {
            setLogs((current) => [
              ...current,
              {
                type: "warning",
                message: `[CONTRACT] Frontend performance capability rejected: ${contractFailureReason(error)}`,
                ts: timestamp(),
              },
            ])
          }
        }
      } catch (error) {
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Stream context rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
      }
      return
    }

    if (data.type === "graph_manifest_ref") {
      try {
        const reference = parseGraphManifestRef(data)
        if (graphManifestVersionRef.current !== reference.graphVersion) {
          void fetchGraphManifest(reference.graphVersion, reference.endpoint)
        }
      } catch (error) {
        setGraphManifest(null)
        setGraphManifestError(graphManifestFailure(error))
      }
      return
    }

    if (data.type === "activity_event") {
      try {
        const activity = parseActivityEvent(data)
        setActiveRequestId(activity.requestId)
        setActivityTimeline((current) => mergeActivityTimeline(current, [activity]))
        setMessages((current) =>
          attachActivityToAssistantMessage(current, activity, assistantMessageIdRef.current),
        )
        if (activity.node) {
          const updateResourceStatus = isResourceActivity(activity)
            ? ensureAssistantResourceStatus
            : updateAssistantResourceStatus
          updateResourceStatus(asstId, (status) => {
            const steps = [...status.steps]
            const stepState: ResourceGenerationStep["state"] =
              activity.status === "failed" || activity.status === "interrupted"
                ? "error"
                : activity.status === "completed" || activity.status === "skipped"
                  ? "done"
                  : "running"
            let stepIndex = -1
            for (let index = steps.length - 1; index >= 0; index -= 1) {
              if (steps[index].node === activity.node) {
                stepIndex = index
                break
              }
            }
            const step: ResourceGenerationStep = {
              ...(stepIndex >= 0 ? steps[stepIndex] : {}),
              node: activity.node,
              title: activity.title,
              detail: activity.summary,
              state: stepState,
              startedAt: activity.startedAt,
              endedAt: stepState === "running" ? undefined : activity.completedAt || activity.updatedAt,
              durationMs: activity.durationMs,
              error: stepState === "error" ? activity.summary : undefined,
            }
            if (stepIndex >= 0) steps[stepIndex] = step
            else steps.push(step)
            return {
              ...status,
              state: stepState === "error" ? "error" : status.state,
              summary: activity.summary || activity.title,
              steps,
              error: stepState === "error" ? activity.summary : status.error,
            }
          })
        }
      } catch (error) {
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Activity event rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
      }
      return
    }

    if (data.type === "llm_input_manifest") {
      try {
        setBackgroundContextWindow(
          parseBackgroundContextWindow(data.background_context_window),
        )
      } catch (error) {
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Background context rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
      }
      if (data.thread_context_window_v2) {
        try {
          setThreadContextWindowV2(
            parseThreadContextWindowV2(data.thread_context_window_v2),
          )
        } catch (error) {
          setLogs((current) => [
            ...current,
            {
              type: "warning",
              message: `[CONTRACT] Thread context v2 rejected: ${contractFailureReason(error)}`,
              ts: timestamp(),
            },
          ])
        }
      }
      setLogs((prev) => [
        ...prev,
        {
          type: "context",
          message: `[CE] LLM input manifest ${data.node_name || data.node || "unknown"} sections=${Array.isArray(data.section_names) ? data.section_names.length : 0}`,
          ts: timestamp(),
        },
      ])
      return
    }

    if (data.type === "context_usage_report_error") {
      let usageError: ContextUsageReportError
      try {
        usageError = parseContextUsageReportError(data)
      } catch (error) {
        usageError = contextUsageContractFailure(error)
      }
      setContextUsageState((current) => applyContextUsageError(current, usageError))
      setLogs((current) => [
        ...current,
        {
          type: "warning",
          message: `[CONTEXT] ${usageError.reason}: ${usageError.warning}`,
          ts: timestamp(),
        },
      ])
      return
    }

    if (data.type === "interrupt") {
      if (data.interrupt_type === "profile_completion_required") {
        const request =
          data.profile_completion_request && typeof data.profile_completion_request === "object"
            ? data.profile_completion_request
            : data
        setProfileCompletion(mapProfileCompletionRequest(request))
        setMemoryConfirmation(null)
        setIsInterrupted(false)
        setInterruptDraft("")
        if (data.thread_id) setActiveThreadId(data.thread_id)
        setIsLoading(false)
        ensureAssistantResourceStatus(asstId, (status) => ({
          ...status,
          state: "waiting_for_profile_completion",
          summary: "等待补充生成学习计划所需的学习信息。",
          waitingForReview: true,
        }))
        setLogs((prev) => [
          ...prev,
          { type: "warning", message: "[HIL] 等待补充学习计划画像信息。", ts: timestamp() },
        ])
        return
      }

      if (data.interrupt_type === "memory_confirmation") {
        setMemoryConfirmation({
          question:
            typeof data.question === "string" && data.question.trim()
              ? data.question
              : "我检测到之前有相关学习记录。你希望这次结合历史内容，还是只根据当前问题重新生成？",
          reason: typeof data.reason === "string" ? data.reason : "",
          selectedMemoryCount: typeof data.selected_memory_count === "number" ? data.selected_memory_count : undefined,
        })
        setIsInterrupted(false)
        setInterruptDraft("")
        if (data.thread_id) setActiveThreadId(data.thread_id)
        updateAssistantResourceStatus(asstId, (status) => ({
          ...status,
          state: "waiting_review",
          summary: "等待确认是否结合历史学习记录。",
          waitingForReview: true,
        }))
        setLogs((prev) => [
          ...prev,
          { type: "warning", message: "[HIL] 等待确认是否结合历史学习记录。", ts: timestamp() },
        ])
        setIsLoading(false)
        return
      }

      setInterruptDraft(data.draft)
      setIsInterrupted(true)
      if (data.thread_id) setActiveThreadId(data.thread_id)
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: "interrupted",
        summary: "个性化学习资源生成状态已更新。",
        waitingForReview: true,
      }))
      setLogs((prev) => [
        ...prev,
        { type: "warning", message: "[HIL] 图执行已暂停，等待你审核学习计划。", ts: timestamp() },
      ])
      setIsLoading(false)
      return
    }

    if (data.type === "context_usage" || data.type === "context_usage_error") return

    if (data.type === "provider_retry") {
      const retryCount = typeof data.retry_count === "number" ? data.retry_count : 0
      const maxRetries = typeof data.max_retries === "number" ? data.max_retries : 0
      const stage = typeof data.stage === "string" ? data.stage : ""
      const node = typeof data.node === "string" && data.node ? data.node : typeof data.llm_node === "string" ? data.llm_node : "provider"
      const errorType = typeof data.error_type === "string" ? data.error_type : "TransportError"
      const isFinalFailure = stage === "final_failure_after_retries"
      const summary = isFinalFailure
        ? `Provider 连接重试 ${retryCount}/${maxRetries} 次后仍失败。`
        : `Provider 连接异常，正在第 ${retryCount}/${maxRetries} 次重试。`
      setLogs((prev) => [
        ...prev,
        {
          type: isFinalFailure ? "error" : "warning",
          message: `[RETRY] ${node}: ${summary} (${errorType})`,
          ts: timestamp(),
        },
      ])
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        summary,
      }))
      return
    }

    if (data.type === "qa_final") {
      try {
        const qaFinal = parseQAFinalEvent(data)
        attachQAFinalToAssistant(qaFinal, "stream")
      } catch (error) {
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] QA final rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
      }
      return
    }

    if (data.type === "token") {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === asstId
            ? { ...msg, content: msg.content + data.content }
            : msg
        )
      )
      return
    }

    if (data.type === "text") {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === asstId ? { ...msg, content: data.content } : msg
        )
      )
      return
    }

    if (data.type === "mindmap_result") {
      const xmindUrl =
        typeof data.xmind_url === "string" && data.xmind_url.startsWith("/")
          ? `${API_BASE_URL}${data.xmind_url}`
          : data.xmind_url
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === asstId
            ? {
                ...msg,
                mindmap: {
                  title: data.title || "Knowledge Mindmap",
                  tree: data.tree,
                  xmindUrl,
                },
              }
            : msg
        )
      )
      return
    }

    if (data.type === "review_doc_result") {
      const markdownUrl =
        typeof data.markdown_url === "string" && data.markdown_url.startsWith("/")
          ? `${API_BASE_URL}${data.markdown_url}`
          : data.markdown_url
      const docxUrl =
        typeof data.docx_url === "string" && data.docx_url.startsWith("/")
          ? `${API_BASE_URL}${data.docx_url}`
          : data.docx_url
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === asstId
            ? {
                ...msg,
                reviewDoc: {
                  title: data.title || "Review Document",
                  filename: data.filename || "",
                  markdownUrl: markdownUrl || "",
                  docxFilename: data.docx_filename || "",
                  docxUrl: docxUrl || "",
                },
              }
            : msg
        )
      )
      return
    }

    if (isCompletedWithoutResourceDiagnostic(data)) {
      ensureAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: "completed_without_resource",
        summary: "资源流程已完成，但没有收到可渲染的资源 payload。",
        waitingForReview: false,
        completionKind: "without_resource",
      }))
      return
    }

    if (data.type === "resource_final") {
      let event: ResourceFinalEvent
      try {
        event = parseResourceFinalEvent(data)
      } catch (error) {
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Resource final rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
        return
      }
      let result
      try {
        result = attachResourceFinalToAssistant(event, "stream")
      } catch (error) {
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Resource final binding rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
        return
      }
      if (!result.attached) return
      const targetMessageId = result.messageId || asstId
      const outcome = resourceFinalOutcome(event)
      if (!outcome) return
      ensureAssistantResourceStatus(targetMessageId, (status) => ({
        ...status,
        state: outcome.state,
        summary: outcome.summary,
        waitingForReview: false,
        error: outcome.state === "failed" ? outcome.summary : undefined,
        hasReceivedResourceFinal: outcome.hasReceivedResourceFinal,
        completionKind: outcome.completionKind,
        lastResourceType: event.resource_type,
      }))
      return
    }

    if (data.type === "done") {
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state:
          status.state === "error" ||
          status.state === "failed" ||
          status.state === "success" ||
          status.state === "partial_success" ||
          status.state === "controlled_stop" ||
          status.state === "stopped" ||
          status.state === "stopping" ||
          status.state === "completed_with_resource" ||
          status.state === "completed_without_resource"
            ? status.state
            : "done",
        summary: status.summary,
        waitingForReview: false,
      }))
      return
    }

    if (data.type === "error") {
      streamHadErrorRef.current = true
      setMessages((current) =>
        current.map((message) =>
          message.id === asstId && message.role === "assistant"
            ? mergeSafeFailureContent(message)
            : message,
        ),
      )
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: "failed",
        summary: "个性化学习资源生成状态已更新。",
        error: data.message,
        waitingForReview: false,
      }))
      setIsLoading(false)
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Server: ${data.message}`, ts: timestamp() },
      ])
      return
    }

    if (data.type === "node_event") {
      // Legacy compatibility event: ActivityEvent is the only source for trail and graph state.
      return
    }

    if (data.type === "usage") {
      const now = timestamp()
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        tokenUsage: {
          input: status.tokenUsage.input + (data.input_tokens ?? 0),
          output: status.tokenUsage.output + (data.output_tokens ?? 0),
          total: status.tokenUsage.total + (data.total_tokens ?? 0),
        },
      }))
      setTokenUsage((prev) => ({
        input: prev.input + (data.input_tokens ?? 0),
        output: prev.output + (data.output_tokens ?? 0),
        total: prev.total + (data.total_tokens ?? 0),
      }))
      setLogs((prev) => [
        ...prev,
        { type: "usage", message: `[USAGE] ${data.node}: 输入 ${data.input_tokens} / 输出 ${data.output_tokens}`, ts: now },
      ])
    }
  }, [
    attachResourceFinalToAssistant,
    clearStopTimeout,
    deliverFrontendPerformance,
    fetchGraphManifest,
    setActiveThreadId,
    updateAssistantResourceStatus,
  ])

  /** Reduce one public V2 event before translating its committed/progress payload. */
  const processAgentStreamEvent = useCallback((event: AgentStreamEventV2) => {
    const current = liveTurnRef.current
    const sameStream =
      current !== null &&
      current.streamId === event.streamId &&
      current.requestId === event.requestId &&
      current.threadId === event.threadId
    if (sameStream && event.sequence <= current.lastSequence) return
    if (current !== null && !sameStream) return

    let next: LiveTurnState
    try {
      next = reduceLiveTurn(current, event)
    } catch (error) {
      const reason =
        error instanceof LiveTurnSequenceError ? error.message : contractFailureReason(error)
      setLogs((items) => [
        ...items,
        {
          type: "error",
          message: `[STREAM GAP] ${reason}`,
          ts: timestamp(),
        },
      ])
      throw error
    }
    liveTurnRef.current = next
    setLiveTurn(next)

    if (event.type === "stream_start") {
      setActiveRequestId(event.requestId)
      processSSEEvent({ type: "thread_id", thread_id: event.threadId })
      return
    }
    if (event.type === "content_block_start" || event.type === "content_block_delta" || event.type === "content_block_stop") {
      return
    }
    if (
      event.type === "activity_update" ||
      event.type === "tool_progress" ||
      event.type === "artifact_progress"
    ) {
      const kind = event.data.kind
      const payload = event.data.payload
      if (typeof kind !== "string" || !payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new LiveTurnSequenceError(`${event.type} payload is invalid`)
      }
      processSSEEvent({ type: kind, ...(payload as Record<string, unknown>) })
      return
    }
    if (event.type === "stopped") {
      processSSEEvent({ type: "run_status", run_status: "stopped", ...event.data })
      return
    }
    if (event.type === "stream_error") {
      processSSEEvent({ type: "error", ...event.data })
      return
    }
    if (event.type === "stream_done") {
      processSSEEvent({ type: "done", ...event.data })
      liveTurnRef.current = null
      setLiveTurn(null)
      return
    }
    processSSEEvent({ type: event.type, ...event.data })
  }, [processSSEEvent, setActiveRequestId])

  /** Read, validate, replay-dedupe, and reconnect an agent_stream_v2 response. */
  const consumeSSEStream = useCallback(async (
    initialBody: ReadableStream<Uint8Array>,
    signal?: AbortSignal,
  ) => {
    streamLifecycleRef.current = beginStreamLifecycle()
    setContextUsageState((current) => beginContextUsageUpdate(current))
    await consumeAgentStreamV2({
      initialBody,
      onEvent: processAgentStreamEvent,
      signal,
      reconnect: async (streamId, lastEventId, reconnectSignal) => {
        const response = await fetch(`${API_BASE_URL}/streams/${encodeURIComponent(streamId)}`, {
          headers: { "Last-Event-ID": lastEventId, ...getAuthHeaders() },
          signal: reconnectSignal,
        })
        if (!response.ok || !response.body) {
          throw new Error(`Stream replay failed: HTTP ${response.status}`)
        }
        return response.body
      },
    })
  }, [processAgentStreamEvent])

  /** Fetch helper with shared HTTP error handling. Returns response body or null on handled error. */
  const fetchWithErrorHandling = useCallback(async (url: string, init: RequestInit): Promise<ReadableStream<Uint8Array> | null> => {
    const response = await fetch(url, init)

    if (response.status === 429) {
      setMessages((prev) => [
        ...prev,
        { id: (Date.now() + 1).toString(), role: "assistant", content: "服务繁忙，请稍后重试。" },
      ])
      setLogs((prev) => [
        ...prev,
        { type: "warning", message: "[WARN] 429 Too Many Requests", ts: timestamp() },
      ])
      return null
    }

    if (response.status === 401) {
      setMessages((prev) => [
        ...prev,
        { id: (Date.now() + 1).toString(), role: "assistant", content: "访问未授权，请检查访问令牌。" },
      ])
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] 401 未授权：访问令牌缺失或无效", ts: timestamp() },
      ])
      if (typeof window !== "undefined") localStorage.removeItem("demo_access_token")
      return null
    }

    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    if (!response.body) throw new Error("No response body")

    return response.body
  }, [])

  const refreshThreadStatus = useCallback(async (threadId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/threads/${encodeURIComponent(threadId)}/status`, {
        headers: { ...getAuthHeaders() },
      })
      if (response.status === 404) {
        setCanContinue(false)
        return
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const status = await response.json()
      try {
        const report =
          status.context_usage_report && Object.keys(status.context_usage_report).length > 0
            ? parseContextUsageReport(status.context_usage_report)
            : null
        setContextUsageState(restoreContextUsageReport(report))
      } catch (error) {
        setContextUsageState((current) =>
          applyContextUsageError(current, contextUsageContractFailure(error)),
        )
      }
      try {
        const background =
          status.background_context_window && Object.keys(status.background_context_window).length > 0
            ? parseBackgroundContextWindow(status.background_context_window)
            : null
        setBackgroundContextWindow(background)
      } catch (error) {
        setBackgroundContextWindow(null)
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Stored background context rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
      }
      try {
        const threadWindow =
          status.thread_context_window_v2 &&
          Object.keys(status.thread_context_window_v2).length > 0
            ? parseThreadContextWindowV2(status.thread_context_window_v2)
            : null
        setThreadContextWindowV2(threadWindow)
      } catch (error) {
        setThreadContextWindowV2(null)
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Stored thread context v2 rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
      }
      try {
        setThreadContextWindowV3(
          parseThreadContextWindowV3(status.thread_context_window_v3),
        )
      } catch (error) {
        setThreadContextWindowV3(null)
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Stored thread context v3 rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
      }
      try {
        const parsedTimeline = parseActivityTimeline(status.activity_timeline ?? [])
        setActivityTimeline(parsedTimeline.items)
        setMessages((current) =>
          restoreActivitiesToMessages(current, parsedTimeline.items, threadId),
        )
        setActiveRequestId(parsedTimeline.items.at(-1)?.requestId ?? "")
        if (parsedTimeline.rejectedCount > 0) {
          setLogs((current) => [
            ...current,
            {
              type: "warning",
              message: `[CONTRACT] Skipped ${parsedTimeline.rejectedCount} invalid stored activity event(s).`,
              ts: timestamp(),
            },
          ])
        }
      } catch (error) {
        setActivityTimeline([])
        setLogs((current) => [
          ...current,
          {
            type: "warning",
            message: `[CONTRACT] Stored activity timeline rejected: ${contractFailureReason(error)}`,
            ts: timestamp(),
          },
        ])
      }
      if (typeof status.graph_version === "string" && status.graph_version) {
        void fetchGraphManifest(status.graph_version)
      }
      if (status.last_qa_response?.type === "qa_final") {
        try {
          attachQAFinalToAssistant(parseQAFinalEvent(status.last_qa_response), "restore")
        } catch (error) {
          setLogs((current) => [
            ...current,
            {
              type: "warning",
              message: `[CONTRACT] Stored QA final rejected: ${contractFailureReason(error)}`,
              ts: timestamp(),
            },
          ])
        }
      }
      if (status.last_resource_final_payload?.type === "resource_final") {
        try {
          const event = parseResourceFinalEvent(status.last_resource_final_payload)
          attachResourceFinalToAssistant(event, "restore")
        } catch (error) {
          setLogs((current) => [
            ...current,
            {
              type: "warning",
              message: `[CONTRACT] Stored resource final rejected: ${contractFailureReason(error)}`,
              ts: timestamp(),
            },
          ])
        }
      }
      const pendingInterruptType =
        typeof status.pending_interrupt_type === "string" ? status.pending_interrupt_type : ""
      if (pendingInterruptType === "profile_completion_required") {
        const request = mapProfileCompletionRequest(status.profile_completion_request)
        if (request) {
          setProfileCompletion(request)
          updateAssistantResourceStatus(assistantMessageIdRef.current, (resourceStatus) => ({
            ...resourceStatus,
            state: "waiting_for_profile_completion",
            summary: "等待补充生成学习计划所需的学习信息。",
            waitingForReview: true,
          }))
        }
      }
      setCanContinue(Boolean(status.resume_available && pendingInterruptType === "user_stop"))
      if (status.schema_version === "legacy") {
        setLogs((prev) => [
          ...prev,
          { type: "warning", message: "[RUN] Legacy checkpoint: run-control status is unknown.", ts: timestamp() },
        ])
      }
    } catch (error: any) {
      setCanContinue(false)
      setLogs((prev) => [
        ...prev,
        { type: "warning", message: `[RUN] Status unavailable: ${error.message}`, ts: timestamp() },
      ])
    }
  }, [attachResourceFinalToAssistant, fetchGraphManifest, updateAssistantResourceStatus])

  useEffect(() => {
    if (!storageReady || !currentThreadId) return
    refreshThreadStatus(currentThreadId)
  }, [currentThreadId, refreshThreadStatus, storageReady])

  const handleSendMessage = useCallback(async (content: string) => {
    const threadId = threadIdRef.current
    const requestId = crypto.randomUUID()
    beginRequestStream(requestId)
    assistantMessageIdRef.current = ""
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
    }

    pendingChatTitleRef.current = content
    setMessages((prev) => [...prev, userMessage])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setCanContinue(false)
    setStopPending(false)
    clearStopTimeout()
    setIsInterrupted(false)
    setInterruptDraft("")
    setMemoryConfirmation(null)
    setProfileCompletion(null)
    setLogs((prev) => [
      ...prev,
      { type: "info" as const, message: `[INFO] 用户问题：${content.slice(0, 60)}`, ts: timestamp() },
    ])
    console.debug("[A3_CHAT] sending", { threadId, selectedChatId, messageCount: messages.length + 1 })

    setIsLoading(true)
    const abortController = new AbortController()
    abortControllerRef.current = abortController

    try {
      streamHadErrorRef.current = false
      beginFrontendPerformanceTracking()
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          query: content,
          request_id: requestId,
          thread_id: threadId,
          user_id: userId,
        }),
        signal: abortController.signal,
      })

      if (!body) return

      // Create an empty assistant message placeholder
      const assistantMessageId = (Date.now() + 1).toString()
      assistantMessageIdRef.current = assistantMessageId
      setMessages((prev) => [
        ...prev,
        { id: assistantMessageId, role: "assistant", content: "" },
      ])

      await consumeSSEStream(body, abortController.signal)

      setLogs((prev) => [
        ...prev,
        {
          type: streamHadErrorRef.current ? "error" : "info",
          message: streamHadErrorRef.current ? "[ERROR] 流式响应因错误结束。" : "[INFO] 流式响应完成。",
          ts: timestamp(),
        },
      ])
    } catch (error: any) {
      if (error?.name === "AbortError") {
        setLogs((prev) => [
          ...prev,
          { type: "warning", message: "[RUN] SSE closed after stop request; backend will save at the next safe checkpoint if still running.", ts: timestamp() },
        ])
        return
      }
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] ${error.message}`, ts: timestamp() },
      ])
    } finally {
      setIsLoading(false)
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null
      }
    }
  }, [beginFrontendPerformanceTracking, beginRequestStream, clearStopTimeout, selectedChatId, messages.length, fetchWithErrorHandling, consumeSSEStream, userId])

  const handleStopGeneration = useCallback(async () => {
    const threadId = threadIdRef.current
    const asstId = assistantMessageIdRef.current
    if (!threadId) {
      abortControllerRef.current?.abort()
      setIsLoading(false)
      return
    }

    setStopPending(true)
    updateAssistantResourceStatus(asstId, (status) => ({
      ...status,
      state: "stopping",
      summary: "Stop requested. Waiting for the next safe checkpoint.",
      waitingForReview: false,
    }))
    setLogs((prev) => [
      ...prev,
      { type: "warning", message: "[RUN] Stop requested. Waiting for safe checkpoint.", ts: timestamp() },
    ])

    try {
      const response = await fetch(`${API_BASE_URL}/threads/${encodeURIComponent(threadId)}/stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ reason: "user_stop" }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      clearStopTimeout()
      stopTimeoutRef.current = setTimeout(() => {
        abortControllerRef.current?.abort()
        setIsLoading(false)
        updateAssistantResourceStatus(asstId, (status) => ({
          ...status,
          state: "stopping",
          summary: "Stop requested. Backend will save at the next safe checkpoint.",
          waitingForReview: false,
        }))
        setLogs((prev) => [
          ...prev,
          {
            type: "warning",
            message: "[RUN] Stop is still pending; closed frontend SSE while backend moves to a safe checkpoint.",
            ts: timestamp(),
          },
        ])
      }, 45000)
    } catch (error: any) {
      setStopPending(false)
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: "error",
        summary: "Stop request failed.",
        error: error.message,
      }))
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[RUN] Stop request failed: ${error.message}`, ts: timestamp() },
      ])
    }
  }, [clearStopTimeout, updateAssistantResourceStatus])

  const handleContinueThread = useCallback(async () => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[RUN] Missing thread_id; cannot continue.", ts: timestamp() },
      ])
      return
    }

    const requestId = crypto.randomUUID()
    beginRequestStream(requestId)
    setIsLoading(true)
    setCanContinue(false)
    setStopPending(false)
    clearStopTimeout()
    streamHadErrorRef.current = false

    const assistantMessageId = (Date.now() + 1).toString()
    assistantMessageIdRef.current = assistantMessageId
    setMessages((prev) => [
      ...prev,
      { id: assistantMessageId, role: "assistant", content: "" },
    ])

    const abortController = new AbortController()
    abortControllerRef.current = abortController

    try {
      beginFrontendPerformanceTracking()
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/threads/${encodeURIComponent(threadId)}/continue`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ request_id: requestId }),
        signal: abortController.signal,
      })
      if (!body) return
      await consumeSSEStream(body, abortController.signal)
      setLogs((prev) => [
        ...prev,
        { type: streamHadErrorRef.current ? "error" : "info", message: "[RUN] Continue stream ended.", ts: timestamp() },
      ])
    } catch (error: any) {
      if (error?.name !== "AbortError") {
        setLogs((prev) => [
          ...prev,
          { type: "error", message: `[RUN] Continue failed: ${error.message}`, ts: timestamp() },
        ])
      }
    } finally {
      setIsLoading(false)
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null
      }
      refreshThreadStatus(threadId)
    }
  }, [beginFrontendPerformanceTracking, beginRequestStream, clearStopTimeout, consumeSSEStream, fetchWithErrorHandling, refreshThreadStatus])

  const handleResume = useCallback(async (editedPlan: string) => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] 缺少 thread_id，无法继续执行。", ts: timestamp() },
      ])
      return
    }

    const requestId = crypto.randomUUID()
    beginRequestStream(requestId)
    setIsResuming(true)
    setLogs((prev) => [
      ...prev,
      { type: "info", message: "[INFO] 正在使用已编辑方案继续执行图流程...", ts: timestamp() },
    ])

    try {
      beginFrontendPerformanceTracking()
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          request_id: requestId,
          thread_id: threadId,
          edited_plan: editedPlan,
        }),
      })

      if (!body) return

      setIsInterrupted(false)
      setInterruptDraft("")

      await consumeSSEStream(body)

      setLogs((prev) => [
        ...prev,
        { type: "info", message: "[INFO] 继续执行的流式响应已完成。", ts: timestamp() },
      ])
    } catch (error: any) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Resume failed: ${error.message}`, ts: timestamp() },
      ])
    } finally {
      setIsResuming(false)
      setIsLoading(false)
    }
  }, [beginFrontendPerformanceTracking, beginRequestStream, fetchWithErrorHandling, consumeSSEStream])

  const handleFeedback = useCallback(async (feedback: string) => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] 缺少 thread_id，无法发送反馈。", ts: timestamp() },
      ])
      return
    }

    const requestId = crypto.randomUUID()
    beginRequestStream(requestId)
    setIsResuming(true)
    setLogs((prev) => [
      ...prev,
      { type: "info", message: `[INFO] Sending feedback: ${feedback.slice(0, 40)}...`, ts: timestamp() },
    ])

    try {
      beginFrontendPerformanceTracking()
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ request_id: requestId, thread_id: threadId, feedback }),
      })

      if (!body) return

      // Hide PlanReview while system processes feedback
      setIsInterrupted(false)
      setInterruptDraft("")

      // Create new assistant message placeholder for the revised plan streaming
      const newAsstId = (Date.now() + 1).toString()
      assistantMessageIdRef.current = newAsstId
      setMessages((prev) => [
        ...prev,
        { id: newAsstId, role: "assistant", content: "" },
      ])

      await consumeSSEStream(body)

      setLogs((prev) => [
        ...prev,
        { type: "info", message: "[INFO] 反馈修订已完成。", ts: timestamp() },
      ])
    } catch (error: any) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Feedback failed: ${error.message}`, ts: timestamp() },
      ])
    } finally {
      setIsResuming(false)
      setIsLoading(false)
    }
  }, [beginFrontendPerformanceTracking, beginRequestStream, fetchWithErrorHandling, consumeSSEStream])

  const handleMemoryConfirmation = useCallback(async (choice: "use" | "ignore") => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] 缺少 thread_id，无法继续执行。", ts: timestamp() },
      ])
      return
    }

    const requestId = crypto.randomUUID()
    beginRequestStream(requestId)
    setIsMemoryConfirming(true)
    setIsLoading(true)
    setLogs((prev) => [
      ...prev,
      {
        type: "info",
        message: choice === "use" ? "[INFO] 已选择结合历史学习记录。" : "[INFO] 已选择只看当前问题。",
        ts: timestamp(),
      },
    ])

    try {
      beginFrontendPerformanceTracking()
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          request_id: requestId,
          thread_id: threadId,
          memory_use_choice: choice,
        }),
      })

      if (!body) return

      setMemoryConfirmation(null)
      await consumeSSEStream(body)

      setLogs((prev) => [
        ...prev,
        { type: "info", message: "[INFO] 历史上下文确认后已继续执行。", ts: timestamp() },
      ])
    } catch (error: any) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Memory confirmation failed: ${error.message}`, ts: timestamp() },
      ])
    } finally {
      setIsMemoryConfirming(false)
      setIsLoading(false)
    }
  }, [beginFrontendPerformanceTracking, beginRequestStream, fetchWithErrorHandling, consumeSSEStream])

  const handleProfileCompletion = useCallback(async (completion: Record<string, string>) => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] 缺少 thread_id，无法继续生成学习计划。", ts: timestamp() },
      ])
      return
    }

    const requestId = crypto.randomUUID()
    beginRequestStream(requestId)
    const nonEmptyCompletion = Object.fromEntries(
      Object.entries(completion).filter(([, value]) => value.trim().length > 0),
    )
    setIsProfileCompleting(true)
    setIsLoading(true)
    setLogs((prev) => [
      ...prev,
      { type: "info", message: "[INFO] 已提交学习计划画像信息，继续生成。", ts: timestamp() },
    ])

    try {
      beginFrontendPerformanceTracking()
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({
          request_id: requestId,
          thread_id: threadId,
          profile_completion: nonEmptyCompletion,
        }),
      })

      if (!body) return

      setProfileCompletion(null)
      await consumeSSEStream(body)

      setLogs((prev) => [
        ...prev,
        { type: "info", message: "[INFO] 学习计划画像补充后已继续执行。", ts: timestamp() },
      ])
    } catch (error: any) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Profile completion failed: ${error.message}`, ts: timestamp() },
      ])
    } finally {
      setIsProfileCompleting(false)
      setIsLoading(false)
    }
  }, [beginFrontendPerformanceTracking, beginRequestStream, consumeSSEStream, fetchWithErrorHandling])

  return (
    <div className="a3-app-shell flex overflow-hidden">
      <LeftSidebar
        chatHistory={chatHistory}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onClearChat={handleClearChat}
        onClearChatHistory={handleClearChatHistory}
        selectedChatId={selectedChatId}
        userId={userId}
        nickname={nickname}
        onStartOnboarding={startOnboarding}
        onClearUser={() => {
          if (typeof window !== "undefined") {
            localStorage.removeItem("a3_user_id")
            localStorage.removeItem("a3_nickname")
            localStorage.removeItem("a3_onboarding_completed")
            window.location.reload()
          }
        }}
      />
      <div className="flex min-w-0 flex-1 flex-col h-full">
        <ChatArea
          messages={messages}
          liveTurnContent={liveTurn?.provisionalAnswer ?? ""}
          liveTurnProgress={
            liveTurn
              ? {
                  lifecycle: liveTurn.lifecycle,
                  activityCount: liveTurn.activities.length,
                  toolCount: liveTurn.tools.length,
                }
              : null
          }
          onSendMessage={handleSendMessage}
          onStopGeneration={handleStopGeneration}
          onContinueThread={handleContinueThread}
          isLoading={isLoading && !isInterrupted}
          canContinue={canContinue && !isLoading && !isInterrupted}
          stopPending={stopPending}
          threadContextWindow={threadContextWindowV3}
          contextWindowCloseSignal={`${currentThreadId || ""}:${currentRequestId}:${isLoading ? "running" : "idle"}`}
        />
        {isInterrupted && (
          <PlanReview
            draft={interruptDraft}
            onConfirm={handleResume}
            onFeedback={handleFeedback}
            isSubmitting={isResuming}
          />
        )}
        {memoryConfirmation && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(36,48,39,0.22)] px-4"
            role="dialog"
            aria-modal="true"
            aria-labelledby="memory-confirmation-title"
          >
            <div className="w-full max-w-[440px] rounded-2xl border border-border bg-card p-5 text-card-foreground shadow-[0_12px_30px_rgba(36,48,39,0.12)]">
              <div className="mb-4">
                <p id="memory-confirmation-title" className="text-base font-semibold text-foreground">
                  是否结合历史学习记录？
                </p>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  {memoryConfirmation.question}
                </p>
                {memoryConfirmation.selectedMemoryCount ? (
                  <p className="mt-3 rounded-lg border border-border bg-muted px-3 py-2 text-xs leading-5 text-muted-foreground">
                    已找到 {memoryConfirmation.selectedMemoryCount} 条可参考的历史证据摘要。选择后我会继续生成，不会修改历史记录。
                  </p>
                ) : null}
              </div>
              <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                <button
                  type="button"
                  className="rounded-lg border border-border bg-card px-4 py-2 text-sm font-medium text-foreground transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30 disabled:cursor-not-allowed disabled:opacity-60"
                  onClick={() => handleMemoryConfirmation("ignore")}
                  disabled={isMemoryConfirming}
                >
                  只看当前问题
                </button>
                <button
                  type="button"
                  className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-[var(--primary-deep)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30 disabled:cursor-not-allowed disabled:opacity-60"
                  onClick={() => handleMemoryConfirmation("use")}
                  disabled={isMemoryConfirming}
                >
                  结合历史
                </button>
              </div>
            </div>
          </div>
        )}
        {profileCompletion && (
          <ProfileCompletionDialog
            request={profileCompletion}
            isSubmitting={isProfileCompleting}
            onSubmit={handleProfileCompletion}
          />
        )}
      </div>
      <RightPanel
        logs={logs}
        activities={activityTimeline}
        tokenUsage={tokenUsage}
        graphManifest={graphManifest}
        graphManifestError={graphManifestError}
        graphManifestLoading={graphManifestLoading}
        currentRequestId={currentRequestId}
        isInterrupted={isInterrupted}
      />
    </div>
  )
}
