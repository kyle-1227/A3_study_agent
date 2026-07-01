"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { LeftSidebar } from "@/components/left-sidebar"
import { useUser } from "@/hooks/use-user"
import { RightPanel, NodeEvent, LogEntry } from "@/components/right-panel"
import {
  ChatArea,
  ContextUsage,
  ContextUsageError,
  Message,
  ResourceGenerationStatus,
  ResourceGenerationStep,
} from "@/components/chat-area"
import { PlanReview } from "@/components/plan-review"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const A3_CHAT_HISTORY_KEY = "a3_chat_history"
const A3_CURRENT_CHAT_ID_KEY = "a3_current_chat_id"
const A3_CURRENT_THREAD_ID_KEY = "a3_current_thread_id"
const A3_MESSAGES_KEY_PREFIX = "a3_messages:"
const CONTROLLED_STOP_SUMMARY = "证据不足，已保存摘要并停止完整资源生成。"

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
  return raw.filter((item: any) => {
    return (
      item &&
      typeof item.id === "string" &&
      (item.role === "user" || item.role === "assistant") &&
      typeof item.content === "string"
    )
  }) as Message[]
}

function makeChatTitle(content: string): string {
  const compact = content.trim().replace(/\s+/g, " ")
  if (!compact) return "新对话"
  return compact.slice(0, 30) + (compact.length > 30 ? "..." : "")
}

function timestamp(): string {
  return new Date().toLocaleTimeString("en-GB", { hour12: false })
}

function numberField(data: any, key: string, legacyKey?: string): number {
  if (typeof data[key] === "number") return data[key]
  if (legacyKey && typeof data[legacyKey] === "number") return data[legacyKey]
  return 0
}

function mapContextUsage(data: any): ContextUsage {
  return {
    node: typeof data.node === "string" ? data.node : "",
    llmNode: typeof data.llm_node === "string" ? data.llm_node : "",
    provider: typeof data.provider === "string" ? data.provider : "",
    model: typeof data.model === "string" ? data.model : "",
    inputEstimatedTokens: numberField(data, "input_estimated_tokens", "prompt_tokens"),
    reservedOutputTokens: numberField(data, "reserved_output_tokens", "output_reserved_tokens"),
    usedTokens: numberField(data, "used_tokens"),
    maxContextTokens: numberField(data, "max_context_tokens"),
    availableTokens: numberField(data, "available_tokens", "remaining_tokens"),
    usedRatio: numberField(data, "used_ratio", "usage_ratio"),
    warningLevel:
      typeof data.warning_level === "string"
        ? data.warning_level
        : typeof data.level === "string"
          ? data.level
          : "ok",
    estimated: Boolean(data.estimated),
    tokenizerMode: typeof data.tokenizer_mode === "string" ? data.tokenizer_mode : "",
    messageCount: numberField(data, "message_count"),
    schemaSizeChars: typeof data.schema_size_chars === "number" ? data.schema_size_chars : undefined,
  }
}

function mapContextUsageError(data: any): ContextUsageError {
  return {
    node: typeof data.node === "string" ? data.node : "",
    llmNode: typeof data.llm_node === "string" ? data.llm_node : "",
    provider: typeof data.provider === "string" ? data.provider : "",
    model: typeof data.model === "string" ? data.model : "",
    reason: typeof data.reason === "string" ? data.reason : "context_usage_unavailable",
    warning:
      typeof data.warning === "string"
        ? data.warning
        : "context usage telemetry unavailable",
  }
}

function getAuthHeaders(): Record<string, string> {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("demo_access_token")
    if (token) return { "X-Access-Token": token }
  }
  return {}
}

const RESOURCE_NODE_COPY: Record<string, { title: string; detail: string }> = {
  supervisor: { title: "解析学习需求", detail: "识别课程主题、学习目标和需要生成的资源类型。" },
  memory_use_decider: { title: "确认历史上下文", detail: "判断本轮检索是否需要结合历史学习记录。" },
  search_query_rewriter: { title: "改写检索查询", detail: "将原始问题改写为适合本地课程库和网络搜索的查询。" },
  academic_router: { title: "选择学习资源链路", detail: "调度本地 RAG、网络搜索和资源生成子 Agent。" },
  rag_retrieve: { title: "本地 RAG", detail: "从本地课程资料库检索候选证据。" },
  web_search: { title: "Tavily 网络搜索", detail: "检索外部学习资料与官方文档候选证据。" },
  evidence_judge: { title: "证据评审", detail: "裁决本地和网络候选证据，只保留可信上下文。" },
  evidence_summary_output: { title: "证据摘要输出", detail: "证据不足时输出摘要、缺口和后续检索建议。" },
  generate_answer: { title: "生成学习回答", detail: "基于已裁决证据生成课程答疑或学习资源建议。" },
  evaluate_hallucination: { title: "可信度校验", detail: "检查回答与证据的一致性。" },
  rewrite_query: { title: "重写检索问题", detail: "根据校验反馈准备下一轮检索。" },
  study_plan_emotional_intel: { title: "情绪画像分析", detail: "分析学习负担、节奏风险和支持需求。" },
  study_plan_planner: { title: "学习计划规划", detail: "基于证据、目标和学习者状态规划学习计划蓝图。" },
  study_plan_agent: { title: "学习计划生成", detail: "生成结构化个性化学习计划。" },
  study_plan_reviewer_academic: { title: "学习计划学术审查", detail: "检查阶段递进、证据一致性和资源可靠性。" },
  study_plan_reviewer_emotional: { title: "学习计划负担审查", detail: "检查任务负担、复盘休息和执行可持续性。" },
  study_plan_consensus: { title: "学习计划共识检查", detail: "汇总双 reviewer 结论，决定输出或修订。" },
  study_plan_rewrite: { title: "学习计划修订", detail: "根据审查意见准备下一轮生成。" },
  study_plan_output: { title: "学习计划输出", detail: "渲染 Markdown 学习计划并生成文档 artifact。" },
  mindmap_planner: { title: "规划知识结构", detail: "规划课程知识点的导图结构。" },
  mindmap_agent: { title: "生成导图", detail: "生成结构化 JSON Tree。" },
  mindmap_reviewer: { title: "审查导图质量", detail: "检查层级、覆盖和学术准确性。" },
  mindmap_rewrite: { title: "修订导图", detail: "根据审查意见重写导图。" },
  mindmap_output: { title: "导出导图", detail: "生成 XMind 等导图 artifact。" },
  exercise_planner: { title: "规划练习结构", detail: "规划基础、进阶、应用和自检练习。" },
  exercise_agent: { title: "生成分层练习", detail: "生成包含答案、解析和易错提醒的练习题。" },
  exercise_reviewer: { title: "审查练习质量", detail: "检查题型覆盖、难度递进和解析完整性。" },
  exercise_rewrite: { title: "修订练习", detail: "根据审查意见重写练习。" },
  exercise_output: { title: "输出练习资源", detail: "整理最终分层练习资源。" },
  review_doc_planner: { title: "规划复习文档", detail: "规划 Markdown 复习文档结构。" },
  review_doc_agent: { title: "生成复习文档", detail: "生成 Markdown 课程复习文档。" },
  review_doc_reviewer: { title: "审查文档质量", detail: "检查结构、证据使用和内容完整性。" },
  review_doc_rewrite: { title: "修订复习文档", detail: "根据审查意见重写文档。" },
  review_doc_output: { title: "输出复习文档", detail: "生成 Markdown/DOCX 文档 artifact。" },
  emotional_response: { title: "生成学业支持建议", detail: "围绕学习压力、适应和执行困难生成支持性建议。" },
  handle_unknown: { title: "确认服务范围", detail: "判断请求是否属于高校课程学习与个性化资源生成范围。" },
}
function createInitialResourceStatus(): ResourceGenerationStatus {
  return {
    state: "running",
    summary: "正在解析学习需求，准备调度多智能体生成个性化学习资源。",
    steps: [],
    tokenUsage: { input: 0, output: 0, total: 0 },
  }
}

function createResourceStep(
  node: string,
  state: ResourceGenerationStep["state"],
  ts: string,
): ResourceGenerationStep {
  const copy = RESOURCE_NODE_COPY[node] ?? {
    title: node,
    detail: "正在处理个性化学习资源生成流程中的一个技术阶段。",
  }

  return {
    node,
    title: copy.title,
    detail: copy.detail,
    state,
    startedAt: ts,
  }
}

function findLastRunningStepIndex(steps: ResourceGenerationStep[], node: string): number {
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i].node === node && steps[i].state === "running") return i
  }
  return -1
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
  const [nodeEvents, setNodeEvents] = useState<NodeEvent[]>([])
  const [tokenUsage, setTokenUsage] = useState({ input: 0, output: 0, total: 0 })
  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null)
  const [contextUsageError, setContextUsageError] = useState<ContextUsageError | null>(null)
  const [canContinue, setCanContinue] = useState(false)
  const [stopPending, setStopPending] = useState(false)

  // HIL state
  const [isInterrupted, setIsInterrupted] = useState(false)
  const [interruptDraft, setInterruptDraft] = useState("")
  const [isResuming, setIsResuming] = useState(false)
  const [memoryConfirmation, setMemoryConfirmation] = useState<MemoryConfirmationState | null>(null)
  const [isMemoryConfirming, setIsMemoryConfirming] = useState(false)
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
  const abortControllerRef = useRef<AbortController | null>(null)
  const stopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

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
        msg.id === messageId && msg.role === "assistant"
          ? { ...msg, resourceStatus: updater(msg.resourceStatus ?? createInitialResourceStatus()) }
          : msg
      )
    )
  }, [])

  const handleNewChat = useCallback(() => {
    setSelectedChatId(undefined)
    setMessages([])
    setNodeEvents([])
    setLogs([{ type: "info", message: "[INFO] 已开始新对话。", ts: timestamp() }])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setContextUsage(null)
    setContextUsageError(null)
    setCanContinue(false)
    setStopPending(false)
    setIsInterrupted(false)
    setInterruptDraft("")
    setMemoryConfirmation(null)
    setActiveThreadId(null)
    pendingChatTitleRef.current = ""
  }, [setActiveThreadId])

  const handleSelectChat = useCallback((id: string) => {
    const chat = chatHistory.find((item) => item.id === id || item.threadId === id)
    const threadId = chat?.threadId || id
    setSelectedChatId(threadId)
    setMessages(normalizeMessages(readJSON<unknown>(messageStorageKey(threadId), [])))
    setNodeEvents([])
    setContextUsage(null)
    setContextUsageError(null)
    setCanContinue(false)
    setStopPending(false)
    setIsInterrupted(false)
    setInterruptDraft("")
    setMemoryConfirmation(null)
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
    setNodeEvents([])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setContextUsage(null)
    setContextUsageError(null)
    setCanContinue(false)
    setStopPending(false)
    setIsInterrupted(false)
    setInterruptDraft("")
    setMemoryConfirmation(null)
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
      setNodeEvents([])
      setTokenUsage({ input: 0, output: 0, total: 0 })
      setIsInterrupted(false)
      setInterruptDraft("")
      setMemoryConfirmation(null)
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

  /** Process a single SSE data payload shared between /stream and /resume */
  const processSSEEvent = useCallback((data: any) => {
    const asstId = assistantMessageIdRef.current

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

    if (data.type === "context_usage") {
      const usage = mapContextUsage(data)
      setContextUsage(usage)
      setContextUsageError(null)
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        contextUsage: usage,
      }))
      setLogs((prev) => [
        ...prev,
        {
          type: "context",
          message: `[CONTEXT] ${usage.node || usage.llmNode}: ${Math.round(usage.usedRatio * 100)}% used, ${usage.availableTokens} available`,
          ts: timestamp(),
        },
      ])
      return
    }

    if (data.type === "context_usage_error") {
      const usageError = mapContextUsageError(data)
      setContextUsage(null)
      setContextUsageError(usageError)
      setLogs((prev) => [
        ...prev,
        {
          type: "warning",
          message: `[CONTEXT] ${usageError.reason}: ${usageError.warning}`,
          ts: timestamp(),
        },
      ])
      return
    }

    if (data.type === "interrupt") {
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
        return
      }

      setInterruptDraft(data.draft)
      setIsInterrupted(true)
      if (data.thread_id) setActiveThreadId(data.thread_id)
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: "waiting_review",
        summary: "个性化学习资源生成状态已更新。",
        waitingForReview: true,
      }))
      setLogs((prev) => [
        ...prev,
        { type: "warning", message: "[HIL] 图执行已暂停，等待你审核学习计划。", ts: timestamp() },
      ])
      return
    }

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

    if (data.type === "resource_final") {
      const finalAnswer = typeof data.answer === "string" ? data.answer : ""
      if (data.resource_type === "evidence_summary" && data.controlled_stop === true) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === asstId
              ? {
                  ...msg,
                  content: finalAnswer || msg.content,
                }
              : msg
          )
        )
        updateAssistantResourceStatus(asstId, (status) => ({
          ...status,
          state: "done",
          summary: CONTROLLED_STOP_SUMMARY,
          waitingForReview: false,
          error: undefined,
        }))
        return
      }

      const mindmap = data.mindmap ?? null
      const reviewDoc = data.review_doc ?? null
      const reviewDocArtifacts = Array.isArray(data.review_doc_artifacts) ? data.review_doc_artifacts : []
      const exerciseArtifact = data.exercise_artifact ?? null
      const codePracticeArtifact = data.code_practice_artifact ?? null
      const videoScriptArtifact = data.video_script_artifact ?? null
      const videoAnimationArtifact = data.video_animation_artifact ?? null
      const studyPlan = data.study_plan ?? null
      const xmindUrl =
        mindmap && typeof mindmap.xmind_url === "string" && mindmap.xmind_url.startsWith("/")
          ? `${API_BASE_URL}${mindmap.xmind_url}`
          : mindmap?.xmind_url
      const markdownUrl =
        reviewDoc && typeof reviewDoc.markdown_url === "string" && reviewDoc.markdown_url.startsWith("/")
          ? `${API_BASE_URL}${reviewDoc.markdown_url}`
          : reviewDoc?.markdown_url
      const docxUrl =
        reviewDoc && typeof reviewDoc.docx_url === "string" && reviewDoc.docx_url.startsWith("/")
          ? `${API_BASE_URL}${reviewDoc.docx_url}`
          : reviewDoc?.docx_url
      const reviewDocs = reviewDocArtifacts.map((artifact: any) => {
        const artifactMarkdownUrl =
          typeof artifact.markdown_url === "string" && artifact.markdown_url.startsWith("/")
            ? `${API_BASE_URL}${artifact.markdown_url}`
            : artifact.markdown_url
        const artifactDocxUrl =
          typeof artifact.docx_url === "string" && artifact.docx_url.startsWith("/")
            ? `${API_BASE_URL}${artifact.docx_url}`
            : artifact.docx_url
        return {
          subject: artifact.subject || "",
          title: artifact.title || "Review Document",
          filename: artifact.filename || "",
          markdownUrl: artifactMarkdownUrl || "",
          docxFilename: artifact.docx_filename || "",
          docxUrl: artifactDocxUrl || "",
          markdown: artifact.markdown || "",
        }
      })
      const exerciseMarkdownUrl =
        exerciseArtifact && typeof exerciseArtifact.markdown_url === "string" && exerciseArtifact.markdown_url.startsWith("/")
          ? `${API_BASE_URL}${exerciseArtifact.markdown_url}`
          : exerciseArtifact?.markdown_url
      const exerciseDocxUrl =
        exerciseArtifact && typeof exerciseArtifact.docx_url === "string" && exerciseArtifact.docx_url.startsWith("/")
          ? `${API_BASE_URL}${exerciseArtifact.docx_url}`
          : exerciseArtifact?.docx_url
      const codePracticeMarkdownUrl =
        codePracticeArtifact && typeof codePracticeArtifact.markdown_url === "string" && codePracticeArtifact.markdown_url.startsWith("/")
          ? `${API_BASE_URL}${codePracticeArtifact.markdown_url}`
          : codePracticeArtifact?.markdown_url
      const codePracticeDocxUrl =
        codePracticeArtifact && typeof codePracticeArtifact.docx_url === "string" && codePracticeArtifact.docx_url.startsWith("/")
          ? `${API_BASE_URL}${codePracticeArtifact.docx_url}`
          : codePracticeArtifact?.docx_url
      const codePracticePythonUrl =
        codePracticeArtifact && typeof codePracticeArtifact.python_url === "string" && codePracticeArtifact.python_url.startsWith("/")
          ? `${API_BASE_URL}${codePracticeArtifact.python_url}`
          : codePracticeArtifact?.python_url
      const videoScriptMarkdownUrl =
        videoScriptArtifact && typeof videoScriptArtifact.markdown_url === "string" && videoScriptArtifact.markdown_url.startsWith("/")
          ? `${API_BASE_URL}${videoScriptArtifact.markdown_url}`
          : videoScriptArtifact?.markdown_url
      const videoScriptDocxUrl =
        videoScriptArtifact && typeof videoScriptArtifact.docx_url === "string" && videoScriptArtifact.docx_url.startsWith("/")
          ? `${API_BASE_URL}${videoScriptArtifact.docx_url}`
          : videoScriptArtifact?.docx_url
      const videoScriptSrtUrl =
        videoScriptArtifact && typeof videoScriptArtifact.srt_url === "string" && videoScriptArtifact.srt_url.startsWith("/")
          ? `${API_BASE_URL}${videoScriptArtifact.srt_url}`
          : videoScriptArtifact?.srt_url
      const videoAnimationHtmlUrl =
        videoAnimationArtifact && typeof videoAnimationArtifact.html_url === "string" && videoAnimationArtifact.html_url.startsWith("/")
          ? `${API_BASE_URL}${videoAnimationArtifact.html_url}`
          : videoAnimationArtifact?.html_url
      const videoAnimationMp4Url =
        videoAnimationArtifact && typeof videoAnimationArtifact.mp4_url === "string" && videoAnimationArtifact.mp4_url.startsWith("/")
          ? `${API_BASE_URL}${videoAnimationArtifact.mp4_url}`
          : videoAnimationArtifact?.mp4_url
      const videoAnimationSrtUrl =
        videoAnimationArtifact && typeof videoAnimationArtifact.srt_url === "string" && videoAnimationArtifact.srt_url.startsWith("/")
          ? `${API_BASE_URL}${videoAnimationArtifact.srt_url}`
          : videoAnimationArtifact?.srt_url
      const videoAnimationJsonUrl =
        videoAnimationArtifact && typeof videoAnimationArtifact.json_url === "string" && videoAnimationArtifact.json_url.startsWith("/")
          ? `${API_BASE_URL}${videoAnimationArtifact.json_url}`
          : videoAnimationArtifact?.json_url
      const studyPlanMarkdownUrl =
        studyPlan && typeof studyPlan.markdown_url === "string" && studyPlan.markdown_url.startsWith("/")
          ? `${API_BASE_URL}${studyPlan.markdown_url}`
          : studyPlan?.markdown_url
      const studyPlanDocxUrl =
        studyPlan && typeof studyPlan.docx_url === "string" && studyPlan.docx_url.startsWith("/")
          ? `${API_BASE_URL}${studyPlan.docx_url}`
          : studyPlan?.docx_url

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === asstId
            ? {
                ...msg,
                content: finalAnswer || msg.content,
                mindmap: mindmap
                  ? {
                      title: mindmap.title || "Knowledge Mindmap",
                      tree: mindmap.tree,
                      xmindUrl: xmindUrl || "",
                    }
                  : msg.mindmap,
                reviewDoc: reviewDocs.length > 0
                  ? undefined
                  : reviewDoc
                  ? {
                      subject: reviewDoc.subject || "",
                      title: reviewDoc.title || "Review Document",
                      filename: reviewDoc.filename || "",
                      markdownUrl: markdownUrl || "",
                      docxFilename: reviewDoc.docx_filename || "",
                      docxUrl: docxUrl || "",
                      markdown: reviewDoc.markdown || "",
                    }
                  : msg.reviewDoc,
                reviewDocs: reviewDocs.length > 0 ? reviewDocs : msg.reviewDocs,
                exercise: exerciseArtifact
                  ? {
                      title: exerciseArtifact.title || "Exercise Resource",
                      filename: exerciseArtifact.filename || "",
                      markdownUrl: exerciseMarkdownUrl || "",
                      docxFilename: exerciseArtifact.docx_filename || "",
                      docxUrl: exerciseDocxUrl || "",
                    }
                  : msg.exercise,
                codePractice: codePracticeArtifact
                  ? {
                      title: codePracticeArtifact.title || "代码实操案例",
                      filename: codePracticeArtifact.filename || "",
                      markdownUrl: codePracticeMarkdownUrl || "",
                      docxFilename: codePracticeArtifact.docx_filename || "",
                      docxUrl: codePracticeDocxUrl || "",
                      pythonFilename: codePracticeArtifact.python_filename || "",
                      pythonUrl: codePracticePythonUrl || "",
                      markdown: codePracticeArtifact.markdown || "",
                    }
                  : msg.codePractice,
                videoScript: videoScriptArtifact
                  ? {
                      title: videoScriptArtifact.title || "教学视频 / 动画脚本",
                      filename: videoScriptArtifact.filename || "",
                      markdownUrl: videoScriptMarkdownUrl || "",
                      docxFilename: videoScriptArtifact.docx_filename || "",
                      docxUrl: videoScriptDocxUrl || "",
                      srtFilename: videoScriptArtifact.srt_filename || "",
                      srtUrl: videoScriptSrtUrl || "",
                      markdown: videoScriptArtifact.markdown || "",
                      srt: videoScriptArtifact.srt || "",
                    }
                  : msg.videoScript,
                videoAnimation: videoAnimationArtifact
                  ? {
                      title: videoAnimationArtifact.title || "教学动画 / MP4 视频",
                      htmlUrl: videoAnimationHtmlUrl || "",
                      mp4Url: videoAnimationMp4Url || "",
                      srtUrl: videoAnimationSrtUrl || "",
                      jsonUrl: videoAnimationJsonUrl || "",
                      durationSeconds: videoAnimationArtifact.duration_seconds,
                      fullDurationSeconds: videoAnimationArtifact.full_duration_seconds,
                      renderDurationSeconds: videoAnimationArtifact.render_duration_seconds,
                      renderMode: videoAnimationArtifact.render_mode || "",
                      renderSuccess: videoAnimationArtifact.render_success === true,
                      mp4Available: videoAnimationArtifact.mp4_available === true,
                      isPreviewVideo: videoAnimationArtifact.is_preview_video === true,
                      videoValidForTeaching: videoAnimationArtifact.video_valid_for_teaching === true,
                      renderLog: videoAnimationArtifact.render_log || "",
                    }
                  : msg.videoAnimation,
                studyPlan: studyPlan
                  ? {
                      title: studyPlan.title || "Personalized Study Plan",
                      filename: studyPlan.filename || "",
                      markdownUrl: studyPlanMarkdownUrl || "",
                      docxFilename: studyPlan.docx_filename || "",
                      docxUrl: studyPlanDocxUrl || "",
                      markdown: studyPlan.markdown || "",
                    }
                  : msg.studyPlan,
              }
            : msg
        )
      )
      return
    }

    if (data.type === "done") {
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: status.state === "error" || status.state === "stopped" || status.state === "stopping" ? status.state : "done",
        summary: status.summary === CONTROLLED_STOP_SUMMARY ? status.summary : "个性化学习资源生成状态已更新。",
        waitingForReview: false,
      }))
      return
    }

    if (data.type === "error") {
      streamHadErrorRef.current = true
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: "error",
        summary: "个性化学习资源生成状态已更新。",
        error: data.message,
        waitingForReview: false,
      }))
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] Server: ${data.message}`, ts: timestamp() },
      ])
      return
    }

    if (data.type === "node_event") {
      const node: string = data.node
      const status: "start" | "end" = data.status
      const now = timestamp()

      updateAssistantResourceStatus(asstId, (resourceStatus) => {
        const steps = [...resourceStatus.steps]
        const copy = RESOURCE_NODE_COPY[node]

        if (status === "start") {
          steps.push(createResourceStep(node, "running", now))
          return {
            ...resourceStatus,
            state: resourceStatus.state === "stopping" ? "stopping" : "running",
            summary: resourceStatus.state === "stopping"
              ? resourceStatus.summary
              : copy?.detail ?? "多智能体正在推进个性化学习资源生成。",
            steps,
            waitingForReview: false,
          }
        }

        const nextState: ResourceGenerationStep["state"] = data.error ? "error" : "done"
        const runningIndex = findLastRunningStepIndex(steps, node)
        const completedStep = {
          ...(runningIndex >= 0 ? steps[runningIndex] : createResourceStep(node, nextState, now)),
          state: nextState,
          endedAt: now,
          durationMs: data.duration_ms ?? undefined,
          error: data.error ?? undefined,
        }

        if (runningIndex >= 0) {
          steps[runningIndex] = completedStep
        } else {
          steps.push(completedStep)
        }

        return {
          ...resourceStatus,
          state: data.error ? "error" : resourceStatus.state,
          summary: data.error
            ? "个性化资源生成的某个阶段遇到异常。"
            : copy?.detail ?? resourceStatus.summary,
          steps,
          error: data.error ?? resourceStatus.error,
        }
      })

      setNodeEvents((prev) => {
        if (status === "start") {
          return [...prev, { node, status: "running", ts: now }]
        }
        const nextStatus: NodeEvent["status"] = data.error ? "error" : "done"
        let updated = false
        const nextEvents = prev.map((e) => {
          if (e.node === node && e.status === "running") {
            updated = true
            return {
              ...e,
              status: nextStatus,
              endTs: now,
              durationMs: data.duration_ms ?? undefined,
              error: data.error ?? undefined,
              synthetic: Boolean(data.synthetic),
            }
          }
          return e
        })
        if (updated) return nextEvents
        return [
          ...nextEvents,
          {
            node,
            status: nextStatus,
            ts: now,
            endTs: now,
            durationMs: data.duration_ms ?? undefined,
            error: data.error ?? undefined,
            synthetic: Boolean(data.synthetic),
          },
        ]
      })

      const label = status === "start" ? "进入" : data.error ? "失败" : "完成"
      setLogs((prev) => [
        ...prev,
        { type: data.error ? "error" : "info", message: `${data.error ? "[ERROR]" : "[INFO]"} 节点 ${node} ${label}`, ts: now },
      ])

      if (status === "end" && data.duration_ms != null) {
        setLogs((prev) => [
          ...prev,
          {
            type: data.error ? "error" : "perf",
            message: data.error
              ? `[ERROR] 节点 "${node}" 在 ${data.duration_ms}ms 后失败`
              : `[PERF] 节点 "${node}" 完成，用时 ${data.duration_ms}ms`,
            ts: now,
          },
        ])
      }

      if (status === "end" && data.error) {
        setLogs((prev) => [
          ...prev,
          { type: "error", message: `[ERROR] 节点 "${node}"：${data.error}`, ts: now },
        ])
      }
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
  }, [clearStopTimeout, setActiveThreadId, updateAssistantResourceStatus])

  /** Read an SSE response body and dispatch events via processSSEEvent */
  const consumeSSEStream = useCallback(async (body: ReadableStream<Uint8Array>) => {
    const reader = body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const parts = buffer.split("\n\n")
      buffer = parts.pop() || ""

      for (const part of parts) {
        if (part.startsWith("data: ")) {
          try {
            const data = JSON.parse(part.slice(6))
            processSSEEvent(data)
          } catch {
            // Ignore partial or malformed JSON chunks
          }
        }
      }
    }
  }, [processSSEEvent])

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
      const usage = status.context_usage && Object.keys(status.context_usage).length > 0
        ? mapContextUsage(status.context_usage)
        : null
      setContextUsage(usage)
      if (usage) setContextUsageError(null)
      setCanContinue(Boolean(status.resume_available && status.pending_interrupt_type === "user_stop"))
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
  }, [])

  useEffect(() => {
    if (!storageReady || !currentThreadId) return
    refreshThreadStatus(currentThreadId)
  }, [currentThreadId, refreshThreadStatus, storageReady])

  const handleSendMessage = useCallback(async (content: string) => {
    const threadId = threadIdRef.current
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
    }

    pendingChatTitleRef.current = content
    setMessages((prev) => [...prev, userMessage])
    setNodeEvents([])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setContextUsage(null)
    setContextUsageError(null)
    setCanContinue(false)
    setStopPending(false)
    clearStopTimeout()
    setIsInterrupted(false)
    setInterruptDraft("")
    setMemoryConfirmation(null)
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
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ query: content, thread_id: threadId, user_id: userId }),
        signal: abortController.signal,
      })

      if (!body) return

      // Create an empty assistant message placeholder
      const assistantMessageId = (Date.now() + 1).toString()
      assistantMessageIdRef.current = assistantMessageId
      setMessages((prev) => [
        ...prev,
        { id: assistantMessageId, role: "assistant", content: "", resourceStatus: createInitialResourceStatus() },
      ])

      await consumeSSEStream(body)

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
  }, [clearStopTimeout, selectedChatId, messages.length, fetchWithErrorHandling, consumeSSEStream, userId])

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

    setIsLoading(true)
    setCanContinue(false)
    setStopPending(false)
    clearStopTimeout()
    streamHadErrorRef.current = false

    const assistantMessageId = (Date.now() + 1).toString()
    assistantMessageIdRef.current = assistantMessageId
    setMessages((prev) => [
      ...prev,
      { id: assistantMessageId, role: "assistant", content: "", resourceStatus: createInitialResourceStatus() },
    ])

    const abortController = new AbortController()
    abortControllerRef.current = abortController

    try {
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/threads/${encodeURIComponent(threadId)}/continue`, {
        method: "POST",
        headers: { ...getAuthHeaders() },
        signal: abortController.signal,
      })
      if (!body) return
      await consumeSSEStream(body)
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
  }, [clearStopTimeout, consumeSSEStream, fetchWithErrorHandling, refreshThreadStatus])

  const handleResume = useCallback(async (editedPlan: string) => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] 缺少 thread_id，无法继续执行。", ts: timestamp() },
      ])
      return
    }

    setIsResuming(true)
    setLogs((prev) => [
      ...prev,
      { type: "info", message: "[INFO] 正在使用已编辑方案继续执行图流程...", ts: timestamp() },
    ])

    try {
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ thread_id: threadId, edited_plan: editedPlan }),
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
  }, [fetchWithErrorHandling, consumeSSEStream])

  const handleFeedback = useCallback(async (feedback: string) => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] 缺少 thread_id，无法发送反馈。", ts: timestamp() },
      ])
      return
    }

    setIsResuming(true)
    setLogs((prev) => [
      ...prev,
      { type: "info", message: `[INFO] Sending feedback: ${feedback.slice(0, 40)}...`, ts: timestamp() },
    ])

    try {
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ thread_id: threadId, feedback }),
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
        { id: newAsstId, role: "assistant", content: "", resourceStatus: createInitialResourceStatus() },
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
  }, [fetchWithErrorHandling, consumeSSEStream])

  const handleMemoryConfirmation = useCallback(async (choice: "use" | "ignore") => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] 缺少 thread_id，无法继续执行。", ts: timestamp() },
      ])
      return
    }

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
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ thread_id: threadId, memory_use_choice: choice }),
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
  }, [fetchWithErrorHandling, consumeSSEStream])

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
          onSendMessage={handleSendMessage}
          onStopGeneration={handleStopGeneration}
          onContinueThread={handleContinueThread}
          isLoading={isLoading && !isInterrupted}
          canContinue={canContinue && !isLoading && !isInterrupted}
          stopPending={stopPending}
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
      </div>
      <RightPanel
        logs={logs}
        nodeEvents={nodeEvents}
        tokenUsage={tokenUsage}
        contextUsage={contextUsage}
        contextUsageError={contextUsageError}
        isInterrupted={isInterrupted}
      />
    </div>
  )
}
