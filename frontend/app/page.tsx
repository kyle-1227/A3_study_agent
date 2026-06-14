"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { LeftSidebar } from "@/components/left-sidebar"
import { RightPanel, NodeEvent, LogEntry } from "@/components/right-panel"
import {
  ChatArea,
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

type ChatHistoryItem = {
  id: string
  threadId: string
  title: string
  updatedAt?: number
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

function getAuthHeaders(): Record<string, string> {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("demo_access_token")
    if (token) return { "X-Access-Token": token }
  }
  return {}
}

const RESOURCE_NODE_COPY: Record<string, { title: string; detail: string }> = {
  supervisor: { title: "解析学习需求", detail: "识别课程主题、学习目标和需要生成的资源类型。" },
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

  // HIL state
  const [isInterrupted, setIsInterrupted] = useState(false)
  const [interruptDraft, setInterruptDraft] = useState("")
  const [isResuming, setIsResuming] = useState(false)
  const threadIdRef = useRef<string | null>(null)
  const assistantMessageIdRef = useRef<string>("")
  const pendingChatTitleRef = useRef<string>("")
  const streamHadErrorRef = useRef(false)

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
    setIsInterrupted(false)
    setInterruptDraft("")
    setActiveThreadId(null)
    pendingChatTitleRef.current = ""
  }, [setActiveThreadId])

  const handleSelectChat = useCallback((id: string) => {
    const chat = chatHistory.find((item) => item.id === id || item.threadId === id)
    const threadId = chat?.threadId || id
    setSelectedChatId(threadId)
    setMessages(normalizeMessages(readJSON<unknown>(messageStorageKey(threadId), [])))
    setNodeEvents([])
    setIsInterrupted(false)
    setInterruptDraft("")
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
    setIsInterrupted(false)
    setInterruptDraft("")
    setActiveThreadId(null)
    pendingChatTitleRef.current = ""
    setLogs([{ type: "info", message: "[INFO] 对话历史已清空。", ts: timestamp() }])
  }, [setActiveThreadId])

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

    if (data.type === "interrupt") {
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
      const mindmap = data.resource_type === "mindmap" ? data.mindmap : null
      const reviewDoc = data.resource_type === "review_doc" ? data.review_doc : null
      const reviewDocArtifacts =
        data.resource_type === "review_doc" && Array.isArray(data.review_doc_artifacts)
          ? data.review_doc_artifacts
          : []
      const exerciseArtifact = data.resource_type === "quiz" ? data.exercise_artifact : null
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
                reviewDoc: reviewDocs.length > 1
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
                reviewDocs: reviewDocs.length > 1 ? reviewDocs : msg.reviewDocs,
                exercise: exerciseArtifact
                  ? {
                      title: exerciseArtifact.title || "Exercise Resource",
                      filename: exerciseArtifact.filename || "",
                      markdownUrl: exerciseMarkdownUrl || "",
                      docxFilename: exerciseArtifact.docx_filename || "",
                      docxUrl: exerciseDocxUrl || "",
                    }
                  : msg.exercise,
              }
            : msg
        )
      )
      return
    }

    if (data.type === "done") {
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: status.state === "error" ? "error" : "done",
        summary: "个性化学习资源生成状态已更新。",
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
            state: "running",
            summary: copy?.detail ?? "多智能体正在推进个性化学习资源生成。",
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
  }, [setActiveThreadId, updateAssistantResourceStatus])

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
    setIsInterrupted(false)
    setInterruptDraft("")
    setLogs((prev) => [
      ...prev,
      { type: "info" as const, message: `[INFO] 用户问题：${content.slice(0, 60)}`, ts: timestamp() },
    ])
    console.debug("[A3_CHAT] sending", { threadId, selectedChatId, messageCount: messages.length + 1 })

    setIsLoading(true)

    try {
      streamHadErrorRef.current = false
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ query: content, thread_id: threadId }),
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
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] ${error.message}`, ts: timestamp() },
      ])
    } finally {
      setIsLoading(false)
    }
  }, [selectedChatId, messages.length, fetchWithErrorHandling, consumeSSEStream])

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

  return (
    <div className="a3-app-shell flex overflow-hidden">
      <LeftSidebar
        chatHistory={chatHistory}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onClearChatHistory={handleClearChatHistory}
        selectedChatId={selectedChatId}
      />
      <div className="flex min-w-0 flex-1 flex-col h-full">
        <ChatArea
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading && !isInterrupted}
        />
        {isInterrupted && (
          <PlanReview
            draft={interruptDraft}
            onConfirm={handleResume}
            onFeedback={handleFeedback}
            isSubmitting={isResuming}
          />
        )}
      </div>
      <RightPanel
        logs={logs}
        nodeEvents={nodeEvents}
        tokenUsage={tokenUsage}
        isInterrupted={isInterrupted}
      />
    </div>
  )
}
