"use client"

import { useState, useCallback, useRef } from "react"
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

const initialChatHistory: any[] = []

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
  supervisor: {
    title: "解析学习需求",
    detail: "识别专业课程、学习目标、知识短板和需要生成的资源类型。",
  },
  academic_router: {
    title: "选择课程资源链路",
    detail: "判断当前任务更适合课程讲解、资料生成、练习设计还是综合答疑。",
  },
  search_query_rewriter: {
    title: "改写检索查询",
    detail: "将学习者原始问题转换为适合课程知识库和网络搜索的精准查询。",
  },
  rag_retrieve: {
    title: "检索课程知识库",
    detail: "从初始课程文档集和知识库中抽取可引用的课程依据。",
  },
  web_search: {
    title: "补充前沿资料",
    detail: "补充高校课程、项目实践和拓展阅读相关参考信息。",
  },
  generate_answer: {
    title: "生成学习资源",
    detail: "生成课程讲解、练习题、实操案例、拓展阅读或多模态脚本内容。",
  },
  evaluate_hallucination: {
    title: "内容可信校验",
    detail: "检查学术事实、引用依据和内容安全，降低幻觉风险。",
  },
  rewrite_query: {
    title: "重写检索问题",
    detail: "根据校验反馈补全查询，准备再次检索关键课程资料。",
  },
  gather_planning_context: {
    title: "检索规划上下文",
    detail: "检索课程资料、学习目标、资源约束和学习路径参考。",
  },
  gather_intel: {
    title: "汇总画像与情报",
    detail: "整合学习者基础、知识短板、目标课程和资源生成需求。",
  },
  drafter: {
    title: "起草个性化方案",
    detail: "生成学习路径、资源清单、练习安排和项目实践建议。",
  },
  reviewer_academic: {
    title: "学术质量审查",
    detail: "审查知识准确性、课程逻辑、难度递进和资源覆盖度。",
  },
  reviewer_emotional: {
    title: "学习负荷审查",
    detail: "检查学习节奏、任务压力和执行可持续性。",
  },
  consensus_check: {
    title: "协同评审汇总",
    detail: "汇总多智能体审查结论，判断方案是否可以输出。",
  },
  adv_rewrite: {
    title: "修订资源方案",
    detail: "根据审查意见优化学习路径、资源顺序和任务颗粒度。",
  },
  plan_output: {
    title: "输出资源方案",
    detail: "整理最终的个性化学习路径与多类型资源生成结果。",
  },
  feedback_router: {
    title: "分析用户反馈",
    detail: "判断反馈需要局部微调还是重新生成资源方案。",
  },
  plan_tweak: {
    title: "微调学习方案",
    detail: "根据反馈更新学习路径、资源推荐和练习安排。",
  },
  mindmap_agent: {
    title: "生成 JSON Tree",
    detail: "将知识结构蓝图转换为统一 JSON Tree，供多格式导图预览与导出使用。",
  },
  mindmap_planner: {
    title: "规划知识结构蓝图",
    detail: "整合课程资料、关键词和学习目标，规划具体到知识点的导图结构。",
  },
  mindmap_reviewer: {
    title: "审查导图质量",
    detail: "检查导图层级、具体知识点覆盖、易错辨析、实践案例和学术准确性。",
  },
  mindmap_rewrite: {
    title: "根据审查意见重写",
    detail: "根据审查反馈补充课程核心知识点，优化分支层级和资源可用性。",
  },
  mindmap_output: {
    title: "导出多格式导图",
    detail: "生成 XMind 下载文件，并准备 Mermaid、Markdown、SVG、PNG 和交互树预览。",
  },
  exercise_planner: {
    title: "规划练习结构",
    detail: "结合课程资料、关键词和学习目标，规划基础题、进阶题、应用题和自我检查题。",
  },
  exercise_agent: {
    title: "生成分层题目",
    detail: "生成包含答案、解析和易错提醒的分层练习题。",
  },
  exercise_reviewer: {
    title: "审查题目质量",
    detail: "检查题型覆盖、难度递进、答案解析、易错提醒和课程主题匹配度。",
  },
  exercise_rewrite: {
    title: "修订练习题",
    detail: "根据审查意见补齐题型层级、解析细节和易错提醒。",
  },
  exercise_output: {
    title: "输出练习资源",
    detail: "整理最终分层练习题，包含基础题、进阶题、应用题、自我检查题和解析。",
  },
  emotional_response: {
    title: "生成学习支持建议",
    detail: "围绕学习压力、专业适应和执行困难生成支持性建议。",
  },
  handle_unknown: {
    title: "确认服务范围",
    detail: "判断请求是否属于高校课程学习与个性化资源生成范围。",
  },
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
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([
    { type: "info", message: "[INFO] System initialized.", ts: timestamp() },
  ])
  const [nodeEvents, setNodeEvents] = useState<NodeEvent[]>([])
  const [tokenUsage, setTokenUsage] = useState({ input: 0, output: 0, total: 0 })

  // HIL state
  const [isInterrupted, setIsInterrupted] = useState(false)
  const [interruptDraft, setInterruptDraft] = useState("")
  const [isResuming, setIsResuming] = useState(false)
  const threadIdRef = useRef<string | null>(null)
  const assistantMessageIdRef = useRef<string>("")

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
    setLogs([{ type: "info", message: "[INFO] New chat session started.", ts: timestamp() }])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setIsInterrupted(false)
    setInterruptDraft("")
    threadIdRef.current = null
  }, [])

  const handleSelectChat = useCallback((id: string) => {
    setSelectedChatId(id)
    setMessages([])
    setNodeEvents([])
    setIsInterrupted(false)
    setInterruptDraft("")
    threadIdRef.current = null
  }, [])

  /** Process a single SSE data payload — shared between /stream and /resume */
  const processSSEEvent = useCallback((data: any) => {
    const asstId = assistantMessageIdRef.current

    if (data.type === "thread_id") {
      threadIdRef.current = data.thread_id
      setLogs((prev) => [
        ...prev,
        { type: "info", message: `[INFO] Thread: ${data.thread_id.slice(0, 8)}...`, ts: timestamp() },
      ])
      return
    }

    if (data.type === "interrupt") {
      setInterruptDraft(data.draft)
      setIsInterrupted(true)
      if (data.thread_id) threadIdRef.current = data.thread_id
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: "waiting_review",
        summary: "个性化学习路径与资源方案已生成，等待确认或反馈后继续优化。",
        waitingForReview: true,
      }))
      setLogs((prev) => [
        ...prev,
        { type: "warning", message: "[HIL] Graph interrupted — awaiting user plan review", ts: timestamp() },
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
                  title: data.title || "知识点思维导图",
                  tree: data.tree,
                  xmindUrl,
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
      const xmindUrl =
        mindmap && typeof mindmap.xmind_url === "string" && mindmap.xmind_url.startsWith("/")
          ? `${API_BASE_URL}${mindmap.xmind_url}`
          : mindmap?.xmind_url

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === asstId
            ? {
                ...msg,
                content: finalAnswer || msg.content,
                mindmap: mindmap
                  ? {
                      title: mindmap.title || "鐭ヨ瘑鐐规€濈淮瀵煎浘",
                      tree: mindmap.tree,
                      xmindUrl: xmindUrl || "",
                    }
                  : msg.mindmap,
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
        summary: "个性化学习资源已生成，可继续根据反馈调整。",
        waitingForReview: false,
      }))
      return
    }

    if (data.type === "error") {
      updateAssistantResourceStatus(asstId, (status) => ({
        ...status,
        state: "error",
        summary: "个性化资源生成遇到异常，请稍后重试或调整输入。",
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
        return prev.map((e) =>
          e.node === node && e.status === "running"
            ? { ...e, status: "done", endTs: now, durationMs: data.duration_ms ?? undefined }
            : e
        )
      })

      const label = status === "start" ? "Entering" : "Leaving"
      setLogs((prev) => [
        ...prev,
        { type: "info", message: `[INFO] ${label} node: ${node}`, ts: now },
      ])

      if (status === "end" && data.duration_ms != null) {
        setLogs((prev) => [
          ...prev,
          { type: "perf", message: `[PERF] Node "${node}" completed in ${data.duration_ms}ms`, ts: now },
        ])
      }

      if (status === "end" && data.error) {
        setLogs((prev) => [
          ...prev,
          { type: "error", message: `[ERROR] Node "${node}": ${data.error}`, ts: now },
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
        { type: "usage", message: `[USAGE] ${data.node}: ${data.input_tokens} in / ${data.output_tokens} out`, ts: now },
      ])
    }
  }, [updateAssistantResourceStatus])

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
        { id: (Date.now() + 1).toString(), role: "assistant", content: "⚠️ 服务繁忙，请稍后重试。" },
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
        { id: (Date.now() + 1).toString(), role: "assistant", content: "🔑 访问未授权，请检查访问令牌是否正确。" },
      ])
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] 401 Unauthorized — invalid or missing access token", ts: timestamp() },
      ])
      if (typeof window !== "undefined") localStorage.removeItem("demo_access_token")
      return null
    }

    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    if (!response.body) throw new Error("No response body")

    return response.body
  }, [])

  const handleSendMessage = useCallback(async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
    }

    setMessages((prev) => [...prev, userMessage])
    setNodeEvents([])
    setTokenUsage({ input: 0, output: 0, total: 0 })
    setIsInterrupted(false)
    setInterruptDraft("")
    threadIdRef.current = null
    setLogs((prev) => [
      ...prev,
      { type: "info" as const, message: `[INFO] User query: ${content.slice(0, 60)}`, ts: timestamp() },
    ])

    setIsLoading(true)

    try {
      const body = await fetchWithErrorHandling(`${API_BASE_URL}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ query: content }),
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
        { type: "info", message: "[INFO] Stream complete.", ts: timestamp() },
      ])
    } catch (error: any) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: `[ERROR] ${error.message}`, ts: timestamp() },
      ])
    } finally {
      setIsLoading(false)

      if (!selectedChatId) {
        const newChat = {
          id: Date.now().toString(),
          title: content.slice(0, 30) + (content.length > 30 ? "..." : ""),
        }
        setChatHistory((prev) => [newChat, ...prev])
        setSelectedChatId(newChat.id)
      }
    }
  }, [selectedChatId, fetchWithErrorHandling, consumeSSEStream])

  const handleResume = useCallback(async (editedPlan: string) => {
    const threadId = threadIdRef.current
    if (!threadId) {
      setLogs((prev) => [
        ...prev,
        { type: "error", message: "[ERROR] No thread_id — cannot resume", ts: timestamp() },
      ])
      return
    }

    setIsResuming(true)
    setLogs((prev) => [
      ...prev,
      { type: "info", message: "[INFO] Resuming graph with edited plan...", ts: timestamp() },
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
        { type: "info", message: "[INFO] Resume stream complete.", ts: timestamp() },
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
        { type: "error", message: "[ERROR] No thread_id — cannot send feedback", ts: timestamp() },
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
        { type: "info", message: "[INFO] Feedback revision complete.", ts: timestamp() },
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
    <div className="flex h-screen overflow-hidden">
      <LeftSidebar
        chatHistory={chatHistory}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        selectedChatId={selectedChatId}
      />
      <div className="flex-1 flex flex-col h-full">
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
