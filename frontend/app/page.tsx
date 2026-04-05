"use client"

import { useState, useCallback, useRef } from "react"
import { LeftSidebar } from "@/components/left-sidebar"
import { RightPanel, NodeEvent, LogEntry } from "@/components/right-panel"
import { ChatArea, Message } from "@/components/chat-area"
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

    if (data.type === "done") {
      return
    }

    if (data.type === "error") {
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
  }, [])

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
        { id: assistantMessageId, role: "assistant", content: "" },
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
