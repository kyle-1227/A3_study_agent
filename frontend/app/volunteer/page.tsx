"use client"

import { useState, useRef, useEffect, Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import {
  ArrowLeft,
  GraduationCap,
  MapPin,
  Target,
  ChevronDown,
  Send,
  Bot,
  User,
  Paperclip,
  X,
  FileText,
  Loader2,
  RotateCcw,
  Home,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { getVolunteerHistory, saveVolunteerHistory, type VolunteerHistoryItem } from "@/components/left-sidebar"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const REGIONS = [
  "北京", "天津", "河北", "山西", "内蒙古",
  "辽宁", "吉林", "黑龙江",
  "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东",
  "河南", "湖北", "湖南", "广东", "广西", "海南",
  "重庆", "四川", "贵州", "云南", "西藏",
  "陕西", "甘肃", "青海", "宁夏", "新疆",
] as const

type Region = (typeof REGIONS)[number]

interface UploadedFile {
  id: string
  name: string
  size: number
  content?: string
}

interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  files?: UploadedFile[]
}

const CHAT_PREFIX = "volunteer_chat_"

function getAuthHeaders(): Record<string, string> {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("demo_access_token")
    if (token) return { "X-Access-Token": token }
  }
  return {}
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
}

function loadMessages(chatId: string): ChatMessage[] {
  try {
    const raw = localStorage.getItem(CHAT_PREFIX + chatId)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveMessages(chatId: string, messages: ChatMessage[]) {
  localStorage.setItem(CHAT_PREFIX + chatId, JSON.stringify(messages))
}

function generateTitle(content: string): string {
  return content.replace(/\n/g, " ").slice(0, 30) + (content.length > 30 ? "..." : "")
}

function VolunteerPageInner() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const chatIdParam = searchParams.get("chatId")

  const [phase, setPhase] = useState<"select" | "chat">("select")
  const [targetRegion, setTargetRegion] = useState<Region | "">("")
  const [homeRegion, setHomeRegion] = useState<Region | "">("")
  const [targetOpen, setTargetOpen] = useState(false)
  const [homeOpen, setHomeOpen] = useState(false)
  const [currentChatId, setCurrentChatId] = useState<string>("")

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const loadedChatRef = useRef(false)

  const canStart = targetRegion && homeRegion

  // Load existing chat from URL param
  useEffect(() => {
    if (loadedChatRef.current || !chatIdParam) return
    loadedChatRef.current = true

    const history = getVolunteerHistory()
    const item = history.find((h) => h.id === chatIdParam)
    if (!item) {
      // Chat not found, redirect to fresh volunteer page
      router.replace("/volunteer")
      return
    }

    const msgs = loadMessages(chatIdParam)
    setTargetRegion(item.targetRegion as Region)
    setHomeRegion(item.homeRegion as Region)
    setCurrentChatId(chatIdParam)
    setMessages(msgs.length > 0 ? msgs : [
      {
        id: "welcome",
        role: "assistant",
        content: `欢迎回来！继续你的志愿填报咨询：\n\n- **报考目标地区**：${item.targetRegion}\n- **考生所在地区**：${item.homeRegion}\n\n请继续提问或上传资料。`,
      },
    ])
    setPhase("chat")
  }, [chatIdParam, router])

  // Auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages])

  // Persist messages when they change
  useEffect(() => {
    if (currentChatId && messages.length > 0) {
      saveMessages(currentChatId, messages)
    }
  }, [messages, currentChatId])

  const upsertHistory = (chatId: string, title: string, target: string, home: string) => {
    const history = getVolunteerHistory()
    const existing = history.findIndex((h) => h.id === chatId)
    const entry: VolunteerHistoryItem = { id: chatId, title, targetRegion: target, homeRegion: home }
    if (existing >= 0) {
      history[existing] = entry
      // Move to top
      history.unshift(history.splice(existing, 1)[0])
    } else {
      history.unshift(entry)
    }
    saveVolunteerHistory(history)
    window.dispatchEvent(new Event("storage"))
    return entry
  }

  const handleStartChat = () => {
    if (!canStart) return

    const chatId = Date.now().toString()
    const welcomeContent = `你好！我已了解你的志愿填报需求：\n\n- **报考目标地区**：${targetRegion}\n- **考生所在地区**：${homeRegion}\n\n你可以：\n1. 直接输入问题，如"${targetRegion}有哪些适合我的院校？"、"${homeRegion}考生报考${targetRegion}的分数线是多少？"\n2. 上传个人资料（成绩单、排名证明等），我会结合你的实际情况给出更精准的建议。\n\n请随时向我提问！`

    const initialMessages: ChatMessage[] = [
      { id: "welcome", role: "assistant", content: welcomeContent },
    ]

    setCurrentChatId(chatId)
    setMessages(initialMessages)
    setPhase("chat")
    saveMessages(chatId, initialMessages)
    upsertHistory(chatId, `志愿咨询 - ${targetRegion}`, targetRegion, homeRegion)
  }

  const handleBackToSelect = () => {
    setPhase("select")
    setMessages([])
    setUploadedFiles([])
    setCurrentChatId("")
  }

  const handleSend = async () => {
    const hasText = input.trim()
    const hasFiles = uploadedFiles.length > 0
    if ((!hasText && !hasFiles) || isLoading) return

    const userContent = input.trim()
    const files = [...uploadedFiles]

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: userContent || "请结合我上传的资料，给我志愿填报建议",
      files,
    }

    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)
    setInput("")
    setUploadedFiles([])
    setIsLoading(true)

    // Update history title based on first user message
    if (currentChatId) {
      const history = getVolunteerHistory()
      const item = history.find((h) => h.id === currentChatId)
      if (item && item.title.startsWith("志愿咨询 - ")) {
        upsertHistory(currentChatId, generateTitle(userContent), targetRegion, homeRegion)
      }
    }

    // Build natural-language query with region context
    let query = `我正在进行高考志愿填报咨询。我的报考目标地区是${targetRegion}，考生所在地区是${homeRegion}。`
    for (const f of files) {
      if (f.content) {
        query += `\n\n以下是我上传的文件「${f.name}」的内容：\n${f.content}`
      }
    }
    query += `\n\n${userContent || "请结合我的地区信息和上传的资料，给我志愿填报建议"}`

    const assistantId = (Date.now() + 1).toString()
    const withAssistant = [...updatedMessages, { id: assistantId, role: "assistant" as const, content: "" }]
    setMessages(withAssistant)

    try {
      const response = await fetch(`${API_BASE_URL}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify({ query }),
      })

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body.getReader()
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
              if (data.type === "token") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId ? { ...m, content: m.content + data.content } : m
                  )
                )
              } else if (data.type === "text") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId ? { ...m, content: data.content } : m
                  )
                )
              }
            } catch { /* skip malformed chunks */ }
          }
        }
      }
    } catch (_err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: "抱歉，咨询请求遇到异常，请稍后重试。" }
            : m
        )
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return

    for (const file of files) {
      if (file.size > 5 * 1024 * 1024) {
        alert(`文件 "${file.name}" 超过 5MB 限制，已跳过`)
        continue
      }
      const reader = new FileReader()
      const id = Date.now().toString() + Math.random().toString(36).slice(2)
      reader.onload = () => {
        setUploadedFiles((prev) => [
          ...prev,
          { id, name: file.name, size: file.size, content: reader.result as string },
        ])
      }
      reader.readAsText(file)
    }
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleRemoveFile = (id: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== id))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Phase 1: Region Selection
  if (phase === "select") {
    return (
      <div className="flex flex-col h-screen bg-background">
        <header className="flex items-center gap-4 px-6 py-4 border-b border-[#C8D6C9] bg-white">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => router.push("/")}
            className="h-9 w-9 rounded-full text-muted-foreground hover:text-[#3D5A40]"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div className="flex items-center gap-2">
            <GraduationCap className="h-5 w-5 text-[#3D5A40]" />
            <h1 className="text-lg font-semibold text-[#3D5A40]">高考志愿填报</h1>
          </div>
        </header>

        <main className="flex-1 flex items-center justify-center px-6">
          <div className="w-full max-w-lg">
            <div className="text-center mb-10">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#3D5A40] to-[#5A7A5E] mx-auto mb-4">
                <GraduationCap className="h-8 w-8 text-[#FFCC99]" />
              </div>
              <h2 className="text-xl font-semibold text-[#3D5A40] mb-2">选择地区信息</h2>
              <p className="text-sm text-muted-foreground leading-relaxed">
                不同省份的高考难度、录取分数线和招生政策差异较大，请选择你的目标报考地区和考生所在地区，以便为你提供更精准的志愿填报建议。
              </p>
            </div>

            <div className="space-y-5">
              <div>
                <label className="flex items-center gap-2 text-sm font-medium text-[#2D2D2D] mb-2">
                  <Target className="h-4 w-4 text-[#3D5A40]" />
                  报考目标地区
                </label>
                <RegionSelect
                  value={targetRegion}
                  onChange={setTargetRegion}
                  isOpen={targetOpen}
                  onToggle={() => { setTargetOpen(!targetOpen); setHomeOpen(false) }}
                  placeholder="请选择报考地区"
                />
              </div>

              <div>
                <label className="flex items-center gap-2 text-sm font-medium text-[#2D2D2D] mb-2">
                  <MapPin className="h-4 w-4 text-[#5A7A5E]" />
                  考生所在地区
                </label>
                <RegionSelect
                  value={homeRegion}
                  onChange={setHomeRegion}
                  isOpen={homeOpen}
                  onToggle={() => { setHomeOpen(!homeOpen); setTargetOpen(false) }}
                  placeholder="请选择考生所在地区"
                />
              </div>
            </div>

            <Button
              onClick={handleStartChat}
              disabled={!canStart}
              className="w-full mt-8 h-12 rounded-xl bg-[#3D5A40] hover:bg-[#4A6B4D] text-white font-medium text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            >
              开始志愿填报咨询
            </Button>

            {!canStart && (
              <p className="text-center text-xs text-muted-foreground mt-3">
                请先选择报考地区与考生所在地区，再开始咨询
              </p>
            )}
          </div>
        </main>
      </div>
    )
  }

  // Phase 2: Consultation Chat
  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-[#C8D6C9] bg-white gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.push("/")}
            className="h-8 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50 gap-1.5 px-2 shrink-0"
          >
            <Home className="h-4 w-4" />
            <span className="text-xs">回主页面</span>
          </Button>
          <div className="flex items-center gap-2 min-w-0">
            <GraduationCap className="h-5 w-5 text-[#3D5A40] shrink-0" />
            <h1 className="text-base font-semibold text-[#3D5A40] truncate">志愿填报咨询</h1>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleBackToSelect}
            className="h-8 w-8 rounded-full text-muted-foreground hover:text-[#3D5A40]"
            title="切换地区"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
          <span className="flex items-center gap-1 bg-[#EDF5EE] rounded-full px-2 py-0.5 text-xs">
            <Target className="h-3 w-3" />
            {targetRegion}
          </span>
          <span className="flex items-center gap-1 bg-[#F5F3E8] rounded-full px-2 py-0.5 text-xs">
            <MapPin className="h-3 w-3" />
            {homeRegion}
          </span>
        </div>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 min-h-0">
        <div className="max-w-3xl mx-auto py-6 flex flex-col gap-5">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={cn("flex items-start gap-3", msg.role === "user" && "flex-row-reverse")}
            >
              <div className={cn(
                "flex h-8 w-8 items-center justify-center rounded-full shrink-0",
                msg.role === "user" ? "bg-[#3D5A40] text-white" : "bg-[#3D5A40]/10 text-[#3D5A40]"
              )}>
                {msg.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
              </div>
              <div className={cn(
                "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
                msg.role === "user"
                  ? "bg-[#3D5A40] text-white rounded-tr-sm"
                  : "bg-white border border-[#C8D6C9] text-[#2D2D2D] rounded-tl-sm"
              )}>
                {msg.files && msg.files.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-2">
                    {msg.files.map((f) => (
                      <div key={f.id} className="flex items-center gap-1.5 bg-white/20 rounded-md px-2 py-1 text-xs">
                        <FileText className="h-3 w-3" />
                        <span>{f.name}</span>
                        <span className="opacity-60">({formatFileSize(f.size)})</span>
                      </div>
                    ))}
                  </div>
                )}
                {msg.content ? (
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                ) : (
                  <div className="flex items-center gap-1">
                    <span className="w-2 h-2 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-2 h-2 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-2 h-2 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Uploaded files preview */}
      {uploadedFiles.length > 0 && (
        <div className="px-6">
          <div className="max-w-3xl mx-auto flex flex-wrap gap-2 pb-2">
            {uploadedFiles.map((f) => (
              <div key={f.id} className="flex items-center gap-1.5 bg-[#EDF5EE] border border-[#C8D6C9] rounded-lg px-3 py-1.5 text-xs text-[#3D5A40]">
                <FileText className="h-3.5 w-3.5" />
                <span className="max-w-[160px] truncate">{f.name}</span>
                <span className="text-muted-foreground">({formatFileSize(f.size)})</span>
                <button type="button" onClick={() => handleRemoveFile(f.id)} className="ml-1 hover:text-[#D97B6C]">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="bg-background px-6 py-4">
        <form
          onSubmit={(e) => { e.preventDefault(); handleSend() }}
          className="max-w-3xl mx-auto"
        >
          <div className="bg-[#F5F3E8] rounded-3xl overflow-hidden border border-[#E8E5D8]">
            <div className="px-4 pt-4 pb-2">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={uploadedFiles.length > 0 ? "输入你的志愿填报问题..." : "输入问题，或上传个人资料..."}
                rows={2}
                className={cn(
                  "w-full resize-none bg-transparent",
                  "text-sm text-foreground placeholder:text-muted-foreground",
                  "focus:outline-none",
                  "min-h-[50px] max-h-[160px]"
                )}
              />
            </div>

            <div className="flex items-center px-3 pb-3 gap-1">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.csv,.md,.json,.pdf,.doc,.docx"
                onChange={handleFileChange}
                className="hidden"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                className="h-9 w-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50"
              >
                <Paperclip className="h-5 w-5" />
              </Button>

              <div className="flex-1" />

              <Button
                type="submit"
                size="icon"
                disabled={(!input.trim() && uploadedFiles.length === 0) || isLoading}
                className="h-9 w-9 rounded-full bg-[#3D5A40] hover:bg-[#4A6B4D] text-white disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </form>
        <p className="text-center text-[11px] text-muted-foreground mt-2 max-w-3xl mx-auto">
          支持上传 TXT、CSV、MD、JSON、PDF、DOC、DOCX 文件，单文件不超过 5MB。上传的个人资料仅用于本次咨询。
        </p>
      </div>
    </div>
  )
}

function RegionSelect({
  value,
  onChange,
  isOpen,
  onToggle,
  placeholder,
}: {
  value: string
  onChange: (v: Region) => void
  isOpen: boolean
  onToggle: () => void
  placeholder: string
}) {
  return (
    <div className="relative">
      <button
        type="button"
        onClick={onToggle}
        className={cn(
          "w-full flex items-center justify-between rounded-xl border bg-white px-4 py-3 text-sm transition-colors",
          value ? "border-[#3D5A40] text-[#2D2D2D]" : "border-[#C8D6C9] text-muted-foreground",
          "hover:border-[#7A9E7E]"
        )}
      >
        <span>{value || placeholder}</span>
        <ChevronDown className={cn("h-4 w-4 text-muted-foreground transition-transform", isOpen && "rotate-180")} />
      </button>
      {isOpen && (
        <div className="absolute top-full left-0 right-0 z-30 mt-1 rounded-xl border border-[#C8D6C9] bg-white shadow-lg max-h-56 overflow-y-auto">
          {REGIONS.map((region) => (
            <button
              key={region}
              type="button"
              onClick={() => onChange(region)}
              className={cn(
                "w-full px-4 py-2.5 text-sm text-left transition-colors",
                region === value
                  ? "bg-[#EDF5EE] text-[#3D5A40] font-medium"
                  : "text-[#2D2D2D] hover:bg-[#F5F3E8]"
              )}
            >
              {region}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function VolunteerPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-screen bg-background">
        <Loader2 className="h-8 w-8 text-[#3D5A40] animate-spin" />
      </div>
    }>
      <VolunteerPageInner />
    </Suspense>
  )
}
