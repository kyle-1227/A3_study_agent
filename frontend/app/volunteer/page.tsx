"use client"

import { useEffect, useRef, useState, Suspense, type ReactNode } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import {
  ArrowLeft,
  Bot,
  ChevronDown,
  FileText,
  GraduationCap,
  Home,
  Loader2,
  MapPin,
  Paperclip,
  RotateCcw,
  Send,
  Target,
  User,
  X,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { getVolunteerHistory, saveVolunteerHistory, type VolunteerHistoryItem } from "@/components/left-sidebar"
import { cn } from "@/lib/utils"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const REGIONS = [
  "北京",
  "天津",
  "河北",
  "山西",
  "内蒙古",
  "辽宁",
  "吉林",
  "黑龙江",
  "上海",
  "江苏",
  "浙江",
  "安徽",
  "福建",
  "江西",
  "山东",
  "河南",
  "湖北",
  "湖南",
  "广东",
  "广西",
  "海南",
  "重庆",
  "四川",
  "贵州",
  "云南",
  "西藏",
  "陕西",
  "甘肃",
  "青海",
  "宁夏",
  "新疆",
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
  const compact = content.replace(/\n/g, " ").trim()
  if (!compact) return "志愿填报咨询"
  return compact.slice(0, 30) + (compact.length > 30 ? "..." : "")
}

function buildWelcomeContent(targetRegion: string, homeRegion: string): string {
  return `你好，我已了解你的志愿填报咨询背景：\n\n- **报考目标地区**：${targetRegion}\n- **考生所在地**：${homeRegion}\n\n你可以直接提问，也可以上传成绩单、排名证明、招生章程等资料。我会结合地区差异、院校信息和你提供的材料，帮你梳理更清晰的志愿填报思路。`
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
  const [currentChatId, setCurrentChatId] = useState("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])

  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const loadedChatRef = useRef(false)

  const canStart = Boolean(targetRegion && homeRegion)

  useEffect(() => {
    if (loadedChatRef.current || !chatIdParam) return
    loadedChatRef.current = true

    const history = getVolunteerHistory()
    const item = history.find((h) => h.id === chatIdParam)
    if (!item) {
      router.replace("/volunteer")
      return
    }

    const msgs = loadMessages(chatIdParam)
    setTargetRegion(item.targetRegion as Region)
    setHomeRegion(item.homeRegion as Region)
    setCurrentChatId(chatIdParam)
    setMessages(
      msgs.length > 0
        ? msgs
        : [
            {
              id: "welcome",
              role: "assistant",
              content: buildWelcomeContent(item.targetRegion, item.homeRegion),
            },
          ],
    )
    setPhase("chat")
  }, [chatIdParam, router])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages])

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
    const initialMessages: ChatMessage[] = [
      {
        id: "welcome",
        role: "assistant",
        content: buildWelcomeContent(targetRegion, homeRegion),
      },
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
      content: userContent || "请结合我上传的资料，给我志愿填报建议。",
      files,
    }

    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)
    setInput("")
    setUploadedFiles([])
    setIsLoading(true)

    if (currentChatId) {
      const history = getVolunteerHistory()
      const item = history.find((h) => h.id === currentChatId)
      if (item && item.title.startsWith("志愿咨询 - ")) {
        upsertHistory(currentChatId, generateTitle(userContent), targetRegion, homeRegion)
      }
    }

    let query = `我正在进行高考志愿填报咨询。我的报考目标地区是${targetRegion}，考生所在地是${homeRegion}。`
    for (const file of files) {
      if (file.content) {
        query += `\n\n以下是我上传的文件「${file.name}」的内容：\n${file.content}`
      }
    }
    query += `\n\n${userContent || "请结合我的地区信息和上传资料，给我志愿填报建议。"}`

    const assistantId = (Date.now() + 1).toString()
    setMessages([...updatedMessages, { id: assistantId, role: "assistant", content: "" }])

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
          if (!part.startsWith("data: ")) continue
          try {
            const data = JSON.parse(part.slice(6))
            if (data.type === "token") {
              setMessages((prev) =>
                prev.map((msg) => (msg.id === assistantId ? { ...msg, content: msg.content + data.content } : msg)),
              )
            } else if (data.type === "text") {
              setMessages((prev) =>
                prev.map((msg) => (msg.id === assistantId ? { ...msg, content: data.content } : msg)),
              )
            }
          } catch {
            // Skip malformed chunks; the stream may still continue normally.
          }
        }
      }
    } catch {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantId ? { ...msg, content: "抱歉，咨询请求遇到异常，请稍后重试。" } : msg,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files) return

    for (const file of files) {
      if (file.size > 5 * 1024 * 1024) {
        alert(`文件 "${file.name}" 超过 5MB 限制，已跳过。`)
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
    setUploadedFiles((prev) => prev.filter((file) => file.id !== id))
  }

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  if (phase === "select") {
    return (
      <div className="a3-app-shell flex flex-col">
        <header className="border-b border-[var(--border)] bg-[var(--surface)]/95 px-5 py-4">
          <div className="mx-auto flex max-w-5xl items-center justify-between gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push("/")}
              className="gap-2 rounded-full text-[var(--muted-foreground)] hover:bg-[var(--muted)] hover:text-[var(--primary)]"
            >
              <ArrowLeft className="h-4 w-4" />
              返回主助手
            </Button>
            <div className="flex items-center gap-2 text-sm font-medium text-[var(--primary)]">
              <GraduationCap className="h-5 w-5" />
              高考志愿填报
            </div>
          </div>
        </header>

        <main className="flex flex-1 items-center justify-center px-5 py-10">
          <section className="w-full max-w-3xl">
            <div className="mb-8 text-center">
              <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--primary)] text-[var(--primary-foreground)] shadow-sm">
                <GraduationCap className="h-7 w-7" />
              </div>
              <h1 className="text-2xl font-semibold tracking-tight text-[var(--foreground)]">选择地区信息</h1>
              <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-[var(--muted-foreground)]">
                志愿填报需要同时考虑报考地区、考生所在地、招生政策和分数线差异。请先选择地区，再进入独立咨询对话。
              </p>
            </div>

            <div className="a3-panel mx-auto max-w-xl p-5">
              <div className="grid gap-5 md:grid-cols-2">
                <RegionField
                  icon={<Target className="h-4 w-4" />}
                  label="报考目标地区"
                  value={targetRegion}
                  onChange={setTargetRegion}
                  isOpen={targetOpen}
                  onToggle={() => {
                    setTargetOpen(!targetOpen)
                    setHomeOpen(false)
                  }}
                  placeholder="选择目标地区"
                />
                <RegionField
                  icon={<MapPin className="h-4 w-4" />}
                  label="考生所在地"
                  value={homeRegion}
                  onChange={setHomeRegion}
                  isOpen={homeOpen}
                  onToggle={() => {
                    setHomeOpen(!homeOpen)
                    setTargetOpen(false)
                  }}
                  placeholder="选择考生所在地"
                />
              </div>

              <Button
                onClick={handleStartChat}
                disabled={!canStart}
                className="a3-button-primary mt-6 h-11 w-full rounded-xl text-sm font-medium disabled:pointer-events-none disabled:opacity-45"
              >
                开始志愿填报咨询
              </Button>

              {!canStart && (
                <p className="mt-3 text-center text-xs text-[var(--muted-foreground)]">
                  请先选择报考目标地区和考生所在地。
                </p>
              )}
            </div>
          </section>
        </main>
      </div>
    )
  }

  return (
    <div className="a3-app-shell flex flex-col">
      <header className="border-b border-[var(--border)] bg-[var(--surface)]/95 px-4 py-3">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push("/")}
              className="shrink-0 gap-1.5 rounded-full text-[var(--muted-foreground)] hover:bg-[var(--muted)] hover:text-[var(--primary)]"
            >
              <Home className="h-4 w-4" />
              <span className="hidden text-xs sm:inline">回到主助手</span>
            </Button>
            <div className="flex min-w-0 items-center gap-2">
              <GraduationCap className="h-5 w-5 shrink-0 text-[var(--primary)]" />
              <h1 className="truncate text-base font-semibold text-[var(--foreground)]">志愿填报咨询</h1>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={handleBackToSelect}
              className="h-8 w-8 rounded-full text-[var(--muted-foreground)] hover:bg-[var(--muted)] hover:text-[var(--primary)]"
              title="切换地区"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <RegionChip icon={<Target className="h-3 w-3" />} label={targetRegion} />
            <RegionChip icon={<MapPin className="h-3 w-3" />} label={homeRegion} muted />
          </div>
        </div>
      </header>

      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4">
        <div className="mx-auto flex max-w-3xl flex-col gap-5 py-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn("flex items-start gap-3", message.role === "user" && "flex-row-reverse")}
            >
              <div
                className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
                  message.role === "user"
                    ? "bg-[var(--primary)] text-[var(--primary-foreground)]"
                    : "bg-[var(--primary-soft)] text-[var(--primary)]",
                )}
              >
                {message.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
              </div>
              <div
                className={cn(
                  "max-w-[82%] rounded-2xl px-4 py-3 text-sm leading-7 shadow-sm",
                  message.role === "user"
                    ? "rounded-tr-sm bg-[var(--primary)] text-[var(--primary-foreground)]"
                    : "rounded-tl-sm border border-[var(--border)] bg-[var(--surface)] text-[var(--foreground)]",
                )}
              >
                {message.files && message.files.length > 0 && (
                  <div className="mb-2 flex flex-wrap gap-2">
                    {message.files.map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center gap-1.5 rounded-md bg-black/5 px-2 py-1 text-xs"
                      >
                        <FileText className="h-3 w-3" />
                        <span className="max-w-[150px] truncate">{file.name}</span>
                        <span className="opacity-65">({formatFileSize(file.size)})</span>
                      </div>
                    ))}
                  </div>
                )}
                {message.content ? (
                  <div className="whitespace-pre-wrap">{message.content}</div>
                ) : (
                  <div className="flex items-center gap-1.5 py-1">
                    <span className="h-2 w-2 animate-bounce rounded-full bg-[var(--primary)]/60 [animation-delay:0ms]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-[var(--primary)]/60 [animation-delay:150ms]" />
                    <span className="h-2 w-2 animate-bounce rounded-full bg-[var(--primary)]/60 [animation-delay:300ms]" />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {uploadedFiles.length > 0 && (
        <div className="px-4">
          <div className="mx-auto flex max-w-3xl flex-wrap gap-2 pb-2">
            {uploadedFiles.map((file) => (
              <div
                key={file.id}
                className="flex items-center gap-1.5 rounded-lg border border-[var(--border)] bg-[var(--primary-soft)] px-3 py-1.5 text-xs text-[var(--primary)]"
              >
                <FileText className="h-3.5 w-3.5" />
                <span className="max-w-[160px] truncate">{file.name}</span>
                <span className="text-[var(--muted-foreground)]">({formatFileSize(file.size)})</span>
                <button
                  type="button"
                  onClick={() => handleRemoveFile(file.id)}
                  className="ml-1 rounded-full p-0.5 hover:bg-[var(--surface)] hover:text-[var(--danger)]"
                  aria-label={`移除 ${file.name}`}
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="border-t border-[var(--border)] bg-[var(--background)] px-4 py-4">
        <form
          onSubmit={(event) => {
            event.preventDefault()
            handleSend()
          }}
          className="mx-auto max-w-3xl"
        >
          <div className="overflow-hidden rounded-3xl border border-[var(--border)] bg-[var(--surface-2)] shadow-sm">
            <div className="px-4 pt-4">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={uploadedFiles.length > 0 ? "输入你的志愿填报问题..." : "输入问题，或上传个人资料..."}
                rows={2}
                className="max-h-[160px] min-h-[52px] w-full resize-none bg-transparent text-sm leading-6 text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] focus:outline-none"
              />
            </div>

            <div className="flex items-center gap-1 px-3 pb-3">
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
                className="h-9 w-9 rounded-full text-[var(--muted-foreground)] hover:bg-[var(--surface)] hover:text-[var(--primary)]"
              >
                <Paperclip className="h-5 w-5" />
              </Button>

              <div className="flex-1" />

              <Button
                type="submit"
                size="icon"
                disabled={(!input.trim() && uploadedFiles.length === 0) || isLoading}
                className="a3-button-primary h-9 w-9 rounded-full disabled:pointer-events-none disabled:opacity-50"
              >
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        </form>
        <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-[var(--muted-foreground)]">
          支持上传 TXT、CSV、MD、JSON、PDF、DOC、DOCX 文件，单文件不超过 5MB。上传资料仅用于本次咨询。
        </p>
      </div>
    </div>
  )
}

function RegionField({
  icon,
  label,
  value,
  onChange,
  isOpen,
  onToggle,
  placeholder,
}: {
  icon: ReactNode
  label: string
  value: string
  onChange: (value: Region) => void
  isOpen: boolean
  onToggle: () => void
  placeholder: string
}) {
  return (
    <div>
      <label className="mb-2 flex items-center gap-2 text-sm font-medium text-[var(--foreground)]">
        <span className="text-[var(--primary)]">{icon}</span>
        {label}
      </label>
      <RegionSelect value={value} onChange={onChange} isOpen={isOpen} onToggle={onToggle} placeholder={placeholder} />
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
  onChange: (value: Region) => void
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
          "a3-focus-ring flex w-full items-center justify-between rounded-xl border bg-[var(--surface)] px-4 py-3 text-sm transition-colors",
          value
            ? "border-[var(--primary)] text-[var(--foreground)]"
            : "border-[var(--border)] text-[var(--muted-foreground)]",
          "hover:border-[var(--primary)]",
        )}
      >
        <span>{value || placeholder}</span>
        <ChevronDown
          className={cn("h-4 w-4 text-[var(--muted-foreground)] transition-transform", isOpen && "rotate-180")}
        />
      </button>
      {isOpen && (
        <div className="a3-popover-shadow absolute left-0 right-0 top-full z-30 mt-2 max-h-60 overflow-y-auto rounded-xl border border-[var(--border)] bg-[var(--surface)] py-1">
          {REGIONS.map((region) => (
            <button
              key={region}
              type="button"
              onClick={() => onChange(region)}
              className={cn(
                "w-full px-4 py-2.5 text-left text-sm transition-colors",
                region === value
                  ? "bg-[var(--primary-soft)] font-medium text-[var(--primary)]"
                  : "text-[var(--foreground)] hover:bg-[var(--muted)]",
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

function RegionChip({ icon, label, muted = false }: { icon: ReactNode; label: string; muted?: boolean }) {
  return (
    <span
      className={cn(
        "hidden items-center gap-1 rounded-full px-2.5 py-1 text-xs sm:flex",
        muted ? "bg-[var(--muted)] text-[var(--muted-foreground)]" : "bg-[var(--primary-soft)] text-[var(--primary)]",
      )}
    >
      {icon}
      {label}
    </span>
  )
}

export default function VolunteerPage() {
  return (
    <Suspense
      fallback={
        <div className="a3-app-shell flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-[var(--primary)]" />
        </div>
      }
    >
      <VolunteerPageInner />
    </Suspense>
  )
}
