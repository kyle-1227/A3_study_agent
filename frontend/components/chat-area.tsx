"use client"

import { useState, useRef, useEffect, type ReactNode } from "react"
import { useRouter } from "next/navigation"
import {
  Send,
  Bot,
  User,
  Plus,
  SlidersHorizontal,
  Mic,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  CircleAlert,
  CircleDotDashed,
  Download,
  FileText,
  GitFork,
  Image,
  Loader2,
  Map,
  PauseCircle,
  GraduationCap,
} from "lucide-react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  resourceStatus?: ResourceGenerationStatus
  mindmap?: MindmapResult
}

export type ResourceGenerationState = "running" | "done" | "error" | "waiting_review"
export type ResourceGenerationStepState = "running" | "done" | "error"

export interface ResourceGenerationStep {
  node: string
  title: string
  detail: string
  state: ResourceGenerationStepState
  startedAt?: string
  endedAt?: string
  durationMs?: number
  error?: string
}

export interface ResourceGenerationStatus {
  state: ResourceGenerationState
  summary: string
  steps: ResourceGenerationStep[]
  tokenUsage: { input: number; output: number; total: number }
  error?: string
  waitingForReview?: boolean
}

export interface MindmapNode {
  title: string
  note?: string
  children?: MindmapNode[]
}

export interface MindmapResult {
  title: string
  tree: MindmapNode
  xmindUrl: string
}

interface ChatAreaProps {
  messages: Message[]
  onSendMessage: (content: string) => void
  isLoading?: boolean
}

export function ChatArea({ messages, onSendMessage, isLoading }: ChatAreaProps) {
  const [input, setInput] = useState("")
  const [isToolsOpen, setIsToolsOpen] = useState(false)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const router = useRouter()

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleMindmapTool = () => {
    const trimmed = input.trim()
    setIsToolsOpen(false)
    if (trimmed && !isLoading) {
      onSendMessage(`请生成知识点思维导图：${trimmed}`)
      setInput("")
      return
    }
    setInput("请生成知识点思维导图：")
    inputRef.current?.focus()
  }

  const handleVolunteerTool = () => {
    router.push("/volunteer")
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      {/* Messages Area — native scroll container with constrained height */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto px-8 min-h-0">
        <div className="max-w-3xl mx-auto py-6 flex flex-col gap-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-[60vh] text-center">
              {/* Phoenix Icon for Empty State */}
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#3D5A40] to-[#5A7A5E] mb-4">
                <svg width="36" height="36" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path 
                    d="M16 28C16 28 12 24 12 18C12 14 14 10 16 8" 
                    stroke="#FFCC99" 
                    strokeWidth="2" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 28C16 28 20 24 20 18C20 14 18 10 16 8" 
                    stroke="#FFCC99" 
                    strokeWidth="2" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 12C16 12 10 10 6 12C4 13 3 15 4 17C5 19 8 18 10 16C12 14 14 13 16 14" 
                    stroke="white" 
                    strokeWidth="1.8" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 12C16 12 22 10 26 12C28 13 29 15 28 17C27 19 24 18 22 16C20 14 18 13 16 14" 
                    stroke="white" 
                    strokeWidth="1.8" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path 
                    d="M16 8C16 8 14 5 16 3C18 5 16 8 16 8Z" 
                    fill="#FFCC99"
                    stroke="#FFCC99"
                    strokeWidth="1"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <circle cx="16" cy="14" r="1.5" fill="#FFCC99" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-[#3D5A40] mb-2">高校学习 AI 助手</h2>
              <p className="text-muted-foreground max-w-md leading-relaxed">
                我是你的高校个性化学习资源助手，可以协同生成课程讲解、思维导图、题库、实操案例、拓展阅读和学习路径。有什么课程目标或知识短板想先处理？
              </p>
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))
          )}
          {isLoading && messages[messages.length - 1]?.role === "user" && (
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#3D5A40]/10 text-[#3D5A40] flex-shrink-0">
                <Bot className="h-4 w-4" />
              </div>
              <div className="bg-white border border-[#C8D6C9] rounded-2xl rounded-tl-sm px-4 py-3">
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 bg-[#3D5A40]/60 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}
          {/* Scroll anchor */}
          <div ref={messagesEndRef} className="h-4 shrink-0" />
        </div>
      </div>

      {/* Input Area - Gemini Style with new palette */}
      <div className="bg-background px-8 py-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="bg-[#F5F3E8] rounded-3xl overflow-hidden border border-[#E8E5D8]">
            {/* Text Area at Top */}
            <div className="px-4 pt-4 pb-2">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入你的问题..."
                rows={2}
                className={cn(
                  "w-full resize-none bg-transparent",
                  "text-sm text-foreground placeholder:text-muted-foreground",
                  "focus:outline-none",
                  "min-h-[60px] max-h-[200px]"
                )}
              />
            </div>
            
            {/* Toolbar at Bottom */}
            <div className="flex items-center px-3 pb-3 gap-1">
              {/* Left Side: Plus and Tools */}
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-9 w-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50"
              >
                <Plus className="h-5 w-5" />
              </Button>
              <div className="relative">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsToolsOpen((open) => !open)}
                  className="h-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50 gap-1.5 px-3"
                >
                  <SlidersHorizontal className="h-4 w-4" />
                  <span className="text-sm">工具</span>
                </Button>
                {isToolsOpen && (
                  <div className="absolute bottom-11 left-0 z-20 w-52 rounded-lg border border-[#C8D6C9] bg-white p-1 shadow-lg">
                    <button
                      type="button"
                      onClick={handleMindmapTool}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm text-[#2D2D2D] hover:bg-[#EDF5EE] hover:text-[#3D5A40]"
                    >
                      <Map className="h-4 w-4" />
                      <span>生成思维导图</span>
                    </button>
                  </div>
                )}
              </div>
              
              {/* 志愿填报 */}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleVolunteerTool}
                className="h-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50 gap-1.5 px-3"
              >
                <GraduationCap className="h-4 w-4" />
                <span className="text-sm">志愿填报</span>
              </Button>

              {/* Flexible Spacer */}
              <div className="flex-1" />
              
              {/* Right Side: Mic, Send */}
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-9 w-9 rounded-full text-muted-foreground hover:text-[#3D5A40] hover:bg-white/50"
              >
                <Mic className="h-5 w-5" />
              </Button>
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isLoading}
                className={cn(
                  "h-9 w-9 rounded-full",
                  "bg-[#3D5A40] hover:bg-[#4A6B4D] text-white",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}

// Tailwind-styled overrides for react-markdown elements
const markdownComponents: Components = {
  h1: ({ children }) => <h1 className="text-lg font-bold mt-4 mb-2 text-[#3D5A40]">{children}</h1>,
  h2: ({ children }) => <h2 className="text-base font-bold mt-3 mb-1.5 text-[#3D5A40]">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mt-2 mb-1 text-[#3D5A40]">{children}</h3>,
  p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed break-words">{children}</p>,
  ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-1 break-words">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-1 break-words">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed break-words">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-[#3D5A40]">{children}</strong>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-3 border-[#7A9E7E] pl-3 my-2 text-muted-foreground italic">
      {children}
    </blockquote>
  ),
  code: ({ className, children }) => {
    // Fenced code blocks get a className like "language-xxx" from remark
    const isBlock = className?.startsWith("language-")
    if (isBlock) {
      return (
        <code className="block bg-[#F5F3E8] rounded-lg p-3 my-2 text-xs font-mono overflow-x-auto whitespace-pre">
          {children}
        </code>
      )
    }
    // Inline code
    return (
      <code className="bg-[#F5F3E8] rounded px-1.5 py-0.5 text-xs font-mono text-[#5C3D2E] whitespace-pre-wrap break-words [overflow-wrap:anywhere]">
        {children}
      </code>
    )
  },
  pre: ({ children }) => <pre className="my-2 max-w-full overflow-x-auto">{children}</pre>,
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="min-w-full text-xs border-collapse">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-[#C8D6C9] bg-[#F5F3E8] px-2 py-1 text-left font-semibold">{children}</th>
  ),
  td: ({ children }) => (
    <td className="border border-[#C8D6C9] px-2 py-1">{children}</td>
  ),
  hr: () => <hr className="border-[#C8D6C9] my-3" />,
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user"

  return (
    <div className={cn("flex items-start gap-3", isUser && "flex-row-reverse")}>
      <div className={cn(
        "flex h-8 w-8 items-center justify-center rounded-full flex-shrink-0",
        isUser ? "bg-[#3D5A40] text-white" : "bg-[#3D5A40]/10 text-[#3D5A40]"
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className={cn(
        "max-w-[80%] min-w-0 rounded-2xl px-4 py-3 text-sm leading-relaxed",
        isUser
          ? "bg-[#3D5A40] text-white rounded-tr-sm"
          : "bg-white border border-[#C8D6C9] text-[#2D2D2D] rounded-tl-sm"
      )}>
        {isUser ? (
          // User messages render as plain text
          <div className="whitespace-pre-wrap">{message.content}</div>
        ) : (
          <div className="min-w-0 space-y-3">
            {message.resourceStatus && (
              <ResourceGenerationStatusPanel status={message.resourceStatus} />
            )}
            {message.mindmap && <MindmapCard mindmap={message.mindmap} />}
            {message.content ? (
              // Assistant messages render as Markdown
              <div className="min-w-0 max-w-full break-words">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                  {message.content}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="text-muted-foreground">正在生成个性化学习资源...</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function MindmapCard({ mindmap }: { mindmap: MindmapResult }) {
  const [tab, setTab] = useState<"mermaid" | "markdown" | "tree">("mermaid")
  const mermaid = mindmapTreeToMermaid(mindmap.tree)
  const markdown = mindmapTreeToMarkdown(mindmap.tree)

  const downloadMarkdown = () => {
    downloadText(`${safeFileName(mindmap.title)}.md`, markdown, "text/markdown;charset=utf-8")
  }

  const downloadSvg = () => {
    downloadText(`${safeFileName(mindmap.title)}.svg`, mindmapTreeToSvg(mindmap.tree), "image/svg+xml;charset=utf-8")
  }

  const downloadPng = async () => {
    await downloadTreePng(mindmap.tree, `${safeFileName(mindmap.title)}.png`)
  }

  return (
    <div className="rounded-lg border border-[#C8D6C9] bg-[#F8FAF6] overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#C8D6C9] px-3 py-2.5">
        <div className="flex items-center gap-2 min-w-0">
          <GitFork className="h-4 w-4 text-[#3D5A40] shrink-0" />
          <div className="min-w-0">
            <p className="font-semibold text-[#3D5A40] truncate">知识点思维导图</p>
            <p className="text-xs text-muted-foreground truncate">{mindmap.title}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1">
          <DownloadButton href={mindmap.xmindUrl} label=".xmind" />
          <SmallButton onClick={downloadMarkdown} label=".md" icon={<FileText className="h-3.5 w-3.5" />} />
          <SmallButton onClick={downloadSvg} label=".svg" icon={<Image className="h-3.5 w-3.5" />} />
          <SmallButton onClick={downloadPng} label=".png" icon={<Image className="h-3.5 w-3.5" />} />
        </div>
      </div>

      <div className="flex gap-1 border-b border-[#C8D6C9] px-3 py-2">
        {(["mermaid", "markdown", "tree"] as const).map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => setTab(item)}
            className={cn(
              "rounded px-2 py-1 text-xs transition-colors",
              tab === item
                ? "bg-[#3D5A40] text-white"
                : "text-[#3D5A40] hover:bg-[#EDF5EE]"
            )}
          >
            {item === "mermaid" ? "Mermaid" : item === "markdown" ? "Markdown" : "交互树"}
          </button>
        ))}
      </div>

      <div className="max-h-80 overflow-auto p-3">
        {tab === "mermaid" && <MermaidPreview chart={mermaid} fallbackSvg={mindmapTreeToSvg(mindmap.tree)} />}
        {tab === "markdown" && (
          <pre className="whitespace-pre-wrap rounded bg-white p-3 text-xs leading-relaxed text-[#2D2D2D] border border-[#E8E5D8]">
            {markdown}
          </pre>
        )}
        {tab === "tree" && <InteractiveMindmapTree node={mindmap.tree} />}
      </div>
    </div>
  )
}

function MermaidPreview({ chart, fallbackSvg }: { chart: string; fallbackSvg: string }) {
  const [svg, setSvg] = useState("")

  useEffect(() => {
    let cancelled = false
    const render = async () => {
      try {
        const loadMermaid = new Function("return import('mermaid')")
        const mod = await loadMermaid()
        const mermaid = mod.default ?? mod
        mermaid.initialize({ startOnLoad: false, securityLevel: "loose" })
        const result = await mermaid.render(`mindmap-${Math.random().toString(36).slice(2)}`, chart)
        if (!cancelled) setSvg(result.svg)
      } catch {
        if (!cancelled) setSvg("")
      }
    }
    render()
    return () => {
      cancelled = true
    }
  }, [chart])

  if (svg) {
    return <div className="rounded bg-white p-3 border border-[#E8E5D8]" dangerouslySetInnerHTML={{ __html: svg }} />
  }

  return (
    <div className="space-y-2">
      <div className="rounded bg-white p-3 border border-[#E8E5D8]" dangerouslySetInnerHTML={{ __html: fallbackSvg }} />
      <pre className="whitespace-pre-wrap rounded bg-white p-3 text-xs text-muted-foreground border border-[#E8E5D8]">
        {chart}
      </pre>
    </div>
  )
}

function InteractiveMindmapTree({ node }: { node: MindmapNode }) {
  return (
    <details open className="rounded border border-[#E8E5D8] bg-white px-3 py-2">
      <summary className="cursor-pointer font-medium text-[#3D5A40]">{node.title}</summary>
      {node.note && <p className="ml-4 mt-1 text-xs text-muted-foreground">{node.note}</p>}
      {!!node.children?.length && (
        <div className="ml-4 mt-2 space-y-2 border-l border-[#C8D6C9] pl-3">
          {node.children.map((child, index) => (
            <InteractiveMindmapTree key={`${child.title}-${index}`} node={child} />
          ))}
        </div>
      )}
    </details>
  )
}

function DownloadButton({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      download
      className="inline-flex h-7 items-center gap-1 rounded border border-[#C8D6C9] bg-white px-2 text-xs text-[#3D5A40] hover:bg-[#EDF5EE]"
    >
      <Download className="h-3.5 w-3.5" />
      {label}
    </a>
  )
}

function SmallButton({ onClick, label, icon }: { onClick: () => void; label: string; icon: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex h-7 items-center gap-1 rounded border border-[#C8D6C9] bg-white px-2 text-xs text-[#3D5A40] hover:bg-[#EDF5EE]"
    >
      {icon}
      {label}
    </button>
  )
}

function ResourceGenerationStatusPanel({ status }: { status: ResourceGenerationStatus }) {
  const [isExpanded, setIsExpanded] = useState(status.state !== "done")
  const previousStateRef = useRef(status.state)
  const completedCount = status.steps.filter((step) => step.state === "done").length
  const currentStep = [...status.steps].reverse().find((step) => step.state === "running")
  const hasSteps = status.steps.length > 0
  const progressText = hasSteps
    ? `${completedCount}/${status.steps.length} 个阶段完成`
    : "等待多智能体调度"

  useEffect(() => {
    if (status.state === "done" && previousStateRef.current !== "done") {
      setIsExpanded(false)
    }
    if (status.state !== "done" && previousStateRef.current === "done") {
      setIsExpanded(true)
    }
    previousStateRef.current = status.state
  }, [status.state])

  return (
    <div className="rounded-lg border border-[#C8D6C9] bg-[#F8FAF6] overflow-hidden">
      <button
        type="button"
        onClick={() => setIsExpanded((open) => !open)}
        className="w-full flex items-start gap-3 px-3 py-2.5 text-left hover:bg-[#EDF5EE] transition-colors"
      >
        <StatusIcon state={status.state} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-3">
            <span className="font-semibold text-[#3D5A40]">个性化资源生成状态</span>
            <span className="text-[11px] text-muted-foreground whitespace-nowrap">{progressText}</span>
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground leading-relaxed">
            {currentStep ? currentStep.detail : status.summary}
          </p>
        </div>
        {isExpanded ? (
          <ChevronDown className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
        )}
      </button>

      {isExpanded && (
        <div className="border-t border-[#C8D6C9] px-3 py-2.5 space-y-2">
          {hasSteps ? (
            <div className="space-y-2">
              {status.steps.map((step, index) => (
                <ResourceGenerationStepRow key={`${step.node}-${index}`} step={step} />
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">正在建立学习画像与资源生成任务队列。</p>
          )}

          {(status.tokenUsage.total > 0 || status.error) && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 pt-1 text-[11px] text-muted-foreground">
              {status.tokenUsage.total > 0 && (
                <span>
                  Tokens {status.tokenUsage.total}
                  <span className="opacity-70"> (in {status.tokenUsage.input} / out {status.tokenUsage.output})</span>
                </span>
              )}
              {status.error && <span className="text-[#D97B6C]">{status.error}</span>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ResourceGenerationStepRow({ step }: { step: ResourceGenerationStep }) {
  return (
    <div className="flex gap-2 text-xs">
      <StepIcon state={step.state} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className={cn(
            "font-medium",
            step.state === "running" && "text-[#5C3D2E]",
            step.state === "done" && "text-[#3D5A40]",
            step.state === "error" && "text-[#D97B6C]"
          )}>
            {step.title}
          </span>
          <span className="text-[11px] text-muted-foreground">
            {step.state === "running"
              ? step.startedAt
              : `${step.startedAt ?? ""}${step.endedAt ? ` -> ${step.endedAt}` : ""}`}
          </span>
          {step.durationMs != null && (
            <span className="rounded bg-white px-1.5 py-0.5 text-[10px] text-muted-foreground border border-[#E8E5D8]">
              {step.durationMs}ms
            </span>
          )}
        </div>
        <p className="mt-0.5 leading-relaxed text-muted-foreground">{step.error || step.detail}</p>
      </div>
    </div>
  )
}

function StatusIcon({ state }: { state: ResourceGenerationState }) {
  const className = "h-4 w-4 mt-0.5 shrink-0"
  if (state === "done") return <CheckCircle2 className={cn(className, "text-[#3D5A40]")} />
  if (state === "error") return <CircleAlert className={cn(className, "text-[#D97B6C]")} />
  if (state === "waiting_review") return <PauseCircle className={cn(className, "text-[#B8860B]")} />
  return <Loader2 className={cn(className, "text-[#3D5A40] animate-spin")} />
}

function StepIcon({ state }: { state: ResourceGenerationStepState }) {
  const className = "h-3.5 w-3.5 mt-0.5 shrink-0"
  if (state === "done") return <CheckCircle2 className={cn(className, "text-[#3D5A40]")} />
  if (state === "error") return <CircleAlert className={cn(className, "text-[#D97B6C]")} />
  return <CircleDotDashed className={cn(className, "text-[#E8A87C] animate-pulse")} />
}

function mindmapTreeToMermaid(tree: MindmapNode): string {
  const lines = ["mindmap"]
  const walk = (node: MindmapNode, depth: number) => {
    const indent = "  ".repeat(depth)
    const title = sanitizeMindmapText(node.title)
    lines.push(`${indent}${depth === 1 ? `root((${title}))` : title}`)
    node.children?.forEach((child) => walk(child, depth + 1))
  }
  walk(tree, 1)
  return lines.join("\n")
}

function mindmapTreeToMarkdown(tree: MindmapNode): string {
  const lines = [`# ${tree.title}`]
  if (tree.note) lines.push("", tree.note)
  const walk = (node: MindmapNode, depth: number) => {
    const indent = "  ".repeat(depth - 1)
    lines.push(`${indent}- ${node.title}${node.note ? `：${node.note}` : ""}`)
    node.children?.forEach((child) => walk(child, depth + 1))
  }
  tree.children?.forEach((child) => walk(child, 1))
  return lines.join("\n")
}

function mindmapTreeToSvg(tree: MindmapNode): string {
  const rows: Array<{ title: string; depth: number; note?: string }> = []
  const walk = (node: MindmapNode, depth: number) => {
    rows.push({ title: node.title, depth, note: node.note })
    node.children?.forEach((child) => walk(child, depth + 1))
  }
  walk(tree, 0)

  const width = 900
  const rowHeight = 46
  const height = Math.max(120, rows.length * rowHeight + 32)
  const content = rows.map((row, index) => {
    const y = 28 + index * rowHeight
    const x = 24 + row.depth * 42
    const color = row.depth === 0 ? "#3D5A40" : row.depth === 1 ? "#5A7A5E" : "#7A9E7E"
    const title = escapeXml(row.title)
    const note = row.note ? escapeXml(row.note.slice(0, 70)) : ""
    return `
      <line x1="${Math.max(24, x - 24)}" y1="${y + 9}" x2="${x}" y2="${y + 9}" stroke="#C8D6C9" stroke-width="1"/>
      <circle cx="${x}" cy="${y + 9}" r="6" fill="${color}"/>
      <text x="${x + 14}" y="${y + 13}" fill="#2D2D2D" font-size="15" font-family="Arial, sans-serif" font-weight="${row.depth === 0 ? "700" : "500"}">${title}</text>
      ${note ? `<text x="${x + 14}" y="${y + 30}" fill="#6B7280" font-size="11" font-family="Arial, sans-serif">${note}</text>` : ""}
    `
  }).join("")

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="100%" height="100%" rx="8" fill="#F8FAF6"/>
    ${content}
  </svg>`
}

function downloadText(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}

async function downloadTreePng(tree: MindmapNode, filename: string) {
  const svg = mindmapTreeToSvg(tree)
  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const img = new window.Image()

  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve()
    img.onerror = () => reject(new Error("PNG export failed"))
    img.src = url
  })

  const canvas = document.createElement("canvas")
  canvas.width = img.width || 900
  canvas.height = img.height || 320
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.fillStyle = "#F8FAF6"
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(img, 0, 0)
  URL.revokeObjectURL(url)

  const pngUrl = canvas.toDataURL("image/png")
  const link = document.createElement("a")
  link.href = pngUrl
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
}

function safeFileName(value: string): string {
  return value.trim().replace(/[\\/:*?"<>|]+/g, "-").replace(/\s+/g, "-").slice(0, 60) || "mindmap"
}

function sanitizeMindmapText(value: string): string {
  return value.replace(/[()\n\r]/g, " ").trim() || "未命名知识点"
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;")
}
