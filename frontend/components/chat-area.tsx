"use client"

import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { useRouter } from "next/navigation"
import {
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  CircleDotDashed,
  Download,
  FileText,
  GitFork,
  GraduationCap,
  Image,
  Loader2,
  Map,
  PauseCircle,
  PlayCircle,
  Plus,
  Send,
  SlidersHorizontal,
  User,
} from "lucide-react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { ActivityStream } from "@/components/activity-stream"
import { ThreadContextCapsule } from "@/components/thread-context-capsule"
import { Button } from "@/components/ui/button"
import { useScrollActivity } from "@/hooks/use-scroll-activity"
import type {
  ActivityEvent,
  ContextUsageReport,
} from "@/lib/observability-contracts"
import type { QAFinalEventV1 } from "@/lib/qa-final"
import { splitStreamingMarkdown } from "@/lib/streaming-markdown"
import type { ThreadContextWindowV3 } from "@/lib/thread-context-window-v3"
import { cn } from "@/lib/utils"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  requestId?: string
  threadId?: string
  activities?: ActivityEvent[]
  resourceStatus?: ResourceGenerationStatus
  mindmap?: MindmapResult
  reviewDoc?: ReviewDocResult
  reviewDocs?: ReviewDocResult[]
  exercise?: ExerciseResult
  codePractice?: CodePracticeResult
  videoScript?: VideoScriptResult
  videoAnimation?: VideoAnimationResult
  studyPlan?: StudyPlanResult
  resourceFinalPayload?: Record<string, unknown>
  resourceFinalDedupeKey?: string
  qaFinal?: QAFinalEventV1
  qaFinalDedupeKey?: string
}

export type ResourceGenerationState =
  | "running"
  | "done"
  | "success"
  | "partial_success"
  | "controlled_stop"
  | "completed_with_resource"
  | "completed_without_resource"
  | "error"
  | "failed"
  | "waiting_review"
  | "waiting_for_profile_completion"
  | "interrupted"
  | "stopping"
  | "stopped"
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
  contextUsage?: ContextUsageReport | null
  error?: string
  waitingForReview?: boolean
  hasReceivedResourceFinal?: boolean
  completionKind?: "with_resource" | "partial_resource" | "without_resource" | "controlled_stop"
  lastResourceType?: string
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

export interface ReviewDocResult {
  title: string
  markdownUrl: string
  docxUrl?: string
  filename?: string
  docxFilename?: string
  subject?: string
  markdown?: string
}

export interface ExerciseResult {
  title: string
  markdownUrl?: string
  docxUrl?: string
  filename?: string
  docxFilename?: string
}

export interface CodePracticeResult {
  title: string
  markdownUrl?: string
  docxUrl?: string
  pythonUrl?: string
  markdown?: string
  filename?: string
  docxFilename?: string
  pythonFilename?: string
}

export interface VideoScriptResult {
  title: string
  markdownUrl?: string
  docxUrl?: string
  srtUrl?: string
  markdown?: string
  srt?: string
  filename?: string
  docxFilename?: string
  srtFilename?: string
}

export interface VideoAnimationResult {
  title: string
  htmlUrl?: string
  mp4Url?: string
  srtUrl?: string
  jsonUrl?: string
  durationSeconds?: number
  fullDurationSeconds?: number
  renderDurationSeconds?: number
  renderMode?: string
  renderSuccess?: boolean
  mp4Available?: boolean
  isPreviewVideo?: boolean
  videoValidForTeaching?: boolean
  renderLog?: string
}

export interface StudyPlanResult {
  title: string
  markdownUrl?: string
  docxUrl?: string
  filename?: string
  docxFilename?: string
  markdown?: string
}

interface ChatAreaProps {
  messages: Message[]
  liveTurnContent?: string
  onSendMessage: (content: string) => void
  onStopGeneration?: () => void
  onContinueThread?: () => void
  isLoading?: boolean
  canContinue?: boolean
  stopPending?: boolean
  threadContextWindow?: ThreadContextWindowV3 | null
  contextWindowCloseSignal?: string
}

export function ChatArea({
  messages,
  liveTurnContent = "",
  onSendMessage,
  onStopGeneration,
  onContinueThread,
  isLoading,
  canContinue,
  stopPending,
  threadContextWindow,
  contextWindowCloseSignal = "",
}: ChatAreaProps) {
  const [input, setInput] = useState("")
  const [isToolsOpen, setIsToolsOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const router = useRouter()
  const isScrolling = useScrollActivity(scrollContainerRef)
  const renderedLiveTurnContent = useAnimationFrameValue(liveTurnContent)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, renderedLiveTurnContent])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (trimmed && !isLoading) {
      onSendMessage(trimmed)
      setInput("")
      setIsToolsOpen(false)
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

  return (
    <main className="flex h-full min-h-0 min-w-0 flex-1 flex-col bg-background">
      <div
        ref={scrollContainerRef}
        className={cn(
          "min-h-0 flex-1 overflow-y-auto px-4 md:px-8",
          "chat-scroll-area",
          isScrolling && "chat-scroll-area--scrolling",
        )}
      >
        <div className="mx-auto flex w-full min-w-0 max-w-3xl flex-col gap-6 py-6">
          {messages.length === 0 ? (
            <div className="flex min-h-[56dvh] flex-col items-center justify-center text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
                <BrainCircuit className="h-9 w-9 text-[#f4d6b8]" strokeWidth={1.8} />
              </div>
              <h2 className="mb-2 text-xl font-semibold text-primary">高校学习 AI 助手</h2>
              <p className="max-w-xl text-sm leading-relaxed text-muted-foreground">
                我可以围绕高校课程学习生成课程讲解、思维导图、分层练习、复习文档和个性化学习计划。告诉我你的课程目标、知识短板或想生成的资源类型就可以开始。
              </p>
              <div className="mt-5 flex flex-wrap justify-center gap-2 text-xs">
                {["帮我制定机器学习入门计划", "生成线性代数知识导图", "给我 Python 分层练习"].map((sample) => (
                  <button
                    key={sample}
                    type="button"
                    onClick={() => setInput(sample)}
                    className="rounded-full border border-border bg-card px-3 py-1.5 text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
                  >
                    {sample}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message, index) => {
              const isLiveProjection = Boolean(
                renderedLiveTurnContent &&
                  index === messages.length - 1 &&
                  message.role === "assistant" &&
                  !message.content,
              )
              return (
                <MessageBubble
                  key={message.id}
                  message={
                    isLiveProjection
                      ? { ...message, content: renderedLiveTurnContent }
                      : message
                  }
                  streaming={isLiveProjection}
                />
              )
            })
          )}

          {isLoading && messages[messages.length - 1]?.role === "user" && (
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Bot className="h-4 w-4" />
              </div>
              <div className="rounded-2xl rounded-tl-sm border border-[var(--primary-line)] bg-card px-4 py-3">
                <div className="flex items-center gap-1.5" aria-label="正在生成">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-primary/70" style={{ animationDelay: "0ms" }} />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-primary/70" style={{ animationDelay: "150ms" }} />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-primary/70" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} className="h-4 shrink-0" />
        </div>
      </div>

      <div className="min-w-0 bg-background px-2 pb-3 pt-2 sm:px-4 sm:pb-4 md:px-8">
        <form onSubmit={handleSubmit} className="mx-auto min-w-0 max-w-3xl">
          <div className="overflow-hidden rounded-3xl bg-background shadow-[0_0_0_1px_rgba(0,0,0,0.03),0_2px_8px_rgba(0,0,0,0.06)]">
            <div className="px-4 pb-2 pt-4">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入你的问题..."
                rows={2}
                className="max-h-[200px] min-h-[60px] w-full resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
              />
            </div>

            <div className="flex min-w-0 items-center gap-0.5 px-2 pb-3 sm:gap-1 sm:px-3">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-9 w-9 rounded-full text-muted-foreground hover:bg-card/70 hover:text-primary"
                title="添加内容"
              >
                <Plus className="h-5 w-5" />
              </Button>
              <div className="relative">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsToolsOpen((open) => !open)}
                  className="h-9 w-9 rounded-full px-0 text-muted-foreground hover:bg-card/70 hover:text-primary sm:w-auto sm:px-3"
                  title="工具"
                >
                  <SlidersHorizontal className="h-4 w-4" />
                  <span className="hidden text-sm sm:inline">工具</span>
                </Button>
                {isToolsOpen && (
                  <div className="a3-popover-shadow absolute bottom-11 left-0 z-20 w-56 rounded-lg border border-border bg-popover p-1">
                    <button
                      type="button"
                      onClick={handleMindmapTool}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm text-foreground hover:bg-accent hover:text-accent-foreground"
                    >
                      <Map className="h-4 w-4" />
                      <span>生成思维导图</span>
                    </button>
                  </div>
                )}
              </div>

              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => router.push("/volunteer")}
                className="h-9 w-9 rounded-full px-0 text-muted-foreground hover:bg-card/70 hover:text-primary md:w-auto md:px-3"
                title="志愿填报"
              >
                <GraduationCap className="h-4 w-4" />
                <span className="hidden text-sm md:inline">志愿填报</span>
              </Button>

              <div className="flex-1" />

              <ThreadContextCapsule
                window={threadContextWindow ?? null}
                closeSignal={contextWindowCloseSignal}
              />
              <Button
                type={isLoading || canContinue ? "button" : "submit"}
                size="icon"
                onClick={isLoading ? onStopGeneration : canContinue ? onContinueThread : undefined}
                disabled={isLoading ? stopPending : canContinue ? false : !input.trim()}
                className="a3-button-primary h-9 w-9 rounded-full disabled:cursor-not-allowed disabled:opacity-50"
                title={isLoading ? "Stop at safe checkpoint" : canContinue ? "Continue from checkpoint" : "Send"}
              >
                {isLoading ? (
                  <PauseCircle className="h-4 w-4" />
                ) : canContinue ? (
                  <PlayCircle className="h-4 w-4" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </form>
      </div>
    </main>
  )
}

const markdownComponents: Components = {
  h1: ({ children }) => <h1 className="mb-2 mt-4 text-lg font-bold text-primary first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mb-1.5 mt-3 text-base font-bold text-primary first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mb-1 mt-2 text-sm font-semibold text-primary first:mt-0">{children}</h3>,
  p: ({ children }) => <p className="mb-2 break-words leading-relaxed last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="mb-2 list-disc space-y-1 pl-5 break-words">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 list-decimal space-y-1 pl-5 break-words">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed break-words">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-primary">{children}</strong>,
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-4 border-[var(--primary-line)] pl-3 text-muted-foreground">
      {children}
    </blockquote>
  ),
  code: ({ className, children }) => {
    const isBlock = className?.startsWith("language-")
    if (isBlock) {
      return (
        <code className="my-2 block overflow-x-auto whitespace-pre rounded-lg bg-[var(--surface-muted)] p-3 font-mono text-xs">
          {children}
        </code>
      )
    }
    return (
      <code className="whitespace-pre-wrap break-words rounded bg-[var(--surface-muted)] px-1.5 py-0.5 font-mono text-xs text-[var(--primary-deep)] [overflow-wrap:anywhere]">
        {children}
      </code>
    )
  },
  pre: ({ children }) => <pre className="my-2 max-w-full overflow-x-auto">{children}</pre>,
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="min-w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border bg-[var(--surface-muted)] px-2 py-1 text-left font-semibold">{children}</th>
  ),
  td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
  hr: () => <hr className="my-3 border-border" />,
}

const StreamingMarkdownFragment = memo(function StreamingMarkdownFragment({
  value,
}: {
  value: string
}) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
      {value}
    </ReactMarkdown>
  )
})

function StreamingMarkdown({ value }: { value: string }) {
  const parts = useMemo(() => splitStreamingMarkdown(value), [value])
  return (
    <>
      {parts.stablePrefix ? <StreamingMarkdownFragment value={parts.stablePrefix} /> : null}
      {parts.unstableSuffix ? <StreamingMarkdownFragment value={parts.unstableSuffix} /> : null}
    </>
  )
}

function useAnimationFrameValue(value: string): string {
  const [rendered, setRendered] = useState(value)
  const frameRef = useRef<number | null>(null)
  useEffect(() => {
    if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current)
    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = null
      setRendered(value)
    })
    return () => {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current)
        frameRef.current = null
      }
    }
  }, [value])
  return rendered
}

function MessageBubble({
  message,
  streaming = false,
}: {
  message: Message
  streaming?: boolean
}) {
  const isUser = message.role === "user"
  const hasAssistantPayload = Boolean(
    message.content ||
      message.resourceStatus ||
      message.activities?.length ||
      message.mindmap ||
      message.reviewDoc ||
      message.reviewDocs?.length ||
      message.exercise ||
      message.codePractice ||
      message.videoScript ||
      message.videoAnimation ||
      message.studyPlan,
  )

  return (
    <div className={cn("flex items-start gap-3", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-primary text-primary-foreground" : "bg-primary/10 text-primary"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div
        className={cn(
          "min-w-0 max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "rounded-tr-sm bg-primary text-primary-foreground"
            : "rounded-tl-sm border border-[var(--primary-line)] bg-card text-card-foreground"
        )}
      >
        {isUser ? (
          <div className="whitespace-pre-wrap">{message.content}</div>
        ) : (
          <div className="min-w-0 space-y-3">
            {message.resourceStatus && <ResourceGenerationStatusPanel status={message.resourceStatus} />}
            {message.activities?.length ? <ActivityStream activities={message.activities} /> : null}
            {message.reviewDocs?.length
              ? message.reviewDocs.map((doc) => (
                  <ReviewDocCard
                    key={`${doc.subject || doc.title}-${doc.markdownUrl || doc.filename || doc.docxUrl}`}
                    reviewDoc={doc}
                    markdownText={doc.markdown || ""}
                  />
                ))
              : message.reviewDoc && (
                  <ReviewDocCard
                    reviewDoc={message.reviewDoc}
                    markdownText={message.reviewDoc.markdown || ""}
                  />
                )}
            {message.mindmap && <MindmapCard mindmap={message.mindmap} />}
            {message.exercise && <ExerciseDownloadCard exercise={message.exercise} markdownText={message.content} />}
            {message.codePractice && (
              <CodePracticeCard
                codePractice={message.codePractice}
                markdownText={message.codePractice.markdown || message.content}
              />
            )}
            {message.videoScript && (
              <VideoScriptCard
                videoScript={message.videoScript}
                markdownText={message.videoScript.markdown || message.content}
              />
            )}
            {message.videoAnimation && <VideoAnimationCard videoAnimation={message.videoAnimation} />}
            {message.studyPlan && (
              <StudyPlanDownloadCard
                studyPlan={message.studyPlan}
                markdownText={message.studyPlan.markdown || message.content}
              />
            )}
            {message.content ? (
              <div className="min-w-0 max-w-full break-words">
                {streaming ? (
                  <StreamingMarkdown value={message.content} />
                ) : (
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                    {message.content}
                  </ReactMarkdown>
                )}
              </div>
            ) : !hasAssistantPayload ? (
              <p className="text-muted-foreground">正在生成个性化学习资源...</p>
            ) : null}
          </div>
        )}
      </div>
    </div>
  )
}

function ReviewDocCard({ reviewDoc, markdownText }: { reviewDoc: ReviewDocResult; markdownText: string }) {
  const printableMarkdown = reviewDoc.markdown || markdownText

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-[var(--surface-subtle)]">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <FileText className="h-4 w-4 shrink-0 text-primary" />
          <div className="min-w-0">
            <p className="truncate font-semibold text-primary">Markdown 复习文档</p>
            <p className="truncate text-xs text-muted-foreground">{reviewDoc.title}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1">
          <DownloadButton href={reviewDoc.markdownUrl} label="下载 .md" />
          {reviewDoc.docxUrl && <DownloadButton href={reviewDoc.docxUrl} label="下载 .docx" />}
          <SmallButton
            onClick={() => openReviewDocPrintPage(reviewDoc.title, printableMarkdown)}
            label="导出 PDF"
            icon={<FileText className="h-3.5 w-3.5" />}
          />
        </div>
      </div>
    </div>
  )
}

function ExerciseDownloadCard({ exercise, markdownText }: { exercise: ExerciseResult; markdownText: string }) {
  return (
    <div className="overflow-hidden rounded-lg border border-[#C8D6C9] bg-[#F8FAF6]">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <GraduationCap className="h-4 w-4 shrink-0 text-[#3D5A40]" />
          <div className="min-w-0">
            <p className="truncate font-semibold text-[#3D5A40]">练习题资源</p>
            <p className="truncate text-xs text-muted-foreground">{exercise.title}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1">
          {exercise.markdownUrl && <DownloadButton href={exercise.markdownUrl} label="下载 .md" />}
          {exercise.docxUrl && <DownloadButton href={exercise.docxUrl} label="下载 .docx" />}
          <SmallButton
            onClick={() => openReviewDocPrintPage(exercise.title || "练习题", markdownText)}
            label="导出 PDF"
            icon={<FileText className="h-3.5 w-3.5" />}
          />
        </div>
      </div>
    </div>
  )
}

function CodePracticeCard({
  codePractice,
  markdownText,
}: {
  codePractice: CodePracticeResult
  markdownText: string
}) {
  const printableMarkdown = codePractice.markdown || markdownText

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-[var(--surface-subtle)]">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <FileText className="h-4 w-4 shrink-0 text-primary" />
          <div className="min-w-0">
            <p className="truncate font-semibold text-primary">代码实操案例</p>
            <p className="truncate text-xs text-muted-foreground">{codePractice.title}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1">
          {codePractice.markdownUrl && <DownloadButton href={codePractice.markdownUrl} label="下载 .md" />}
          {codePractice.docxUrl && <DownloadButton href={codePractice.docxUrl} label="下载 .docx" />}
          {codePractice.pythonUrl && <DownloadButton href={codePractice.pythonUrl} label="下载 .py" />}
          <SmallButton
            onClick={() => openReviewDocPrintPage(codePractice.title || "代码实操案例", printableMarkdown)}
            label="导出 PDF"
            icon={<FileText className="h-3.5 w-3.5" />}
          />
        </div>
      </div>
    </div>
  )
}

function StudyPlanDownloadCard({ studyPlan, markdownText }: { studyPlan: StudyPlanResult; markdownText: string }) {
  const printableMarkdown = studyPlan.markdown || markdownText

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-[var(--surface-subtle)]">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <GraduationCap className="h-4 w-4 shrink-0 text-primary" />
          <div className="min-w-0">
            <p className="truncate font-semibold text-primary">学习计划文档</p>
            <p className="truncate text-xs text-muted-foreground">{studyPlan.title}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1">
          {studyPlan.markdownUrl && <DownloadButton href={studyPlan.markdownUrl} label="下载 .md" />}
          {studyPlan.docxUrl && <DownloadButton href={studyPlan.docxUrl} label="下载 .docx" />}
          <SmallButton
            onClick={() => openReviewDocPrintPage(studyPlan.title || "学习计划", printableMarkdown)}
            label="导出 PDF"
            icon={<FileText className="h-3.5 w-3.5" />}
          />
        </div>
      </div>
    </div>
  )
}

function VideoScriptCard({
  videoScript,
  markdownText,
}: {
  videoScript: VideoScriptResult
  markdownText: string
}) {
  const printableMarkdown = videoScript.markdown || markdownText

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-[var(--surface-subtle)]">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <FileText className="h-4 w-4 shrink-0 text-primary" />
          <div className="min-w-0">
            <p className="truncate font-semibold text-primary">教学视频 / 动画脚本</p>
            <p className="truncate text-xs text-muted-foreground">{videoScript.title}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1">
          {videoScript.markdownUrl && <DownloadButton href={videoScript.markdownUrl} label="下载 .md" />}
          {videoScript.docxUrl && <DownloadButton href={videoScript.docxUrl} label="下载 .docx" />}
          {videoScript.srtUrl && <DownloadButton href={videoScript.srtUrl} label="下载字幕 .srt" />}
          <SmallButton
            onClick={() => openReviewDocPrintPage(videoScript.title || "教学视频 / 动画脚本", printableMarkdown)}
            label="导出 PDF"
            icon={<FileText className="h-3.5 w-3.5" />}
          />
        </div>
      </div>
    </div>
  )
}

function VideoAnimationCard({ videoAnimation }: { videoAnimation: VideoAnimationResult }) {
  const durationText =
    typeof videoAnimation.durationSeconds === "number" && videoAnimation.durationSeconds > 0
      ? `${Math.round(videoAnimation.durationSeconds)} 秒`
      : "HTML / MP4 / SRT / JSON"
  const hasMp4 =
    videoAnimation.renderSuccess === true && videoAnimation.mp4Available === true && Boolean(videoAnimation.mp4Url)
  const showRenderWarning = videoAnimation.renderSuccess === false || videoAnimation.mp4Available === false
  const showPreviewNotice = videoAnimation.isPreviewVideo === true
  const showTeachingNotice = videoAnimation.videoValidForTeaching === true

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-[var(--surface-subtle)]">
      <div className="space-y-2 px-3 py-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <Image className="h-4 w-4 shrink-0 text-primary" />
            <div className="min-w-0">
              <p className="truncate font-semibold text-primary">教学动画 / MP4 视频</p>
              <p className="truncate text-xs text-muted-foreground">
                {videoAnimation.title} · {durationText}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-1">
            {videoAnimation.htmlUrl && <PreviewButton href={videoAnimation.htmlUrl} label="预览动画 .html" />}
            {hasMp4 && videoAnimation.mp4Url && <DownloadButton href={videoAnimation.mp4Url} label="下载 .mp4" />}
            {videoAnimation.srtUrl && <DownloadButton href={videoAnimation.srtUrl} label="下载字幕 .srt" />}
            {videoAnimation.jsonUrl && <DownloadButton href={videoAnimation.jsonUrl} label="下载动画结构 .json" />}
          </div>
        </div>
        {showRenderWarning && (
          <div className="rounded-md border border-amber-200 bg-amber-50 px-2.5 py-2 text-xs text-amber-800">
            <p>MP4 渲染失败，但 HTML 动画预览、JSON 动画结构和 SRT 字幕已生成。请检查 ffmpeg / Playwright 环境。</p>
            {videoAnimation.renderLog && <p className="mt-1 truncate text-amber-700">{videoAnimation.renderLog}</p>}
          </div>
        )}
        {showPreviewNotice && (
          <div className="rounded-md border border-sky-200 bg-sky-50 px-2.5 py-2 text-xs text-sky-800">
            这是 5 秒测试渲染视频，仅用于验证 MP4 渲染链路，不是最终教学视频。
          </div>
        )}
        {showTeachingNotice && (
          <div className="rounded-md border border-emerald-200 bg-emerald-50 px-2.5 py-2 text-xs text-emerald-800">
            正式教学动画视频。
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
    <div className="overflow-hidden rounded-lg border border-border bg-[var(--surface-subtle)]">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <GitFork className="h-4 w-4 shrink-0 text-primary" />
          <div className="min-w-0">
            <p className="truncate font-semibold text-primary">知识点思维导图</p>
            <p className="truncate text-xs text-muted-foreground">{mindmap.title}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1">
          <DownloadButton href={mindmap.xmindUrl} label=".xmind" />
          <SmallButton onClick={downloadMarkdown} label=".md" icon={<FileText className="h-3.5 w-3.5" />} />
          <SmallButton onClick={downloadSvg} label=".svg" icon={<Image className="h-3.5 w-3.5" />} />
          <SmallButton onClick={downloadPng} label=".png" icon={<Image className="h-3.5 w-3.5" />} />
        </div>
      </div>

      <div className="flex gap-1 border-b border-border px-3 py-2">
        {(["mermaid", "markdown", "tree"] as const).map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => setTab(item)}
            className={cn(
              "rounded px-2 py-1 text-xs transition-colors",
              tab === item ? "bg-primary text-primary-foreground" : "text-primary hover:bg-accent"
            )}
          >
            {item === "mermaid" ? "Mermaid" : item === "markdown" ? "Markdown" : "交互树"}
          </button>
        ))}
      </div>

      <div className="max-h-80 overflow-auto p-3">
        {tab === "mermaid" && <MermaidPreview chart={mermaid} fallbackSvg={mindmapTreeToSvg(mindmap.tree)} />}
        {tab === "markdown" && (
          <pre className="whitespace-pre-wrap rounded border border-border bg-card p-3 text-xs leading-relaxed text-card-foreground">
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
    return <div className="rounded border border-border bg-card p-3" dangerouslySetInnerHTML={{ __html: svg }} />
  }

  return (
    <div className="space-y-2">
      <div className="rounded border border-border bg-card p-3" dangerouslySetInnerHTML={{ __html: fallbackSvg }} />
      <pre className="whitespace-pre-wrap rounded border border-border bg-card p-3 text-xs text-muted-foreground">
        {chart}
      </pre>
    </div>
  )
}

function InteractiveMindmapTree({ node }: { node: MindmapNode }) {
  return (
    <details open className="rounded border border-border bg-card px-3 py-2">
      <summary className="cursor-pointer font-medium text-primary">{node.title}</summary>
      {node.note && <p className="ml-4 mt-1 text-xs text-muted-foreground">{node.note}</p>}
      {!!node.children?.length && (
        <div className="ml-4 mt-2 space-y-2 border-l border-border pl-3">
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
      className="inline-flex h-7 items-center gap-1 rounded-md border border-border bg-card px-2 text-xs text-primary hover:bg-accent"
    >
      <Download className="h-3.5 w-3.5" />
      {label}
    </a>
  )
}

function PreviewButton({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="inline-flex h-7 items-center gap-1 rounded-md border border-border bg-card px-2 text-xs text-primary hover:bg-accent"
    >
      <FileText className="h-3.5 w-3.5" />
      {label}
    </a>
  )
}

function SmallButton({ onClick, label, icon }: { onClick: () => void; label: string; icon: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex h-7 items-center gap-1 rounded-md border border-border bg-card px-2 text-xs text-primary hover:bg-accent"
    >
      {icon}
      {label}
    </button>
  )
}

function ResourceGenerationStatusPanel({ status }: { status: ResourceGenerationStatus }) {
  const [isExpanded, setIsExpanded] = useState(!isCompletedResourceState(status.state))
  const previousStateRef = useRef(status.state)
  const completedCount = status.steps.filter((step) => step.state === "done").length
  const currentStep = [...status.steps].reverse().find((step) => step.state === "running")
  const hasSteps = status.steps.length > 0
  const progressText = hasSteps ? `${completedCount}/${status.steps.length} 个阶段完成` : "等待多智能体调度"

  useEffect(() => {
    if (isCompletedResourceState(status.state) && !isCompletedResourceState(previousStateRef.current)) {
      setIsExpanded(false)
    }
    if (!isCompletedResourceState(status.state) && isCompletedResourceState(previousStateRef.current)) {
      setIsExpanded(true)
    }
    previousStateRef.current = status.state
  }, [status.state])

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-[var(--surface-subtle)]">
      <button
        type="button"
        onClick={() => setIsExpanded((open) => !open)}
        className="flex w-full items-start gap-3 px-3 py-2.5 text-left transition-colors hover:bg-accent"
      >
        <StatusIcon state={status.state} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-3">
            <span className="font-semibold text-primary">个性化资源生成状态</span>
            <span className="whitespace-nowrap text-[11px] text-muted-foreground">{progressText}</span>
          </div>
          <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
            {currentStep ? currentStep.detail : status.summary}
          </p>
        </div>
        {isExpanded ? (
          <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        )}
      </button>

      {isExpanded && (
        <div className="space-y-2 border-t border-border px-3 py-2.5">
          {hasSteps ? (
            <div className="space-y-2">
              {status.steps.map((step, index) => (
                <ResourceGenerationStepRow key={`${step.node}-${index}`} step={step} />
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">正在建立学习画像与资源生成任务队列。</p>
          )}

          {status.contextUsage && (
            <div className="rounded-md border border-border bg-card px-2 py-1.5 text-[11px] text-muted-foreground">
              <div className="flex items-center justify-between gap-2">
              <span className="font-medium text-primary">LLM call usage</span>
                <span>{Math.round(status.contextUsage.usedRatio * 100)}%</span>
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
                <span>used {status.contextUsage.usedTokens}</span>
                <span>max {status.contextUsage.maxContextTokens}</span>
                <span>available {status.contextUsage.availableTokens}</span>
                {status.contextUsage.estimated && <span>estimated</span>}
              </div>
            </div>
          )}

          {(status.tokenUsage.total > 0 || status.error) && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 pt-1 text-[11px] text-muted-foreground">
              {status.tokenUsage.total > 0 && (
                <span>
                  Tokens {status.tokenUsage.total}
                  <span className="opacity-70"> (in {status.tokenUsage.input} / out {status.tokenUsage.output})</span>
                </span>
              )}
              {status.error && <span className="text-[var(--danger)]">{status.error}</span>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function isCompletedResourceState(state: ResourceGenerationState): boolean {
  return (
    state === "done" ||
    state === "success" ||
    state === "partial_success" ||
    state === "controlled_stop" ||
    state === "completed_with_resource" ||
    state === "completed_without_resource" ||
    state === "failed"
  )
}

function ResourceGenerationStepRow({ step }: { step: ResourceGenerationStep }) {
  return (
    <div className="flex gap-2 text-xs">
      <StepIcon state={step.state} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span
            className={cn(
              "font-medium",
              step.state === "running" && "text-[var(--warning)]",
              step.state === "done" && "text-primary",
              step.state === "error" && "text-[var(--danger)]"
            )}
          >
            {step.title}
          </span>
          <span className="text-[11px] text-muted-foreground">
            {step.state === "running" ? step.startedAt : `${step.startedAt ?? ""}${step.endedAt ? ` -> ${step.endedAt}` : ""}`}
          </span>
          {step.durationMs != null && (
            <span className="rounded border border-border bg-card px-1.5 py-0.5 text-[10px] text-muted-foreground">
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
  if (state === "done" || state === "success" || state === "completed_with_resource") {
    return <CheckCircle2 className={cn(className, "text-[var(--success)]")} />
  }
  if (
    state === "partial_success" ||
    state === "controlled_stop" ||
    state === "completed_without_resource"
  ) {
    return <CircleAlert className={cn(className, "text-[var(--warning)]")} />
  }
  if (state === "error" || state === "failed") return <CircleAlert className={cn(className, "text-[var(--danger)]")} />
  if (state === "stopped") return <PauseCircle className={cn(className, "text-[var(--warning)]")} />
  if (state === "waiting_review" || state === "waiting_for_profile_completion" || state === "interrupted") {
    return <PauseCircle className={cn(className, "text-[var(--warning)]")} />
  }
  return <Loader2 className={cn(className, "animate-spin text-primary")} />
}

function StepIcon({ state }: { state: ResourceGenerationStepState }) {
  const className = "h-3.5 w-3.5 mt-0.5 shrink-0"
  if (state === "done") return <CheckCircle2 className={cn(className, "text-[var(--success)]")} />
  if (state === "error") return <CircleAlert className={cn(className, "text-[var(--danger)]")} />
  return <CircleDotDashed className={cn(className, "animate-pulse text-[var(--warning)]")} />
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
    const color = row.depth === 0 ? "#35593f" : row.depth === 1 ? "#4f7657" : "#6b956f"
    const title = escapeXml(row.title)
    const note = row.note ? escapeXml(row.note.slice(0, 70)) : ""
    return `
      <line x1="${Math.max(24, x - 24)}" y1="${y + 9}" x2="${x}" y2="${y + 9}" stroke="#b9c8b9" stroke-width="1"/>
      <circle cx="${x}" cy="${y + 9}" r="6" fill="${color}"/>
      <text x="${x + 14}" y="${y + 13}" fill="#243027" font-size="15" font-family="Arial, sans-serif" font-weight="${row.depth === 0 ? "700" : "500"}">${title}</text>
      ${note ? `<text x="${x + 14}" y="${y + 30}" fill="#667060" font-size="11" font-family="Arial, sans-serif">${note}</text>` : ""}
    `
  }).join("")

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="100%" height="100%" rx="8" fill="#faf9f2"/>
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

function openReviewDocPrintPage(title: string, markdownText: string) {
  try {
    window.sessionStorage.setItem(
      "review_doc_print_payload",
      JSON.stringify({
        title: title || "Markdown 复习文档",
        markdown: markdownText || `# ${title || "Markdown 复习文档"}`,
      }),
    )
  } catch {
    return
  }
  window.open("/print/review-doc", "_blank")
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
  ctx.fillStyle = "#faf9f2"
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
