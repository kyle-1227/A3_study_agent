"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  CircleDashed,
  GripHorizontal,
  Loader2,
  PauseCircle,
} from "lucide-react"

import { ManifestGraph } from "@/components/manifest-graph"
import { EvidenceProgressPanel } from "@/components/evidence-progress-panel"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { activitiesForRequest } from "@/lib/activity-reducer"
import type { GraphViewMode } from "@/lib/graph-layout"
import type { EvidenceProgressTimeline } from "@/lib/evidence-progress"
import type {
  ActivityEvent,
  GraphManifest,
  GraphManifestUnavailable,
} from "@/lib/observability-contracts"
import { cn } from "@/lib/utils"

export interface LogEntry {
  type: "info" | "error" | "warning" | "perf" | "usage" | "context"
  message: string
  ts: string
}

interface RightPanelProps {
  logs: LogEntry[]
  activities: ActivityEvent[]
  evidenceProgress: EvidenceProgressTimeline
  tokenUsage: { input: number; output: number; total: number }
  graphManifest: GraphManifest | null
  graphManifestError: GraphManifestUnavailable | null
  graphManifestLoading: boolean
  currentRequestId: string
  isInterrupted?: boolean
}

export function RightPanel({
  logs,
  activities,
  evidenceProgress,
  tokenUsage,
  graphManifest,
  graphManifestError,
  graphManifestLoading,
  currentRequestId,
  isInterrupted,
}: RightPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [viewTab, setViewTab] = useState<"trail" | "graph">("trail")
  const [isLogsCollapsed, setIsLogsCollapsed] = useState(false)
  const [splitPct, setSplitPct] = useState(68)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLElement>(null)
  const draggingRef = useRef(false)
  const startYRef = useRef(0)
  const startSplitRef = useRef(68)

  useEffect(() => {
    if (window.matchMedia("(max-width: 1023px)").matches) setIsCollapsed(true)
  }, [])

  const requestActivities = useMemo(
    () =>
      currentRequestId
        ? activitiesForRequest(activities, currentRequestId)
        : activities.slice(-40),
    [activities, currentRequestId],
  )
  const graphFitViewSignal = `${viewTab}:${isCollapsed}:${isLogsCollapsed}:${splitPct}`

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  const handleDragStart = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault()
      draggingRef.current = true
      startYRef.current = event.clientY
      startSplitRef.current = splitPct
      document.body.style.cursor = "row-resize"
      document.body.style.userSelect = "none"
    },
    [splitPct],
  )

  useEffect(() => {
    const handleDragMove = (event: MouseEvent) => {
      if (!draggingRef.current || !panelRef.current) return
      const deltaPct = ((event.clientY - startYRef.current) / panelRef.current.clientHeight) * 100
      setSplitPct(Math.min(88, Math.max(30, startSplitRef.current + deltaPct)))
    }
    const handleDragEnd = () => {
      if (!draggingRef.current) return
      draggingRef.current = false
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
    window.addEventListener("mousemove", handleDragMove)
    window.addEventListener("mouseup", handleDragEnd)
    return () => {
      window.removeEventListener("mousemove", handleDragMove)
      window.removeEventListener("mouseup", handleDragEnd)
    }
  }, [])

  return (
    <aside
      ref={panelRef}
      className={cn(
        "relative flex h-[100dvh] shrink-0 self-stretch flex-col overflow-hidden border-l border-sidebar-border bg-sidebar text-sidebar-foreground",
        "select-none transition-[width] duration-200 ease-out motion-reduce:transition-none",
        isCollapsed ? "w-12" : "w-80",
      )}
      aria-label="运行状态面板"
    >
      {isCollapsed ? (
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(false)}
          className="absolute left-1 top-4 h-8 w-8 text-muted-foreground hover:text-foreground"
          title="展开运行状态面板"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      ) : (
        <>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsCollapsed(true)}
            className="absolute left-2 top-4 z-10 h-8 w-8 text-muted-foreground hover:text-foreground"
            title="收起运行状态面板"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          <div
            className="flex min-h-0 flex-col"
            style={isLogsCollapsed ? { flex: "1 1 0%" } : { height: `${splitPct}%`, flexShrink: 0 }}
          >
            <div className="flex min-h-0 flex-1 flex-col pt-4">
              <div className="mb-3 flex items-center gap-2 px-4 pl-12">
                <PanelTab active={viewTab === "trail"} onClick={() => setViewTab("trail")}>
                  活动轨迹
                </PanelTab>
                <PanelTab active={viewTab === "graph"} onClick={() => setViewTab("graph")}>
                  图视图
                </PanelTab>
              </div>

              <ScrollArea className="min-h-0 flex-1 px-3">
                {viewTab === "trail" ? (
                  <div className="space-y-2">
                    {(!currentRequestId || evidenceProgress.requestId === currentRequestId) ? (
                      <EvidenceProgressPanel timeline={evidenceProgress} />
                    ) : null}
                    <ActivityTrail activities={requestActivities} />
                  </div>
                ) : (
                  <GraphSurface
                    manifest={graphManifest}
                    error={graphManifestError}
                    loading={graphManifestLoading}
                    activities={requestActivities}
                    fitViewSignal={graphFitViewSignal}
                  />
                )}
              </ScrollArea>

              {isInterrupted ? (
                <div className="border-t border-[var(--warning)] bg-[var(--warning-soft)] px-4 py-2 pl-12">
                  <p className="flex items-center gap-1.5 text-xs font-medium text-[var(--warning)]">
                    <PauseCircle className="h-3.5 w-3.5" />
                    等待用户输入
                  </p>
                </div>
              ) : null}

              {tokenUsage.total > 0 ? (
                <div className="border-t border-border bg-[var(--surface-muted)]/70 px-4 py-2 pl-12">
                  <p className="font-mono text-[11px] tabular-nums text-primary">
                    Tokens {tokenUsage.total}
                    <span className="ml-1 text-muted-foreground">
                      ({tokenUsage.input} in / {tokenUsage.output} out)
                    </span>
                  </p>
                </div>
              ) : null}

            </div>
          </div>

          {!isLogsCollapsed ? (
            <div
              onMouseDown={handleDragStart}
              className="group z-10 flex h-6 shrink-0 cursor-row-resize items-center justify-center border-y border-border bg-border/40 transition-colors hover:bg-primary/10"
              role="separator"
              aria-orientation="horizontal"
              aria-label="调整活动面板与日志高度"
            >
              <GripHorizontal className="h-3 w-6 text-muted-foreground/60 transition-colors group-hover:text-primary" />
            </div>
          ) : null}

          <SystemLogs
            logs={logs}
            collapsed={isLogsCollapsed}
            onCollapsedChange={setIsLogsCollapsed}
            endRef={logsEndRef}
          />
        </>
      )}
    </aside>
  )
}

function PanelTab({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-md px-2 py-1 text-xs transition-colors",
        active ? "bg-primary text-primary-foreground" : "text-primary hover:bg-primary/10",
      )}
    >
      {children}
    </button>
  )
}

function ActivityTrail({ activities }: { activities: readonly ActivityEvent[] }) {
  if (activities.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--primary-line)] px-4 py-7 text-center">
        <CircleDashed className="mx-auto h-5 w-5 text-muted-foreground" />
        <p className="mt-2 text-xs text-muted-foreground">等待活动事件</p>
      </div>
    )
  }
  return (
    <ol className="space-y-1 rounded-lg border border-border bg-[var(--surface-subtle)] p-2">
      {activities.map((activity) => (
        <li key={activity.activityId} className="flex gap-2 rounded-md px-2 py-2 hover:bg-muted/70">
          <ActivityIcon activity={activity} />
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-2">
              <p className="min-w-0 break-words text-xs font-medium leading-4 text-foreground">
                {activity.title}
              </p>
              <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
                {activity.durationMs !== undefined ? `${activity.durationMs}ms` : activity.status}
              </span>
            </div>
            {activity.summary ? (
              <p className="mt-0.5 line-clamp-2 text-[11px] leading-4 text-muted-foreground">
                {activity.summary}
              </p>
            ) : null}
          </div>
        </li>
      ))}
    </ol>
  )
}

function ActivityIcon({ activity }: { activity: ActivityEvent }) {
  if (activity.status === "completed") {
    return <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--success)]" />
  }
  if (["running", "retrying"].includes(activity.status)) {
    return <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin text-[var(--warning)] motion-reduce:animate-none" />
  }
  if (["waiting", "interrupted"].includes(activity.status)) {
    return <PauseCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--info)]" />
  }
  if (activity.status === "failed") {
    return <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive" />
  }
  return (
    <CircleDashed
      className={cn(
        "mt-0.5 h-3.5 w-3.5 shrink-0",
        activity.status === "queued" ? "text-[var(--warning)]" : "text-muted-foreground",
      )}
    />
  )
}

function GraphSurface({
  manifest,
  error,
  loading,
  activities,
  fitViewSignal,
}: {
  manifest: GraphManifest | null
  error: GraphManifestUnavailable | null
  loading: boolean
  activities: readonly ActivityEvent[]
  fitViewSignal: string
}) {
  const [viewMode, setViewMode] = useState<GraphViewMode>("full_graph")
  if (loading) {
    return (
      <div className="flex h-80 items-center justify-center rounded-lg border border-border bg-[var(--surface-subtle)] text-xs text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin motion-reduce:animate-none" />
        正在同步图清单
      </div>
    )
  }
  if (error) {
    return (
      <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-xs leading-5 text-destructive">
        <p className="font-semibold">图清单不可用</p>
        <p className="mt-1 break-words">{error.reason}</p>
      </div>
    )
  }
  if (!manifest) {
    return (
      <div className="flex h-80 items-center justify-center rounded-lg border border-dashed border-[var(--primary-line)] text-xs text-muted-foreground">
        尚未收到图清单
      </div>
    )
  }
  const visibleNodeCount = manifest.nodes.filter((node) => node.visible).length
  return (
    <section className="space-y-2" aria-label="图视图">
      <div className="flex items-center justify-between gap-2">
        <div
          className="inline-flex rounded-md border border-border bg-[var(--surface-muted)] p-0.5"
          role="group"
          aria-label="图视图范围"
        >
          <GraphViewModeButton
            active={viewMode === "current_path"}
            onClick={() => setViewMode("current_path")}
          >
            当前路径
          </GraphViewModeButton>
          <GraphViewModeButton
            active={viewMode === "full_graph"}
            onClick={() => setViewMode("full_graph")}
          >
            完整图
          </GraphViewModeButton>
        </div>
        <span
          data-testid="graph-visible-node-count"
          className="shrink-0 text-[11px] tabular-nums text-muted-foreground"
          aria-live="polite"
        >
          可见节点 {visibleNodeCount}
        </span>
      </div>
      <ManifestGraph
        manifest={manifest}
        error={null}
        loading={false}
        activities={activities}
        viewMode={viewMode}
        fitViewSignal={fitViewSignal}
      />
    </section>
  )
}

function GraphViewModeButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-sm px-2 py-1 text-[11px] transition-colors",
        active ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-primary",
      )}
      aria-pressed={active}
    >
      {children}
    </button>
  )
}

function SystemLogs({
  logs,
  collapsed,
  onCollapsedChange,
  endRef,
}: {
  logs: readonly LogEntry[]
  collapsed: boolean
  onCollapsedChange: (collapsed: boolean) => void
  endRef: React.RefObject<HTMLDivElement | null>
}) {
  if (collapsed) {
    return (
      <button
        type="button"
        onClick={() => onCollapsedChange(false)}
        className="flex shrink-0 items-center gap-1.5 border-t border-border px-4 py-2 text-xs text-muted-foreground transition-colors hover:bg-[var(--surface-muted)] hover:text-primary"
      >
        <ChevronUp className="h-3 w-3" />
        系统日志
        {logs.length > 0 ? (
          <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] text-primary">
            {logs.length}
          </span>
        ) : null}
      </button>
    )
  }
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2">
        <h3 className="text-sm font-semibold text-primary">系统日志</h3>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => onCollapsedChange(true)}
          className="h-6 w-6 text-muted-foreground hover:text-foreground"
          title="收起系统日志"
        >
          <ChevronDown className="h-3.5 w-3.5" />
        </Button>
      </div>
      <ScrollArea className="min-h-0 flex-1 px-4">
        <div className="flex flex-col gap-1 pb-4">
          {logs.map((log, index) => (
            <div
              key={`${log.ts}-${index}`}
              className={cn(
                "flex gap-2 rounded px-2 py-1 font-mono text-xs",
                log.type === "error" && "bg-[var(--danger-soft)] text-[var(--danger)]",
                log.type === "info" && "bg-[var(--surface-muted)] text-muted-foreground",
                log.type === "warning" && "bg-[var(--warning-soft)] text-[var(--warning)]",
                log.type === "perf" && "bg-[var(--info-soft)] text-[var(--info)]",
                log.type === "usage" && "bg-primary/10 text-primary",
                log.type === "context" && "bg-[var(--info-soft)] text-[var(--info)]",
              )}
            >
              <span className="shrink-0 opacity-55" suppressHydrationWarning>
                {log.ts}
              </span>
              <span className="min-w-0 break-words">{log.message}</span>
            </div>
          ))}
          <div ref={endRef} />
        </div>
      </ScrollArea>
    </div>
  )
}
