"use client"

import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  Clock3,
  Loader2,
  PauseCircle,
  RotateCw,
  SkipForward,
} from "lucide-react"

import type { ActivityEvent, ActivityStatus } from "@/lib/observability-contracts"
import { cn } from "@/lib/utils"

const STATUS_COPY: Record<ActivityStatus, string> = {
  queued: "排队中",
  running: "执行中",
  waiting: "等待输入",
  completed: "已完成",
  retrying: "重试中",
  interrupted: "已中断",
  failed: "失败",
  skipped: "已跳过",
}

export function ActivityStream({ activities }: { activities: readonly ActivityEvent[] }) {
  if (activities.length === 0) return null
  const ordered = [...activities].sort(
    (left, right) => left.sequence - right.sequence || left.activityId.localeCompare(right.activityId),
  )
  const active = ordered.some((event) => ["queued", "running", "retrying", "waiting"].includes(event.status))

  return (
    <details
      className="group/activity overflow-hidden rounded-lg border border-border bg-[var(--surface-subtle)]"
      open={active || undefined}
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2 text-xs font-semibold text-foreground outline-none transition hover:bg-muted/70 focus-visible:ring-2 focus-visible:ring-ring/30 [&::-webkit-details-marker]:hidden">
        <span className="flex min-w-0 items-center gap-2">
          <Clock3 className="h-3.5 w-3.5 shrink-0 text-primary" />
          <span>活动记录</span>
          <span className="font-normal text-muted-foreground">{ordered.length}</span>
        </span>
        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground transition-transform duration-200 group-open/activity:rotate-180" />
      </summary>
      <ol className="border-t border-border px-3 py-2.5">
        {ordered.map((event, index) => (
          <li key={event.activityId} className="relative flex gap-2.5 pb-3 last:pb-0">
            {index < ordered.length - 1 ? (
              <span className="absolute left-[7px] top-4 h-[calc(100%-8px)] w-px bg-border" aria-hidden="true" />
            ) : null}
            <ActivityStatusIcon status={event.status} />
            <div className="min-w-0 flex-1">
              <div className="flex items-start justify-between gap-2">
                <p className="min-w-0 break-words text-xs font-medium leading-5 text-foreground">
                  {event.title}
                </p>
                <span className="shrink-0 text-[11px] text-muted-foreground">
                  {STATUS_COPY[event.status]}
                </span>
              </div>
              {event.summary ? (
                <p className="mt-0.5 break-words text-[11px] leading-4 text-muted-foreground">
                  {event.summary}
                </p>
              ) : null}
              <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5 font-mono text-[10px] text-muted-foreground">
                {event.node ? <span>{event.node}</span> : null}
                {event.model ? <span>{event.model}</span> : null}
                {event.durationMs !== undefined ? <span>{event.durationMs} ms</span> : null}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </details>
  )
}

function ActivityStatusIcon({ status }: { status: ActivityStatus }) {
  const className = cn(
    "relative z-[1] mt-0.5 h-4 w-4 shrink-0",
    status === "completed" && "text-[var(--success)]",
    ["queued", "running", "retrying"].includes(status) && "text-[var(--warning)]",
    ["waiting", "interrupted"].includes(status) && "text-[var(--info)]",
    status === "failed" && "text-destructive",
    status === "skipped" && "text-muted-foreground",
  )
  if (status === "completed") return <CheckCircle2 className={className} />
  if (status === "running") return <Loader2 className={cn(className, "animate-spin motion-reduce:animate-none")} />
  if (status === "retrying") return <RotateCw className={cn(className, "animate-spin motion-reduce:animate-none")} />
  if (status === "waiting" || status === "interrupted") return <PauseCircle className={className} />
  if (status === "failed") return <AlertCircle className={className} />
  if (status === "skipped") return <SkipForward className={className} />
  return <CircleDashed className={className} />
}
