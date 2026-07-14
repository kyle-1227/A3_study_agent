"use client"

import { AlertCircle, CheckCircle2, CircleDashed, Loader2, Search } from "lucide-react"

import {
  evidenceProgressItems,
  evidenceProgressSummary,
  evidenceProgressTitle,
  type EvidenceProgressEventV1,
  type EvidenceProgressTimeline,
} from "@/lib/evidence-progress"
import { cn } from "@/lib/utils"

export function EvidenceProgressPanel({
  timeline,
}: {
  timeline: EvidenceProgressTimeline
}) {
  const items = evidenceProgressItems(timeline)
  if (items.length === 0) return null
  const active = items.some((item) => item.phaseStatus === "running") && !timeline.aborted

  return (
    <section
      className="rounded-lg border border-[var(--primary-line)] bg-primary/5 p-2.5"
      aria-label="证据补搜进度"
      aria-live="polite"
    >
      <div className="mb-2 flex items-center justify-between gap-2 px-1">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
          <Search className="h-3.5 w-3.5 text-primary" />
          证据补搜闭环
        </h3>
        <span className="font-mono text-[10px] text-muted-foreground">
          {timeline.aborted ? "已中止" : active ? "检索中" : timeline.terminal ? "已结束" : "处理中"}
        </span>
      </div>
      <ol className="space-y-1">
        {items.map((item) => (
          <li
            key={item.progressId}
            className="flex gap-2 rounded-md border border-border/70 bg-background/80 px-2 py-2"
          >
            <ProgressIcon event={item} />
            <div className="min-w-0 flex-1">
              <p className="break-words text-[11px] font-medium leading-4 text-foreground">
                {evidenceProgressTitle(item)}
              </p>
              <p className="mt-0.5 break-words text-[10px] leading-4 text-muted-foreground">
                {evidenceProgressSummary(item)}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}

function ProgressIcon({ event }: { event: EvidenceProgressEventV1 }) {
  const className = "mt-0.5 h-3.5 w-3.5 shrink-0"
  if (event.phaseStatus === "running") {
    return <Loader2 className={cn(className, "animate-spin text-[var(--warning)] motion-reduce:animate-none")} />
  }
  if (event.phaseStatus === "failed") {
    return <AlertCircle className={cn(className, "text-destructive")} />
  }
  if (event.details.stage === "evidence_orchestration.resource.assigned" && event.details.status === "blocked") {
    return <AlertCircle className={cn(className, "text-[var(--warning)]")} />
  }
  if (event.phaseStatus === "completed") {
    return <CheckCircle2 className={cn(className, "text-[var(--success)]")} />
  }
  return <CircleDashed className={cn(className, "text-muted-foreground")} />
}
