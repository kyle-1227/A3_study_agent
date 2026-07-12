"use client"

import { useEffect, useId, useRef, useState } from "react"
import * as Popover from "@radix-ui/react-popover"
import * as Tooltip from "@radix-ui/react-tooltip"
import { BookOpenText, CircleHelp, X } from "lucide-react"

import type {
  ContextSectionEstimate,
  ThreadContextWindowV2,
} from "@/lib/observability-contracts"
import { cn } from "@/lib/utils"

interface ThreadContextCapsuleProps {
  window: ThreadContextWindowV2 | null
  closeSignal: string
}

export function ThreadContextCapsule({
  window,
  closeSignal,
}: ThreadContextCapsuleProps) {
  const [open, setOpen] = useState(false)
  const contentId = useId()
  const triggerRef = useRef<HTMLButtonElement>(null)
  const estimate = window?.nextCallContextEstimate
  const ratio = estimate?.usedRatio ?? 0
  const percentLabel = estimate?.maxContextTokens
    ? `${Math.round(ratio * 100)}%`
    : "--"
  const tooltipLabel = estimate?.maxContextTokens
    ? `预计 ${Math.round(ratio * 100)}% 已用`
    : "下一次上下文估算暂不可用"

  useEffect(() => {
    setOpen(false)
  }, [closeSignal])

  return (
    <Tooltip.Provider delayDuration={300} skipDelayDuration={150}>
      <Popover.Root open={open} onOpenChange={setOpen}>
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <Popover.Trigger asChild>
              <button
                ref={triggerRef}
                type="button"
                aria-label="查看线程背景信息窗口"
                aria-expanded={open}
                aria-controls={contentId}
                className={cn(
                  "flex h-8 min-w-[4.25rem] items-center justify-center gap-1.5 rounded-full border px-2.5",
                  "border-border bg-card/80 text-xs font-medium tabular-nums text-muted-foreground",
                  "transition-colors hover:border-primary/35 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
                )}
              >
                <BookOpenText className="h-3.5 w-3.5" aria-hidden="true" />
                <span>{percentLabel}</span>
              </button>
            </Popover.Trigger>
          </Tooltip.Trigger>
          <Tooltip.Portal>
            <Tooltip.Content
              side="top"
              sideOffset={7}
              className="z-[70] rounded-md bg-foreground px-2 py-1 text-[11px] text-background shadow-sm"
            >
              {tooltipLabel}
              <Tooltip.Arrow className="fill-foreground" />
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>

        <Popover.Portal>
          <Popover.Content
            id={contentId}
            role="dialog"
            aria-label="线程背景信息详情"
            side="top"
            align="end"
            sideOffset={10}
            collisionPadding={12}
            onCloseAutoFocus={(event) => {
              event.preventDefault()
              triggerRef.current?.focus()
            }}
            className={cn(
              "z-[60] flex max-h-[min(76vh,42rem)] w-[min(25rem,calc(100vw-1.5rem))] flex-col overflow-hidden rounded-lg border border-border bg-popover text-popover-foreground shadow-xl",
              "data-[state=open]:animate-in data-[state=closed]:animate-out motion-reduce:animate-none",
              "max-sm:!fixed max-sm:!inset-x-2 max-sm:!bottom-2 max-sm:!top-auto max-sm:!w-auto max-sm:!translate-x-0 max-sm:!translate-y-0 max-sm:!transform-none",
            )}
          >
            <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
              <div className="flex min-w-0 items-center gap-2">
                <BookOpenText className="h-4 w-4 shrink-0 text-primary" />
                <h2 className="truncate text-sm font-semibold">背景信息窗口</h2>
              </div>
              <Popover.Close asChild>
                <button
                  type="button"
                  aria-label="收回背景信息窗口"
                  title="收回"
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
                >
                  <X className="h-4 w-4" />
                </button>
              </Popover.Close>
            </div>

            <div className="min-h-0 overflow-y-auto px-4 py-4">
              {window && estimate ? (
                <ContextWindowDetails window={window} />
              ) : (
                <div className="flex items-start gap-2 rounded-md border border-border bg-muted/35 px-3 py-3 text-sm text-muted-foreground">
                  <CircleHelp className="mt-0.5 h-4 w-4 shrink-0" />
                  <p>当前线程还没有可验证的上下文估算。</p>
                </div>
              )}
            </div>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
    </Tooltip.Provider>
  )
}

function ContextWindowDetails({ window }: { window: ThreadContextWindowV2 }) {
  const estimate = window.nextCallContextEstimate
  const lastCall = window.lastLlmCallUsage
  const inventory = window.backgroundInventory
  const percent = estimate.maxContextTokens
    ? `${Math.round(estimate.usedRatio * 100)}% 已用`
    : "占用比例待确定"

  return (
    <div className="space-y-5 text-xs">
      <section aria-labelledby="next-context-heading">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p id="next-context-heading" className="font-semibold text-foreground">
              下一次调用估算
            </p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-foreground">
              {percent}
            </p>
          </div>
          <div className="text-right text-[11px] leading-5 text-muted-foreground">
            <p>{basisLabel(estimate.basis)}</p>
            <p>置信度：{confidenceLabel(estimate.confidence)}</p>
          </div>
        </div>
        <div
          className="mt-3 h-2 overflow-hidden rounded-full bg-muted"
          role="progressbar"
          aria-label="预计上下文占用"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.min(100, Math.round(estimate.usedRatio * 100))}
        >
          <div
            className="h-full bg-primary transition-[width] duration-200 motion-reduce:transition-none"
            style={{ width: `${Math.min(100, estimate.usedRatio * 100)}%` }}
          />
        </div>
        <div className="mt-2 flex items-center justify-between gap-3 font-mono text-[11px] tabular-nums text-muted-foreground">
          <span>{formatTokens(estimate.estimatedInputTokens)} 输入</span>
          <span>{formatTokens(estimate.maxContextTokens)} 最大窗口</span>
        </div>
        <SectionAccounting sections={estimate.sections} />
        {estimate.unknownSections.length > 0 ? (
          <div className="mt-3">
            <p className="font-medium text-foreground">待确定区段</p>
            <p className="mt-1 break-words leading-5 text-muted-foreground">
              {estimate.unknownSections.join(" · ")}
            </p>
          </div>
        ) : null}
      </section>

      <section className="border-t border-border pt-4" aria-labelledby="last-call-heading">
        <p id="last-call-heading" className="font-semibold text-foreground">
          最近一次 LLM 调用
        </p>
        {lastCall.present ? (
          <dl className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 text-muted-foreground">
            <dt className="truncate">节点</dt>
            <dd className="max-w-48 truncate text-right text-foreground" title={lastCall.nodeName}>
              {lastCall.nodeName}
            </dd>
            <dt>输入</dt>
            <dd className="font-mono tabular-nums text-foreground">
              {formatTokens(lastCall.inputEstimatedTokens)}
            </dd>
            <dt>输出预留</dt>
            <dd className="font-mono tabular-nums text-foreground">
              {formatTokens(lastCall.outputReservedTokens)}
            </dd>
            <dt>占用</dt>
            <dd className="font-mono tabular-nums text-foreground">
              {Math.round(lastCall.usedRatio * 100)}%
            </dd>
          </dl>
        ) : (
          <p className="mt-2 text-muted-foreground">尚无已完成的 LLM 输入账单。</p>
        )}
      </section>

      <section className="border-t border-border pt-4" aria-labelledby="inventory-heading">
        <p id="inventory-heading" className="font-semibold text-foreground">
          线程背景库存
        </p>
        {inventory.workspaceActiveSubject ? (
          <p className="mt-1 truncate text-muted-foreground" title={inventory.workspaceActiveSubject}>
            当前主题：{inventory.workspaceActiveSubject}
          </p>
        ) : null}
        <dl className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 text-muted-foreground">
          <InventoryRow label="Conversation summary" value={inventory.conversationSummaryPresent ? 1 : 0} />
          <InventoryRow label="Selected memory / profile" value={inventory.selectedMemoryCount} />
          <InventoryRow label="Evidence summaries" value={inventory.evidenceSummaryCount} />
          <InventoryRow label="Artifact summaries" value={inventory.artifactSummaryCount} />
          <InventoryRow label="Workspace evidence" value={inventory.workspaceEvidenceSummaryCount} />
          <InventoryRow label="Workspace gaps" value={inventory.workspaceGapCount} />
          <InventoryRow label="Workspace artifacts" value={inventory.workspaceArtifactCount} />
          <InventoryRow label="Manifest history" value={inventory.manifestCount} />
        </dl>
      </section>
    </div>
  )
}

function SectionAccounting({ sections }: { sections: ContextSectionEstimate[] }) {
  if (sections.length === 0) return null
  return (
    <div className="mt-4">
      <p className="font-medium text-foreground">已知输入区段</p>
      <dl className="mt-2 space-y-1.5">
        {sections.map((section) => (
          <div key={`${section.section}:${section.source}`} className="flex items-center justify-between gap-4">
            <dt className="min-w-0 truncate text-muted-foreground" title={section.section}>
              {section.section}
            </dt>
            <dd className="shrink-0 font-mono tabular-nums text-foreground">
              {formatTokens(section.estimatedTokens)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function InventoryRow({ label, value }: { label: string; value: number }) {
  return (
    <>
      <dt className="truncate">{label}</dt>
      <dd className="font-mono tabular-nums text-foreground">{value}</dd>
    </>
  )
}

function basisLabel(value: "known_next_node" | "thread_baseline"): string {
  return value === "known_next_node" ? "已知下一节点" : "线程基线"
}

function confidenceLabel(value: "high" | "medium" | "low"): string {
  return { high: "高", medium: "中", low: "低" }[value]
}

function formatTokens(value: number): string {
  if (!value) return "0"
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}k`
  return String(value)
}
