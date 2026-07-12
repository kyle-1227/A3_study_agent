"use client"

import { useEffect, useId, useRef, useState } from "react"
import * as Popover from "@radix-ui/react-popover"
import * as Tooltip from "@radix-ui/react-tooltip"
import { BookOpenText, CircleHelp, X } from "lucide-react"

import {
  CONTEXT_INJECTION_TYPES,
  type ContextInjectionType,
  type ThreadContextWindowV3,
} from "@/lib/thread-context-window-v3"
import { cn } from "@/lib/utils"

interface ThreadContextCapsuleProps {
  window: ThreadContextWindowV3 | null
  closeSignal: string
}

const SOURCE_LABELS: Record<ContextInjectionType, string> = {
  profile: "学习画像",
  memory: "会话记忆",
  evidence: "证据",
  artifact: "产物",
  rules: "规则",
  curriculum: "课程",
  trajectory: "学习轨迹",
  pipeline: "流程上下文",
}

export function ThreadContextCapsule({ window, closeSignal }: ThreadContextCapsuleProps) {
  const [open, setOpen] = useState(false)
  const contentId = useId()
  const triggerRef = useRef<HTMLButtonElement>(null)
  const percent = window ? Math.round(window.retainedRatio * 100) : null

  useEffect(() => setOpen(false), [closeSignal])

  return (
    <Tooltip.Provider delayDuration={300} skipDelayDuration={150}>
      <Popover.Root open={open} onOpenChange={setOpen}>
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <Popover.Trigger asChild>
              <button
                ref={triggerRef}
                type="button"
                aria-label="查看上下文记忆"
                aria-expanded={open}
                aria-controls={contentId}
                className={cn(
                  "flex h-8 min-w-[6.5rem] items-center justify-center gap-1.5 rounded-full border px-2.5",
                  "border-border bg-card/80 text-xs font-medium tabular-nums text-muted-foreground",
                  "transition-colors hover:border-primary/35 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30",
                )}
              >
                <BookOpenText className="h-3.5 w-3.5" aria-hidden="true" />
                <span>上下文记忆 {percent === null ? "--" : `${percent}%`}</span>
              </button>
            </Popover.Trigger>
          </Tooltip.Trigger>
          <Tooltip.Portal>
            <Tooltip.Content
              side="top"
              sideOffset={7}
              className="z-[70] rounded-md bg-foreground px-2 py-1 text-[11px] text-background shadow-sm"
            >
              {window
                ? `当前保留 ${formatTokens(window.retainedMemoryTokens)} / ${formatTokens(window.contextWindowLimitTokens)}`
                : "当前会话尚无上下文记忆统计"}
              <Tooltip.Arrow className="fill-foreground" />
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>

        <Popover.Portal>
          <Popover.Content
            id={contentId}
            role="dialog"
            aria-label="上下文记忆详情"
            side="top"
            align="end"
            sideOffset={10}
            collisionPadding={12}
            onCloseAutoFocus={(event) => {
              event.preventDefault()
              triggerRef.current?.focus()
            }}
            className={cn(
              "z-[60] flex max-h-[min(76vh,42rem)] w-[min(28rem,calc(100vw-1.5rem))] flex-col overflow-hidden rounded-lg border border-border bg-popover text-popover-foreground shadow-xl",
              "data-[state=open]:animate-in data-[state=closed]:animate-out motion-reduce:animate-none",
              "max-sm:!fixed max-sm:!inset-x-2 max-sm:!bottom-2 max-sm:!top-auto max-sm:!w-auto max-sm:!translate-x-0 max-sm:!translate-y-0 max-sm:!transform-none",
            )}
          >
            <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
              <div className="flex min-w-0 items-center gap-2">
                <BookOpenText className="h-4 w-4 shrink-0 text-primary" />
                <h2 className="truncate text-sm font-semibold">上下文记忆</h2>
                {window?.updating ? (
                  <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] text-primary">
                    更新中
                  </span>
                ) : null}
              </div>
              <Popover.Close asChild>
                <button
                  type="button"
                  aria-label="关闭上下文记忆详情"
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/30"
                >
                  <X className="h-4 w-4" />
                </button>
              </Popover.Close>
            </div>

            <div className="min-h-0 overflow-y-auto px-4 py-4">
              {window ? (
                <ContextWindowDetails window={window} />
              ) : (
                <div className="flex items-start gap-2 rounded-md border border-border bg-muted/35 px-3 py-3 text-sm text-muted-foreground">
                  <CircleHelp className="mt-0.5 h-4 w-4 shrink-0" />
                  <p>发送请求后，这里会展示实际注入并保留的会话记忆。</p>
                </div>
              )}
            </div>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
    </Tooltip.Provider>
  )
}

function ContextWindowDetails({ window }: { window: ThreadContextWindowV3 }) {
  const percent = Math.round(window.retainedRatio * 100)
  return (
    <div className="space-y-5 text-xs">
      <section aria-labelledby="retained-context-heading">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p id="retained-context-heading" className="font-semibold text-foreground">
              当前保留记忆
            </p>
            <p className="mt-1 text-2xl font-semibold tabular-nums text-foreground">{percent}%</p>
          </div>
          <p className="text-right text-[11px] leading-5 text-muted-foreground">
            {window.windowModel}
            <br />
            {formatTokens(window.retainedMemoryTokens)} / {formatTokens(window.contextWindowLimitTokens)}
          </p>
        </div>
        <div
          className="mt-3 h-2 overflow-hidden rounded-full bg-muted"
          role="progressbar"
          aria-label="当前保留上下文记忆占比"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.min(100, percent)}
        >
          <div
            className="h-full bg-primary transition-[width] duration-200 motion-reduce:transition-none"
            style={{ width: `${Math.min(100, window.retainedRatio * 100)}%` }}
          />
        </div>
      </section>

      <section className="border-t border-border pt-4" aria-labelledby="lifetime-heading">
        <p id="lifetime-heading" className="font-semibold text-foreground">会话累计</p>
        <dl className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 text-muted-foreground">
          <StatRow label="历史累计注入" value={formatTokens(window.lifetimeInjectedTokens)} />
          <StatRow label="历史去重内容" value={formatTokens(window.lifetimeUniqueTokens)} />
          <StatRow label="重复注入次数" value={String(window.repeatInjectionCount)} />
          <StatRow label="有注入的请求数" value={String(window.requestCount)} />
          <StatRow label="注入记录数" value={String(window.injectionCount)} />
          <StatRow label="当前活跃项" value={String(window.memorySummary.activeItemCount)} />
        </dl>
      </section>

      <section className="border-t border-border pt-4" aria-labelledby="types-heading">
        <p id="types-heading" className="font-semibold text-foreground">注入类型</p>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full min-w-[24rem] text-left text-[11px]">
            <thead className="text-muted-foreground">
              <tr>
                <th className="pb-2 font-medium">类型</th>
                <th className="pb-2 text-right font-medium">保留</th>
                <th className="pb-2 text-right font-medium">累计</th>
                <th className="pb-2 text-right font-medium">去重</th>
                <th className="pb-2 text-right font-medium">次数</th>
                <th className="pb-2 text-right font-medium">活跃项</th>
              </tr>
            </thead>
            <tbody>
              {CONTEXT_INJECTION_TYPES.map((source) => {
                const stats = window.injectionTypes[source]
                return (
                  <tr key={source} className="border-t border-border/70">
                    <td className="py-2 text-foreground">{SOURCE_LABELS[source]}</td>
                    <td className="py-2 text-right font-mono">{formatTokens(stats.retainedTokens)}</td>
                    <td className="py-2 text-right font-mono">{formatTokens(stats.lifetimeInjectedTokens)}</td>
                    <td className="py-2 text-right font-mono">{formatTokens(stats.lifetimeUniqueTokens)}</td>
                    <td className="py-2 text-right font-mono">{stats.injectionCount}</td>
                    <td className="py-2 text-right font-mono">{stats.activeItemCount}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="border-t border-border pt-4" aria-labelledby="compact-heading">
        <p id="compact-heading" className="font-semibold text-foreground">压缩状态</p>
        <p className="mt-2 leading-5 text-muted-foreground">
          {window.compaction.status === "compacted"
            ? `已压缩：${formatTokens(window.compaction.beforeTokens)} → ${formatTokens(window.compaction.afterTokens)}`
            : "本会话尚未执行完整压缩"}
        </p>
        <p className="mt-1 break-all text-[10px] text-muted-foreground">
          计量：{window.measurement.lastTokenizerMode || "尚无注入"}
          {window.measurement.lastEstimated ? "（估算）" : ""}
        </p>
      </section>
    </div>
  )
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd className="font-mono tabular-nums text-foreground">{value}</dd>
    </>
  )
}

function formatTokens(value: number): string {
  if (!value) return "0"
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}k`
  return String(value)
}
