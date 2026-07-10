"use client"

import { AlertCircle, BookOpenText, ChevronDown, Loader2 } from "lucide-react"

import type { ContextUsageState } from "@/lib/context-usage-state"
import type {
  BackgroundContextWindow,
  ContextUsageCategory,
} from "@/lib/observability-contracts"
import { cn } from "@/lib/utils"

const CATEGORY_COPY: Record<string, string> = {
  system_prompt: "系统与业务提示",
  tool_definitions: "工具定义",
  rules: "规则",
  skills: "Skills",
  subagent_definitions: "子代理能力",
  conversation: "对话与任务上下文",
  unclassified: "未分类",
  injected_context: "CE 注入块",
  output_reserved: "输出预留",
}

const CATEGORY_COLORS = [
  "bg-[#35593F]",
  "bg-[#C9824A]",
  "bg-[#3E6F99]",
  "bg-[#8A5B52]",
  "bg-[#4F7A70]",
  "bg-[#A45F55]",
  "bg-[#78806F]",
]

interface ContextUsagePanelProps {
  state: ContextUsageState
  background: BackgroundContextWindow | null
}

export function ContextUsagePanel({ state, background }: ContextUsagePanelProps) {
  const report = state.report
  const usedTokens = report?.usedTokens ?? background?.usedTokens ?? 0
  const maxTokens = report?.maxContextTokens ?? background?.maxContextTokens ?? 0
  const usedRatio = report?.usedRatio ?? background?.usedRatio ?? 0
  const present = Boolean(report || background)

  return (
    <section
      className="mx-3 mb-3 rounded-lg border border-white/10 bg-[#252725] px-3 py-3 text-[#F4F5F2] shadow-[0_8px_20px_rgba(20,24,21,0.14)]"
      aria-label="背景信息窗口"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <BookOpenText className="h-4 w-4 shrink-0 text-[#B9D5BE]" />
          <h2 className="truncate text-xs font-semibold text-[#F4F5F2]">背景信息窗口</h2>
        </div>
        {state.updating ? (
          <span className="flex items-center gap-1 text-[11px] text-[#E1A46E]">
            <Loader2 className="h-3 w-3 animate-spin motion-reduce:animate-none" />
            更新中
          </span>
        ) : null}
      </div>

      {!present ? (
        <p className="mt-2 text-xs leading-5 text-[#A9AEA8]">等待首个受保护的模型输入。</p>
      ) : (
        <div className="mt-2.5">
          <div className="flex items-baseline justify-between gap-3">
            <p className="text-xl font-semibold tabular-nums text-[#F4F5F2]">
              {formatPercent(usedRatio)}
            </p>
            <p className="font-mono text-[11px] tabular-nums text-[#A9AEA8]">
              {formatTokens(usedTokens)} / {formatTokens(maxTokens)}
            </p>
          </div>
          <UsageBar categories={report?.mainCategories ?? []} usedTokens={usedTokens} maxTokens={maxTokens} />
          <p className="mt-1.5 text-[11px] text-[#A9AEA8]">
            {report ? `最近调用 · ${report.nodeName}` : "已恢复的线程快照"}
          </p>
        </div>
      )}

      {state.error ? (
        <div className="mt-2 flex gap-2 rounded-md border border-[#C96A60]/40 bg-[#C96A60]/10 px-2.5 py-2 text-[11px] leading-4 text-[#F0A39A]">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{state.error.warning}</span>
        </div>
      ) : null}

      {present ? (
        <details className="group/report mt-2.5 border-t border-white/10 pt-2.5">
          <summary className="flex cursor-pointer list-none items-center justify-between text-xs font-medium text-[#CBE2CE] outline-none focus-visible:ring-2 focus-visible:ring-[#B9D5BE]/40 [&::-webkit-details-marker]:hidden">
            <span>查看报告</span>
            <ChevronDown className="h-3.5 w-3.5 transition-transform duration-200 group-open/report:rotate-180" />
          </summary>
          <div className="mt-3 space-y-3 text-[11px]">
            {report ? <CategoryReport title="输入主分类" categories={report.mainCategories} /> : null}
            {report ? <CategoryReport title="详细来源" categories={report.detailedCategories} /> : null}
            {report?.overlapRollups.length ? (
              <CategoryReport title="重叠视图（不重复计数）" categories={report.overlapRollups} />
            ) : null}
            <BackgroundInventory background={background} />
            {report?.reconciliationWarnings.length ? (
              <div>
                <p className="font-semibold text-[#F4F5F2]">对账提示</p>
                <p className="mt-1 break-words text-[#A9AEA8]">
                  {report.reconciliationWarnings.join(" · ")}
                </p>
              </div>
            ) : null}
          </div>
        </details>
      ) : null}
    </section>
  )
}

function UsageBar({
  categories,
  usedTokens,
  maxTokens,
}: {
  categories: readonly ContextUsageCategory[]
  usedTokens: number
  maxTokens: number
}) {
  const denominator = Math.max(maxTokens, 1)
  const reserved = Math.max(0, usedTokens - categories.reduce((sum, item) => sum + item.estimatedTokens, 0))
  const segments = [
    ...categories.map((item) => ({ key: item.category, tokens: item.estimatedTokens })),
    ...(reserved > 0 ? [{ key: "output_reserved", tokens: reserved }] : []),
  ]
  return (
    <div className="mt-2 flex h-2 w-full overflow-hidden rounded-full bg-white/10" aria-label="上下文窗口用量">
      {segments.map((segment, index) => (
        <span
          key={segment.key}
          className={cn("h-full min-w-px", CATEGORY_COLORS[index % CATEGORY_COLORS.length])}
          style={{ width: `${Math.min(100, (segment.tokens / denominator) * 100)}%` }}
          title={`${CATEGORY_COPY[segment.key] ?? segment.key}: ${segment.tokens}`}
        />
      ))}
    </div>
  )
}

function CategoryReport({ title, categories }: { title: string; categories: readonly ContextUsageCategory[] }) {
  return (
    <div>
      <p className="font-semibold text-[#F4F5F2]">{title}</p>
      <dl className="mt-1.5 space-y-1">
        {categories.map((item) => (
          <div key={item.category} className="flex items-center justify-between gap-3">
            <dt className="min-w-0 truncate text-[#A9AEA8]" title={item.category}>
              {CATEGORY_COPY[item.category] ?? item.category}
            </dt>
            <dd className="shrink-0 font-mono tabular-nums text-[#F4F5F2]">
              {formatTokens(item.estimatedTokens)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function BackgroundInventory({ background }: { background: BackgroundContextWindow | null }) {
  if (!background) return null
  return (
    <div>
      <p className="font-semibold text-[#F4F5F2]">线程背景库存</p>
      <dl className="mt-1.5 grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 text-[#A9AEA8]">
        <dt>Conversation summary</dt><dd className="text-[#F4F5F2]">{background.conversationSummaryPresent ? "1" : "0"}</dd>
        <dt>Memory</dt><dd className="text-[#F4F5F2]">{background.selectedMemoryCount}</dd>
        <dt>Evidence summaries</dt><dd className="text-[#F4F5F2]">{background.evidenceSummaryCount}</dd>
        <dt>Artifact summaries</dt><dd className="text-[#F4F5F2]">{background.artifactSummaryCount}</dd>
        <dt>Workspace evidence</dt><dd className="text-[#F4F5F2]">{background.workspaceEvidenceSummaryCount}</dd>
        <dt>Workspace gaps</dt><dd className="text-[#F4F5F2]">{background.workspaceGapCount}</dd>
        <dt>Workspace artifacts</dt><dd className="text-[#F4F5F2]">{background.workspaceArtifactCount}</dd>
        <dt>Influence ledger</dt><dd className="text-[#F4F5F2]">{background.influenceEntryCount}</dd>
      </dl>
    </div>
  )
}

function formatTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m`
  if (value >= 1_000) return `${Math.round(value / 1_000)}k`
  return String(value)
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}% 已用`
}
