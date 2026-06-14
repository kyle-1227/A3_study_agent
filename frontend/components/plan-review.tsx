"use client"

import { useEffect, useState } from "react"
import { Download, FilePenLine, MessageSquareText } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface PlanReviewProps {
  draft: string
  onConfirm: (editedPlan: string) => void
  onFeedback: (feedback: string) => void
  isSubmitting?: boolean
}

function downloadPlan(content: string) {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `study-plan-${new Date().toISOString().slice(0, 10)}.md`
  a.click()
  URL.revokeObjectURL(url)
}

export function PlanReview({ draft, onConfirm, onFeedback, isSubmitting }: PlanReviewProps) {
  const [editedPlan, setEditedPlan] = useState(draft)
  const [feedbackText, setFeedbackText] = useState("")
  const isModified = editedPlan !== draft

  useEffect(() => {
    setEditedPlan(draft)
    setFeedbackText("")
  }, [draft])

  return (
    <section className="mx-auto my-4 max-w-3xl rounded-2xl border border-[var(--warning)] bg-[var(--warning-soft)] p-5">
      <div className="mb-3 flex items-center gap-2">
        <FilePenLine className="h-4 w-4 text-[var(--warning)]" />
        <span className="text-sm font-semibold text-[var(--warning)]">学习计划草稿等待审核</span>
      </div>

      <textarea
        value={editedPlan}
        onChange={(e) => setEditedPlan(e.target.value)}
        className={cn(
          "min-h-[200px] max-h-[400px] w-full resize-y rounded-lg border border-border bg-card",
          "p-3 font-mono text-sm leading-relaxed text-card-foreground",
          "focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/25"
        )}
        disabled={isSubmitting}
      />

      <div className="mt-3">
        <textarea
          value={feedbackText}
          onChange={(e) => setFeedbackText(e.target.value)}
          placeholder="例如：把第三周的项目任务拆得更细一些。"
          className={cn(
            "min-h-[60px] max-h-[120px] w-full resize-y rounded-lg border border-border bg-card",
            "p-3 text-sm leading-relaxed text-card-foreground placeholder:text-muted-foreground",
            "focus:border-[var(--warning)] focus:outline-none focus:ring-2 focus:ring-[var(--warning)]/25"
          )}
          disabled={isSubmitting}
        />
        <p className="mb-2 mt-1 text-xs text-muted-foreground">修改意见</p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs text-muted-foreground">
          {editedPlan.length} 字符
          {isModified && <span className="ml-2 text-[var(--warning)]">已修改</span>}
        </span>

        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => downloadPlan(editedPlan)}
            disabled={isSubmitting}
            className="gap-1.5 border-primary/30 text-xs text-primary hover:bg-primary/5"
          >
            <Download className="h-3.5 w-3.5" />
            下载计划
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onFeedback(feedbackText)}
            disabled={isSubmitting || !feedbackText.trim()}
            className="gap-1.5 border-[var(--warning)] text-xs text-[var(--warning)] hover:bg-[var(--warning)]/10"
          >
            <MessageSquareText className="h-3.5 w-3.5" />
            {isSubmitting ? "处理中..." : "要求修改"}
          </Button>
          <Button
            size="sm"
            onClick={() => onConfirm(editedPlan)}
            disabled={isSubmitting}
            className="a3-button-primary px-4 text-xs"
          >
            {isSubmitting ? "提交中..." : isModified ? "修改后确认" : "确认计划"}
          </Button>
        </div>
      </div>
    </section>
  )
}
