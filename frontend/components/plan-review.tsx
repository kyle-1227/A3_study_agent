"use client"

import { useState, useEffect } from "react"
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

  // Sync when draft prop changes (re-interrupt after feedback revision)
  useEffect(() => {
    setEditedPlan(draft)
    setFeedbackText("")
  }, [draft])

  return (
    <div className="bg-[#FFF9E6] border border-[#E8A87C] rounded-2xl p-5 my-4 max-w-3xl mx-auto">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-sm font-semibold text-[#5C3D2E]">📋 学习计划草稿 — 请审阅</span>
      </div>

      <textarea
        value={editedPlan}
        onChange={(e) => setEditedPlan(e.target.value)}
        className={cn(
          "w-full min-h-[200px] max-h-[400px] resize-y rounded-lg border border-[#E8E5D8] bg-white",
          "p-3 text-sm leading-relaxed font-mono",
          "focus:outline-none focus:ring-2 focus:ring-[#3D5A40]/30 focus:border-[#3D5A40]"
        )}
        disabled={isSubmitting}
      />

      <div className="mt-3">
        <textarea
          value={feedbackText}
          onChange={(e) => setFeedbackText(e.target.value)}
          placeholder="例如：把周三的数学改成物理"
          className={cn(
            "w-full min-h-[60px] max-h-[120px] resize-y rounded-lg border border-[#E8E5D8] bg-white",
            "p-3 text-sm leading-relaxed",
            "focus:outline-none focus:ring-2 focus:ring-[#E8A87C]/30 focus:border-[#E8A87C]"
          )}
          disabled={isSubmitting}
        />
        <p className="text-xs text-muted-foreground mt-1 mb-2">修改意见</p>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {editedPlan.length} 字符
          {isModified && <span className="ml-2 text-[#E8A87C]">（已修改）</span>}
        </span>

        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => downloadPlan(editedPlan)}
            disabled={isSubmitting}
            className="text-xs border-[#3D5A40]/30 text-[#3D5A40] hover:bg-[#3D5A40]/5"
          >
            下载计划
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onFeedback(feedbackText)}
            disabled={isSubmitting || !feedbackText.trim()}
            className="text-xs border-[#E8A87C] text-[#5C3D2E] hover:bg-[#E8A87C]/10"
          >
            {isSubmitting ? "处理中..." : "要求修改"}
          </Button>
          <Button
            size="sm"
            onClick={() => onConfirm(editedPlan)}
            disabled={isSubmitting}
            className="bg-[#3D5A40] hover:bg-[#4A6B4D] text-white text-xs px-4"
          >
            {isSubmitting ? "提交中..." : isModified ? "修改后确认" : "确认计划"}
          </Button>
        </div>
      </div>
    </div>
  )
}
