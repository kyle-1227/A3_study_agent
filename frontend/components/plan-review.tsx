"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface PlanReviewProps {
  draft: string
  onConfirm: (editedPlan: string) => void
  isSubmitting?: boolean
}

export function PlanReview({ draft, onConfirm, isSubmitting }: PlanReviewProps) {
  const [editedPlan, setEditedPlan] = useState(draft)
  const isModified = editedPlan !== draft

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

      <div className="flex items-center justify-between mt-3">
        <span className="text-xs text-muted-foreground">
          {editedPlan.length} 字符
          {isModified && <span className="ml-2 text-[#E8A87C]">（已修改）</span>}
        </span>

        <div className="flex gap-2">
          {isModified && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setEditedPlan(draft)}
              disabled={isSubmitting}
              className="text-xs text-muted-foreground hover:text-[#5C3D2E]"
            >
              重置
            </Button>
          )}
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
