"use client"

import { useId, useRef, useState } from "react"

import { Button } from "@/components/ui/button"
import {
  parseAssessmentFinalV1,
  parseAssessmentSubmissionInput,
  parsePublicExerciseCardV1,
  type AssessmentErrorType,
  type AssessmentFinalV1,
  type AssessmentSubmissionInput,
  type PublicExerciseCardV1,
} from "@/lib/assessment-contracts"
import { cn } from "@/lib/utils"

export type ExerciseAssessmentState =
  | "idle"
  | "editing"
  | "submitting"
  | "correct"
  | "incorrect"
  | "failed"
  | "conflict"

export type ExerciseAssessmentSubmitResult =
  | { status: "completed"; final: AssessmentFinalV1 }
  | { status: "failed" }
  | { status: "conflict" }

export interface ExerciseAssessmentProps {
  resourceId: string
  question: PublicExerciseCardV1
  onSubmit: (
    submission: AssessmentSubmissionInput,
  ) => Promise<ExerciseAssessmentSubmitResult>
  className?: string
  disabled?: boolean
  disabledReason?: string
}

export function ExerciseAssessment(props: ExerciseAssessmentProps) {
  if (props.disabled && !props.disabledReason?.trim()) {
    throw new Error("ExerciseAssessment requires disabledReason when disabled")
  }
  const question = parsePublicExerciseCardV1(props.question)
  const questionKey = `${props.resourceId}:${question.question_id}`
  return <ExerciseAssessmentQuestion key={questionKey} {...props} question={question} />
}

function ExerciseAssessmentQuestion({
  resourceId,
  question,
  onSubmit,
  className,
  disabled = false,
  disabledReason,
}: ExerciseAssessmentProps) {
  const inputId = useId()
  const startedAtRef = useRef(Date.now())
  const submittingRef = useRef(false)
  const [answer, setAnswer] = useState("")
  const [state, setState] = useState<ExerciseAssessmentState>("idle")
  const [final, setFinal] = useState<AssessmentFinalV1 | null>(null)

  const hasAnswer = answer.trim().length > 0
  const answerLocked =
    disabled || state === "submitting" || state === "correct" || state === "incorrect"

  const updateAnswer = (nextAnswer: string) => {
    if (answerLocked) return
    setAnswer(nextAnswer)
    setFinal(null)
    setState(nextAnswer.trim() ? "editing" : "idle")
  }

  const submit = async () => {
    if (disabled || !hasAnswer || answerLocked || submittingRef.current) return

    submittingRef.current = true
    setFinal(null)
    setState("submitting")
    const timeSpentSeconds = Math.min(
      86_400,
      Math.max(0, (Date.now() - startedAtRef.current) / 1_000),
    )

    try {
      const submission = parseAssessmentSubmissionInput({
        resourceId,
        question,
        answer,
        timeSpentSeconds,
      })
      const result = await onSubmit(submission)
      if (result.status === "conflict") {
        setState("conflict")
        return
      }
      if (result.status === "failed") {
        setState("failed")
        return
      }

      const parsedFinal = parseAssessmentFinalV1(result.final)
      if (
        parsedFinal.resource_id !== resourceId ||
        parsedFinal.question_id !== question.question_id
      ) {
        setState("conflict")
        return
      }
      setFinal(parsedFinal)
      setState(parsedFinal.is_correct ? "correct" : "incorrect")
    } catch {
      setState("failed")
    } finally {
      submittingRef.current = false
    }
  }

  return (
    <section
      aria-labelledby={`${inputId}-question`}
      aria-busy={state === "submitting"}
      aria-describedby={disabled ? `${inputId}-disabled-reason` : undefined}
      data-assessment-state={state}
      className={cn("space-y-4 rounded-lg border border-border bg-card p-4", className)}
    >
      <header className="space-y-2">
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span className="rounded-full border border-border px-2 py-0.5">
            {levelLabel(question.level)}
          </span>
          {question.tags.map((tag) => (
            <span key={tag}>#{tag}</span>
          ))}
        </div>
        <h3 id={`${inputId}-question`} className="text-sm font-semibold leading-relaxed">
          {question.question}
        </h3>
      </header>

      {question.question_type === "free_text" ? (
        <div className="space-y-2">
          <label htmlFor={inputId} className="text-xs font-medium text-muted-foreground">
            你的答案
          </label>
          <textarea
            id={inputId}
            value={answer}
            onChange={(event) => updateAnswer(event.currentTarget.value)}
            disabled={answerLocked}
            autoComplete="off"
            maxLength={10_000}
            rows={4}
            className="w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-70"
          />
        </div>
      ) : (
        <fieldset disabled={answerLocked} className="space-y-2">
          <legend className="text-xs font-medium text-muted-foreground">选择一个答案</legend>
          {question.choices.map((choice, index) => {
            const choiceId = `${inputId}-choice-${index}`
            return (
              <label
                key={choice}
                htmlFor={choiceId}
                className={cn(
                  "flex cursor-pointer items-start gap-2 rounded-md border border-border px-3 py-2 text-sm",
                  answer === choice && "border-primary bg-accent",
                  answerLocked && "cursor-not-allowed opacity-70",
                )}
              >
                <input
                  id={choiceId}
                  type="radio"
                  name={`${inputId}-choices`}
                  value={choice}
                  checked={answer === choice}
                  onChange={(event) => updateAnswer(event.currentTarget.value)}
                  className="mt-0.5"
                />
                <span>{choice}</span>
              </label>
            )
          })}
        </fieldset>
      )}

      <div className="flex items-center gap-3">
        <Button
          type="button"
          onClick={() => void submit()}
          disabled={!hasAnswer || answerLocked}
        >
          {state === "submitting" ? "正在提交…" : "提交答案"}
        </Button>
        <StateSummary state={state} />
      </div>

      {disabled ? (
        <p
          id={`${inputId}-disabled-reason`}
          role="status"
          className="text-sm text-muted-foreground"
        >
          {disabledReason}
        </p>
      ) : null}

      {state === "incorrect" && final ? <IncorrectResult final={final} /> : null}
      {state === "failed" ? (
        <p role="alert" className="text-sm text-[var(--danger)]">
          提交失败，答案尚未完成评估。请确认连接后手动重试。
        </p>
      ) : null}
      {state === "conflict" ? (
        <p role="alert" className="text-sm text-[var(--warning)]">
          提交与当前题目或会话状态冲突。请刷新题目后再提交。
        </p>
      ) : null}
    </section>
  )
}

function StateSummary({ state }: { state: ExerciseAssessmentState }) {
  if (state === "correct") {
    return (
      <span role="status" aria-live="polite" className="text-sm text-[var(--success)]">
        回答正确
      </span>
    )
  }
  if (state === "incorrect") {
    return (
      <span role="status" aria-live="polite" className="text-sm text-[var(--warning)]">
        需要继续练习
      </span>
    )
  }
  if (state === "submitting") {
    return (
      <span role="status" aria-live="polite" className="text-sm text-muted-foreground">
        正在评估答案
      </span>
    )
  }
  return null
}

function IncorrectResult({ final }: { final: AssessmentFinalV1 }) {
  const classification = final.error_classification
  if (!classification) return null

  return (
    <div className="space-y-4 border-t border-border pt-4">
      <div className="space-y-1 rounded-md bg-[var(--warning-soft)] p-3 text-sm">
        <p className="font-semibold">错误类型：{errorTypeLabel(classification.error_type)}</p>
        <p>{classification.concept_gap}</p>
        <p className="text-muted-foreground">建议：{classification.suggestion}</p>
      </div>

      <div className="space-y-3">
        <h4 className="text-sm font-semibold">自适应练习</h4>
        {final.adaptive_tasks.map((task) => (
          <article key={task.question_id} className="space-y-2 rounded-md border border-border p-3">
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span>{practiceTypeLabel(task.task_type)}</span>
              <span>难度 {Math.round(task.difficulty * 100)}%</span>
            </div>
            <p className="text-sm font-medium">{task.question}</p>
            <details className="text-sm">
              <summary className="cursor-pointer text-primary">查看答案与解析</summary>
              <div className="mt-2 space-y-1 text-muted-foreground">
                <p>答案：{task.answer}</p>
                <p>解析：{task.explanation}</p>
                <p>练习原因：{task.reason}</p>
              </div>
            </details>
          </article>
        ))}
      </div>
    </div>
  )
}

function levelLabel(level: PublicExerciseCardV1["level"]): string {
  return {
    basic: "基础",
    intermediate: "进阶",
    application: "应用",
    self_check: "自测",
  }[level]
}

function errorTypeLabel(errorType: AssessmentErrorType): string {
  return {
    concept: "概念理解",
    logic: "推理逻辑",
    implementation: "实现过程",
  }[errorType]
}

function practiceTypeLabel(
  taskType: AssessmentFinalV1["adaptive_tasks"][number]["task_type"],
): string {
  return {
    similar: "同类巩固",
    harder: "进阶挑战",
    review: "回顾练习",
  }[taskType]
}
