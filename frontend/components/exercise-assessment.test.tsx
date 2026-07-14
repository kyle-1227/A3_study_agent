// @vitest-environment jsdom

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import {
  ExerciseAssessment,
  type ExerciseAssessmentSubmitResult,
} from "@/components/exercise-assessment"
import type {
  AssessmentFinalV1,
  PublicExerciseCardV1,
} from "@/lib/assessment-contracts"

const RESOURCE_ID = `resource:v3:${"a".repeat(64)}`
const QUESTION_ID = `question:v1:${"b".repeat(64)}`

function freeTextQuestion(): PublicExerciseCardV1 {
  return {
    schema_version: "exercise_card_v1",
    question_id: QUESTION_ID,
    question_type: "free_text",
    level: "intermediate",
    question: "为什么二分查找要求搜索区间有序？",
    choices: [],
    tags: ["算法", "二分查找"],
  }
}

function singleChoiceQuestion(): PublicExerciseCardV1 {
  return {
    schema_version: "exercise_card_v1",
    question_id: QUESTION_ID,
    question_type: "single_choice",
    level: "basic",
    question: "1 + 2 等于多少？",
    choices: ["2", "3", "4"],
    tags: ["算术"],
  }
}

function correctFinal(): AssessmentFinalV1 {
  return {
    schema_version: "assessment_final_v1",
    type: "assessment_final",
    thread_id: "thread-1",
    request_id: "11111111-1111-4111-8111-111111111111",
    resource_id: RESOURCE_ID,
    question_id: QUESTION_ID,
    terminal_status: "correct",
    is_correct: true,
    time_spent_seconds: 2.5,
    error_classification: null,
    adaptive_tasks: [],
    payload_hash: `assessment-final:v1:${"c".repeat(64)}`,
  }
}

function incorrectFinal(): AssessmentFinalV1 {
  return {
    schema_version: "assessment_final_v1",
    type: "assessment_final",
    thread_id: "thread-1",
    request_id: "22222222-2222-4222-8222-222222222222",
    resource_id: RESOURCE_ID,
    question_id: QUESTION_ID,
    terminal_status: "incorrect",
    is_correct: false,
    time_spent_seconds: 1,
    error_classification: {
      schema_version: "assessment_error_classification_v1",
      error_type: "concept",
      concept_gap: "加法进位规则尚未掌握。",
      suggestion: "先用实物计数复核加法过程。",
      confidence: 0.9,
    },
    adaptive_tasks: [
      {
        schema_version: "adaptive_practice_task_v1",
        question_id: `question:v1:${"d".repeat(64)}`,
        task_type: "similar",
        question: "2 + 3 等于多少？",
        answer: "5",
        explanation: "从 2 开始再向后数 3 个数。",
        reason: "巩固一位数加法。",
        tags: ["算术"],
        difficulty: 0.2,
      },
    ],
    payload_hash: `assessment-final:v1:${"e".repeat(64)}`,
  }
}

describe("ExerciseAssessment", () => {
  it("moves from idle through editing to a bound correct result", async () => {
    const nowSpy = vi.spyOn(Date, "now").mockReturnValue(1_000)
    const onSubmit = vi.fn().mockResolvedValue({
      status: "completed",
      final: correctFinal(),
    } satisfies ExerciseAssessmentSubmitResult)

    render(
      <ExerciseAssessment
        resourceId={RESOURCE_ID}
        question={freeTextQuestion()}
        onSubmit={onSubmit}
      />,
    )

    const region = screen.getByRole("region", {
      name: "为什么二分查找要求搜索区间有序？",
    })
    const submitButton = screen.getByRole("button", { name: "提交答案" })
    expect(region).toHaveAttribute("data-assessment-state", "idle")
    expect(submitButton).toBeDisabled()

    fireEvent.change(screen.getByLabelText("你的答案"), {
      target: { value: "因为每一步都要排除有序区间的一半" },
    })
    expect(region).toHaveAttribute("data-assessment-state", "editing")
    expect(submitButton).toBeEnabled()

    nowSpy.mockReturnValue(3_500)
    fireEvent.click(submitButton)
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("回答正确"))
    expect(region).toHaveAttribute("data-assessment-state", "correct")
    expect(onSubmit).toHaveBeenCalledWith({
      resourceId: RESOURCE_ID,
      question: freeTextQuestion(),
      answer: "因为每一步都要排除有序区间的一半",
      timeSpentSeconds: 2.5,
    })
    expect(screen.getByLabelText("你的答案")).toBeDisabled()
  })

  it("blocks blank and duplicate submissions while one request is pending", async () => {
    let settle: ((result: ExerciseAssessmentSubmitResult) => void) | undefined
    const onSubmit = vi.fn(
      () =>
        new Promise<ExerciseAssessmentSubmitResult>((resolve) => {
          settle = resolve
        }),
    )
    render(
      <ExerciseAssessment
        resourceId={RESOURCE_ID}
        question={freeTextQuestion()}
        onSubmit={onSubmit}
      />,
    )

    const answer = screen.getByLabelText("你的答案")
    const submitButton = screen.getByRole("button", { name: "提交答案" })
    fireEvent.change(answer, { target: { value: "   " } })
    expect(submitButton).toBeDisabled()

    fireEvent.change(answer, { target: { value: "有效答案" } })
    fireEvent.click(submitButton)
    expect(screen.getByRole("button", { name: "正在提交…" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "正在提交…" }))
    expect(onSubmit).toHaveBeenCalledTimes(1)

    await act(async () => {
      settle?.({ status: "completed", final: correctFinal() })
    })
    expect(await screen.findByText("回答正确")).toBeInTheDocument()
  })

  it("renders strict single-choice remediation after an incorrect result", async () => {
    const onSubmit = vi.fn().mockResolvedValue({
      status: "completed",
      final: incorrectFinal(),
    } satisfies ExerciseAssessmentSubmitResult)
    render(
      <ExerciseAssessment
        resourceId={RESOURCE_ID}
        question={singleChoiceQuestion()}
        onSubmit={onSubmit}
      />,
    )

    fireEvent.click(screen.getByRole("radio", { name: "3" }))
    fireEvent.click(screen.getByRole("button", { name: "提交答案" }))

    expect(await screen.findByText("需要继续练习")).toBeInTheDocument()
    expect(screen.getByText("错误类型：概念理解")).toBeInTheDocument()
    expect(screen.getByText("加法进位规则尚未掌握。")).toBeInTheDocument()
    expect(screen.getByText("2 + 3 等于多少？")).toBeInTheDocument()
    expect(screen.getByText("答案：5")).toBeInTheDocument()
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ answer: "3", resourceId: RESOURCE_ID }),
    )
  })

  it("shows a safe explicit failure without logging or persisting the answer", async () => {
    const secretAnswer = "privacy-canary-answer"
    const storageSpy = vi.spyOn(Storage.prototype, "setItem")
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined)
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const onSubmit = vi.fn().mockRejectedValue(new Error(`provider body: ${secretAnswer}`))
    render(
      <ExerciseAssessment
        resourceId={RESOURCE_ID}
        question={freeTextQuestion()}
        onSubmit={onSubmit}
      />,
    )

    fireEvent.change(screen.getByLabelText("你的答案"), {
      target: { value: secretAnswer },
    })
    fireEvent.click(screen.getByRole("button", { name: "提交答案" }))

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("提交失败")
    expect(alert).not.toHaveTextContent(secretAnswer)
    expect(storageSpy).not.toHaveBeenCalled()
    expect(logSpy).not.toHaveBeenCalled()
    expect(warnSpy).not.toHaveBeenCalled()
    expect(errorSpy).not.toHaveBeenCalled()
  })

  it("surfaces an explicit conflict and does not retry silently", async () => {
    const onSubmit = vi.fn().mockResolvedValue({
      status: "conflict",
    } satisfies ExerciseAssessmentSubmitResult)
    render(
      <ExerciseAssessment
        resourceId={RESOURCE_ID}
        question={singleChoiceQuestion()}
        onSubmit={onSubmit}
      />,
    )

    fireEvent.click(screen.getByRole("radio", { name: "2" }))
    fireEvent.click(screen.getByRole("button", { name: "提交答案" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("状态冲突")
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it("rejects a final bound to another resource as a conflict", async () => {
    const onSubmit = vi.fn().mockResolvedValue({
      status: "completed",
      final: {
        ...correctFinal(),
        resource_id: `resource:v3:${"f".repeat(64)}`,
      },
    } satisfies ExerciseAssessmentSubmitResult)
    render(
      <ExerciseAssessment
        resourceId={RESOURCE_ID}
        question={freeTextQuestion()}
        onSubmit={onSubmit}
      />,
    )

    fireEvent.change(screen.getByLabelText("你的答案"), {
      target: { value: "有效答案" },
    })
    fireEvent.click(screen.getByRole("button", { name: "提交答案" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("状态冲突")
    expect(screen.queryByText("回答正确")).not.toBeInTheDocument()
  })

  it("disables the question with an explicit reason without marking it failed", () => {
    const onSubmit = vi.fn()
    render(
      <ExerciseAssessment
        resourceId={RESOURCE_ID}
        question={singleChoiceQuestion()}
        onSubmit={onSubmit}
        disabled
        disabledReason="主会话仍在运行，请等待本轮结束后作答。"
      />,
    )

    const region = screen.getByRole("region", { name: "1 + 2 等于多少？" })
    expect(region).toHaveAttribute("data-assessment-state", "idle")
    expect(screen.getByText("主会话仍在运行，请等待本轮结束后作答。")).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "3" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "提交答案" })).toBeDisabled()
    fireEvent.click(screen.getByRole("radio", { name: "3" }))
    fireEvent.click(screen.getByRole("button", { name: "提交答案" }))
    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })
})
