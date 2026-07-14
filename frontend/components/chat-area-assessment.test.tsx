// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeAll, describe, expect, it, vi } from "vitest"

import { ChatArea, type Message } from "@/components/chat-area"
import type { AssessmentFinalV1 } from "@/lib/assessment-contracts"

const RESOURCE_ID = `resource:v3:${"a".repeat(64)}`
const QUESTION_ID = `question:v1:${"b".repeat(64)}`

beforeAll(() => {
  Object.defineProperty(Element.prototype, "scrollIntoView", {
    configurable: true,
    value: vi.fn(),
  })
  vi.stubGlobal(
    "requestAnimationFrame",
    (callback: FrameRequestCallback) => window.setTimeout(() => callback(0), 0),
  )
  vi.stubGlobal("cancelAnimationFrame", (handle: number) => window.clearTimeout(handle))
})

function assessmentMessage(): Message {
  return {
    id: "assistant-quiz",
    role: "assistant",
    content: "Quiz ready",
    threadId: "thread-assessment-1",
    requestId: "00000000-0000-4000-8000-000000000301",
    exercise: {
      title: "Arithmetic quiz",
      resourceId: RESOURCE_ID,
      questions: [
        {
          schema_version: "exercise_card_v1",
          question_id: QUESTION_ID,
          question_type: "free_text",
          level: "basic",
          question: "What is 2 + 2?",
          choices: [],
          tags: ["arithmetic"],
        },
      ],
    },
  }
}

function correctFinal(): AssessmentFinalV1 {
  return {
    schema_version: "assessment_final_v1",
    type: "assessment_final",
    thread_id: "thread-assessment-1",
    request_id: "00000000-0000-4000-8000-000000000302",
    resource_id: RESOURCE_ID,
    question_id: QUESTION_ID,
    terminal_status: "correct",
    is_correct: true,
    time_spent_seconds: 1,
    error_classification: null,
    adaptive_tasks: [],
    payload_hash: `assessment-final:v1:${"c".repeat(64)}`,
  }
}

describe("ChatArea assessment integration", () => {
  it("renders a validated resource question and delegates one private submission", async () => {
    const onSubmitAssessment = vi.fn().mockResolvedValue({
      status: "completed",
      final: correctFinal(),
    })
    render(
      <ChatArea
        messages={[assessmentMessage()]}
        onSendMessage={vi.fn()}
        onSubmitAssessment={onSubmitAssessment}
      />,
    )

    fireEvent.change(screen.getByLabelText("你的答案"), {
      target: { value: "4" },
    })
    fireEvent.click(screen.getByRole("button", { name: "提交答案" }))

    await waitFor(() => expect(screen.getByText("回答正确")).toBeInTheDocument())
    expect(onSubmitAssessment).toHaveBeenCalledTimes(1)
    expect(onSubmitAssessment).toHaveBeenCalledWith(
      expect.objectContaining({
        resourceId: RESOURCE_ID,
        answer: "4",
      }),
    )
  })

  it("locks chat input and other questions while an assessment request is active", () => {
    render(
      <ChatArea
        messages={[assessmentMessage()]}
        onSendMessage={vi.fn()}
        onSubmitAssessment={vi.fn()}
        activeAssessmentKey={`resource:v3:${"d".repeat(64)}:${QUESTION_ID}`}
      />,
    )

    expect(screen.getByPlaceholderText("输入你的问题...")).toBeDisabled()
    expect(screen.getByLabelText("你的答案")).toBeDisabled()
    expect(screen.getByText("另一道题正在评估，请等待结果后继续。")).toBeInTheDocument()
  })
})
