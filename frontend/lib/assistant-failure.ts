export interface AssistantFailureMessage {
  role: "user" | "assistant"
  content: string
}

// This is a terminal UI state, not a generated-answer substitute.
export const SAFE_ASSISTANT_FAILURE_CONTENT = "本次回答未能完成，请稍后重试。"

export function mergeSafeFailureContent<T extends AssistantFailureMessage>(message: T): T {
  if (message.role !== "assistant") return message
  if (message.content.trim()) return message
  return { ...message, content: SAFE_ASSISTANT_FAILURE_CONTENT }
}
