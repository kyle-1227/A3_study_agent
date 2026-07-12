import { describe, expect, it } from "vitest"

import { splitStreamingMarkdown } from "@/lib/streaming-markdown"

describe("splitStreamingMarkdown", () => {
  it("freezes completed paragraphs", () => {
    expect(splitStreamingMarkdown("first\n\nsecond")).toEqual({
      stablePrefix: "first\n\n",
      unstableSuffix: "second",
    })
  })

  it("keeps an open code fence in the unstable suffix", () => {
    expect(splitStreamingMarkdown("intro\n\n```ts\nconst x = 1\n\nnext")).toEqual({
      stablePrefix: "intro\n\n",
      unstableSuffix: "```ts\nconst x = 1\n\nnext",
    })
  })

  it("allows closed code fences to become stable", () => {
    expect(splitStreamingMarkdown("intro\n\n```ts\nx\n```\n\nafter")).toEqual({
      stablePrefix: "intro\n\n```ts\nx\n```\n\n",
      unstableSuffix: "after",
    })
  })
})
