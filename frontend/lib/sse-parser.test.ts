import { describe, expect, it } from "vitest"

import { SSEParseError, SSEParser } from "@/lib/sse-parser"

const encoder = new TextEncoder()

describe("SSEParser", () => {
  it("handles UTF-8 splits, CRLF, multiline data, id, event, and retry", () => {
    const parser = new SSEParser()
    const bytes = encoder.encode(
      "event: content_block_delta\r\nid: stream:2\r\nretry: 1500\r\ndata: {\"text\":\"你\r\ndata: 好\"}\r\n\r\n",
    )
    const frames = [
      ...parser.feed(bytes.slice(0, bytes.indexOf(0xe4) + 1)),
      ...parser.feed(bytes.slice(bytes.indexOf(0xe4) + 1)),
      ...parser.finish(),
    ]
    expect(frames).toEqual([
      {
        event: "content_block_delta",
        id: "stream:2",
        retry: 1500,
        data: '{"text":"你\n好"}',
      },
    ])
  })

  it("flushes a final event without a trailing blank line", () => {
    const parser = new SSEParser()
    expect(parser.feed(encoder.encode("data: {\"ok\":true}"))).toEqual([])
    expect(parser.finish()).toEqual([
      { event: "message", id: "", data: '{"ok":true}' },
    ])
  })

  it("rejects malformed retry values", () => {
    const parser = new SSEParser()
    expect(() => parser.feed(encoder.encode("retry: nope\n"))).toThrow(SSEParseError)
  })
})
