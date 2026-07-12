export interface SSEFrame {
  event: string
  id: string
  data: string
  retry?: number
}

export class SSEParseError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "SSEParseError"
  }
}

export class SSEParser {
  private readonly decoder = new TextDecoder("utf-8", { fatal: true })
  private buffer = ""
  private event = "message"
  private id = ""
  private retry: number | undefined
  private dataLines: string[] = []

  feed(chunk: Uint8Array): SSEFrame[] {
    this.buffer += this.decoder.decode(chunk, { stream: true })
    return this.consumeLines(false)
  }

  finish(): SSEFrame[] {
    this.buffer += this.decoder.decode()
    const frames = this.consumeLines(true)
    const final = this.dispatch()
    if (final) frames.push(final)
    return frames
  }

  private consumeLines(finishing: boolean): SSEFrame[] {
    const frames: SSEFrame[] = []
    while (true) {
      const boundary = lineBoundary(this.buffer, finishing)
      if (!boundary) break
      const line = this.buffer.slice(0, boundary.index)
      this.buffer = this.buffer.slice(boundary.index + boundary.length)
      const frame = this.processLine(line)
      if (frame) frames.push(frame)
    }
    if (finishing && this.buffer.length > 0) {
      const frame = this.processLine(this.buffer)
      this.buffer = ""
      if (frame) frames.push(frame)
    }
    return frames
  }

  private processLine(line: string): SSEFrame | null {
    if (line === "") return this.dispatch()
    if (line.startsWith(":")) return null
    const colon = line.indexOf(":")
    const field = colon >= 0 ? line.slice(0, colon) : line
    let value = colon >= 0 ? line.slice(colon + 1) : ""
    if (value.startsWith(" ")) value = value.slice(1)
    if (field === "event") this.event = value || "message"
    else if (field === "id") {
      if (value.includes("\0")) throw new SSEParseError("id cannot contain NUL")
      this.id = value
    } else if (field === "data") this.dataLines.push(value)
    else if (field === "retry") {
      if (!/^\d+$/.test(value) || Number(value) <= 0) {
        throw new SSEParseError("retry must be a positive integer")
      }
      this.retry = Number(value)
    }
    return null
  }

  private dispatch(): SSEFrame | null {
    if (this.dataLines.length === 0) {
      this.event = "message"
      this.retry = undefined
      return null
    }
    const frame: SSEFrame = {
      event: this.event,
      id: this.id,
      data: this.dataLines.join("\n"),
      ...(this.retry === undefined ? {} : { retry: this.retry }),
    }
    this.event = "message"
    this.retry = undefined
    this.dataLines = []
    return frame
  }
}

function lineBoundary(
  value: string,
  finishing: boolean,
): { index: number; length: number } | null {
  for (let index = 0; index < value.length; index += 1) {
    if (value[index] === "\n") return { index, length: 1 }
    if (value[index] !== "\r") continue
    if (index + 1 < value.length) {
      return { index, length: value[index + 1] === "\n" ? 2 : 1 }
    }
    return finishing ? { index, length: 1 } : null
  }
  return null
}
