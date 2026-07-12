export interface StreamingMarkdownParts {
  stablePrefix: string
  unstableSuffix: string
}

/** Keep the currently open block in the suffix while freezing completed paragraphs. */
export function splitStreamingMarkdown(value: string): StreamingMarkdownParts {
  if (!value) return { stablePrefix: "", unstableSuffix: "" }

  const openFenceStart = findOpenFenceStart(value)
  let boundary = lastParagraphBoundary(value)
  if (openFenceStart >= 0 && boundary > openFenceStart) boundary = openFenceStart
  if (boundary <= 0) return { stablePrefix: "", unstableSuffix: value }
  return {
    stablePrefix: value.slice(0, boundary),
    unstableSuffix: value.slice(boundary),
  }
}

function lastParagraphBoundary(value: string): number {
  const match = /(?:\r?\n){2,}/g
  let boundary = 0
  for (const item of value.matchAll(match)) {
    boundary = (item.index ?? 0) + item[0].length
  }
  return boundary
}

function findOpenFenceStart(value: string): number {
  const linePattern = /^( {0,3})(```+|~~~+).*$/gm
  let open: { marker: string; start: number } | null = null
  for (const match of value.matchAll(linePattern)) {
    const marker = match[2]
    if (open === null) {
      open = { marker: marker[0], start: match.index ?? 0 }
    } else if (marker[0] === open.marker) {
      open = null
    }
  }
  return open?.start ?? -1
}
