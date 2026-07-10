import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"

import { describe, expect, it } from "vitest"

function frontendSource(relativePath: string): string {
  return readFileSync(fileURLToPath(new URL(`../${relativePath}`, import.meta.url)), "utf8")
}

describe("frontend observability boundaries", () => {
  it("does not restore a frontend-owned graph topology", () => {
    const source = frontendSource("components/right-panel.tsx")
    expect(source).not.toContain("NODE_LABELS")
    expect(source).not.toContain("DAG_NODE_IDS")
    expect(source).not.toContain("DAG_EDGE_DEFS")
    expect(source).toContain("ManifestGraph")
  })

  it("requires the public API endpoint instead of defaulting to localhost", () => {
    const source = frontendSource("lib/public-config.ts")
    expect(source).toContain("NEXT_PUBLIC_API_URL is required")
    expect(source).not.toContain('|| "http://')
    expect(source).not.toContain('?? "http://')
  })
})
