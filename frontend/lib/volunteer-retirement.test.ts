import { existsSync, readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"

import { describe, expect, it } from "vitest"

function frontendPath(relativePath: string): string {
  return fileURLToPath(new URL(`../${relativePath}`, import.meta.url))
}

describe("volunteer module retirement boundary", () => {
  it("does not expose the retired route or navigation entries", () => {
    expect(existsSync(frontendPath("app/volunteer/page.tsx"))).toBe(false)

    const chatArea = readFileSync(frontendPath("components/chat-area.tsx"), "utf8")
    const leftSidebar = readFileSync(frontendPath("components/left-sidebar.tsx"), "utf8")
    expect(chatArea).not.toContain('router.push("/volunteer")')
    expect(leftSidebar).not.toContain("/volunteer")
    expect(leftSidebar).not.toContain("VolunteerHistoryItem")
  })

  it("mounts the one-release storage purge from the root layout", () => {
    const layout = readFileSync(frontendPath("app/layout.tsx"), "utf8")
    expect(layout).toContain("<LegacyVolunteerStoragePurge />")
  })
})
