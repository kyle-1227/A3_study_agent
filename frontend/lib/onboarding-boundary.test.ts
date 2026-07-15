import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"

import { describe, expect, it } from "vitest"

function frontendSource(relativePath: string): string {
  return readFileSync(fileURLToPath(new URL(`../${relativePath}`, import.meta.url)), "utf8")
}

describe("onboarding V2 frontend boundary", () => {
  it("uses only the learning-guidance catalog and configured API base", () => {
    const page = frontendSource("app/onboarding/page.tsx")
    const client = frontendSource("lib/onboarding-client.ts")
    const userHook = frontendSource("hooks/use-user.tsx")

    expect(client).toContain("/learning-guidance/catalog")
    expect(page).toContain("requirePublicApiBaseUrl")
    expect(userHook).toContain("requirePublicApiBaseUrl")
    expect(page).not.toContain('fetch("http://localhost')
    expect(userHook).not.toContain('fetch(`http://localhost')
    expect(page).not.toContain("/subjects")
    expect(client).not.toContain("/subjects")
    expect(page).not.toContain("SUBJECT_META")
    expect(page).not.toContain("customSubject")
  })

  it("does not log profile payloads or raw remote failures", () => {
    const page = frontendSource("app/onboarding/page.tsx")
    const client = frontendSource("lib/onboarding-client.ts")

    expect(page).not.toContain("console.log")
    expect(page).not.toContain("console.error")
    expect(client).not.toContain("response.text()")
  })
})
