import { describe, expect, it } from "vitest"

import {
  LEGACY_VOLUNTEER_PURGE_MARKER_KEY,
  purgeLegacyVolunteerStorage,
  type LegacyVolunteerStorage,
} from "@/lib/legacy-volunteer-storage"

function memoryStorage(initial: Record<string, string>): {
  storage: LegacyVolunteerStorage
  snapshot: () => Record<string, string>
} {
  const values = new Map(Object.entries(initial))
  const storage: LegacyVolunteerStorage = {
    get length() {
      return values.size
    },
    getItem(key) {
      return values.get(key) ?? null
    },
    key(index) {
      return [...values.keys()][index] ?? null
    },
    removeItem(key) {
      values.delete(key)
    },
    setItem(key, value) {
      values.set(key, value)
    },
  }
  return { storage, snapshot: () => Object.fromEntries(values) }
}

describe("retired volunteer storage purge", () => {
  it("removes only the retired history and per-chat keys", () => {
    const { storage, snapshot } = memoryStorage({
      volunteer_chat_history: "[]",
      volunteer_chat_first: "first",
      volunteer_chat_second: "second",
      chat_history: "keep",
      unrelated: "keep",
    })

    expect(purgeLegacyVolunteerStorage(storage)).toEqual({
      status: "purged",
      removedKeys: [
        "volunteer_chat_history",
        "volunteer_chat_first",
        "volunteer_chat_second",
      ],
    })
    expect(snapshot()).toEqual({
      chat_history: "keep",
      unrelated: "keep",
      [LEGACY_VOLUNTEER_PURGE_MARKER_KEY]: "1",
    })
  })

  it("removes keys rewritten by an old tab after the release marker was set", () => {
    const { storage, snapshot } = memoryStorage({
      [LEGACY_VOLUNTEER_PURGE_MARKER_KEY]: "1",
      volunteer_chat_late: "stale",
    })

    expect(purgeLegacyVolunteerStorage(storage)).toEqual({
      status: "purged",
      removedKeys: ["volunteer_chat_late"],
    })
    expect(snapshot()).toEqual({
      [LEGACY_VOLUNTEER_PURGE_MARKER_KEY]: "1",
    })
    expect(purgeLegacyVolunteerStorage(storage)).toEqual({
      status: "already_purged",
      removedKeys: [],
    })
  })
})
