export const LEGACY_VOLUNTEER_HISTORY_KEY = "volunteer_chat_history"
export const LEGACY_VOLUNTEER_CHAT_PREFIX = "volunteer_chat_"
export const LEGACY_VOLUNTEER_PURGE_MARKER_KEY =
  "a3_migration_volunteer_storage_purged_v1"

export interface LegacyVolunteerStorage {
  readonly length: number
  getItem(key: string): string | null
  key(index: number): string | null
  removeItem(key: string): void
  setItem(key: string, value: string): void
}

export interface LegacyVolunteerStoragePurgeResult {
  status: "already_purged" | "purged"
  removedKeys: string[]
}

export function purgeLegacyVolunteerStorage(
  storage: LegacyVolunteerStorage,
): LegacyVolunteerStoragePurgeResult {
  const wasPreviouslyPurged =
    storage.getItem(LEGACY_VOLUNTEER_PURGE_MARKER_KEY) === "1"

  const keysToRemove: string[] = []
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index)
    if (
      key === LEGACY_VOLUNTEER_HISTORY_KEY ||
      key?.startsWith(LEGACY_VOLUNTEER_CHAT_PREFIX)
    ) {
      keysToRemove.push(key)
    }
  }

  for (const key of keysToRemove) storage.removeItem(key)
  if (!wasPreviouslyPurged) {
    storage.setItem(LEGACY_VOLUNTEER_PURGE_MARKER_KEY, "1")
  }

  return {
    status:
      wasPreviouslyPurged && keysToRemove.length === 0
        ? "already_purged"
        : "purged",
    removedKeys: keysToRemove,
  }
}
