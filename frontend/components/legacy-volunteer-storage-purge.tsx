"use client"

import { useEffect } from "react"

import { purgeLegacyVolunteerStorage } from "@/lib/legacy-volunteer-storage"

export function LegacyVolunteerStoragePurge() {
  useEffect(() => {
    try {
      purgeLegacyVolunteerStorage(window.localStorage)
    } catch (error) {
      console.error("Failed to purge retired volunteer module storage", error)
    }
  }, [])

  return null
}
