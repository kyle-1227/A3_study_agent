"use client"

import { useCallback, useEffect, useState } from "react"
import { useRouter } from "next/navigation"

import { ONBOARDING_ATTEMPT_STORAGE_PREFIX } from "@/lib/onboarding-client"
import { requirePublicApiBaseUrl } from "@/lib/public-config"

const USER_ID_KEY = "a3_user_id"
const NICKNAME_KEY = "a3_nickname"
const ONBOARDING_COMPLETED_KEY = "a3_onboarding_completed"
const API_BASE_URL = requirePublicApiBaseUrl()

export type ProfileAvailability =
  | "loading"
  | "available"
  | "missing"
  | "unavailable"

function generateUserId(): string {
  return `u_${crypto.randomUUID()}`
}

function getStoredUserId(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(USER_ID_KEY)
}

function getStoredNickname(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(NICKNAME_KEY)
}

export function useUser(): {
  userId: string | null
  nickname: string | null
  hasProfile: boolean
  profileAvailability: ProfileAvailability
  isLoading: boolean
  startOnboarding: () => void
  clearUser: () => void
} {
  const [userId, setUserId] = useState<string | null>(null)
  const [nickname, setNickname] = useState<string | null>(null)
  const [profileAvailability, setProfileAvailability] =
    useState<ProfileAvailability>("loading")
  const router = useRouter()

  // Check for existing user on mount
  useEffect(() => {
    const stored = getStoredUserId()
    if (stored) {
      setUserId(stored)
      setNickname(getStoredNickname())

      // Verify the profile exists on the backend
      fetch(`${API_BASE_URL}/profile/${encodeURIComponent(stored)}`, {
        headers: { Accept: "application/json" },
      })
        .then((res) => {
          if (res.ok) {
            setProfileAvailability("available")
          } else if (res.status === 404) {
            setProfileAvailability("missing")
          } else {
            setProfileAvailability("unavailable")
          }
        })
        .catch(() => setProfileAvailability("unavailable"))
    } else {
      setProfileAvailability("missing")
    }
  }, [])

  const startOnboarding = useCallback(() => {
    const uid = generateUserId()
    localStorage.setItem(USER_ID_KEY, uid)
    setUserId(uid)
    setProfileAvailability("missing")
    router.push("/onboarding")
  }, [router])

  const clearUser = useCallback(() => {
    const stored = getStoredUserId()
    if (stored) localStorage.removeItem(`${ONBOARDING_ATTEMPT_STORAGE_PREFIX}${stored}`)
    localStorage.removeItem(USER_ID_KEY)
    localStorage.removeItem(NICKNAME_KEY)
    localStorage.removeItem(ONBOARDING_COMPLETED_KEY)
    setUserId(null)
    setNickname(null)
    setProfileAvailability("missing")
  }, [])

  return {
    userId,
    nickname,
    hasProfile: profileAvailability === "available",
    profileAvailability,
    isLoading: profileAvailability === "loading",
    startOnboarding,
    clearUser,
  }
}
