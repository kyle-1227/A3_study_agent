"use client"

import { useCallback, useEffect, useState } from "react"
import { useRouter } from "next/navigation"

const USER_ID_KEY = "a3_user_id"
const NICKNAME_KEY = "a3_nickname"

function generateUserId(): string {
  const t = Date.now().toString(36)
  const r = Math.random().toString(36).slice(2, 10)
  return `u_${t}_${r}`
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
  isLoading: boolean
  startOnboarding: () => void
  clearUser: () => void
} {
  const [userId, setUserId] = useState<string | null>(null)
  const [nickname, setNickname] = useState<string | null>(null)
  const [hasProfile, setHasProfile] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  // Check for existing user on mount
  useEffect(() => {
    const stored = getStoredUserId()
    if (stored) {
      setUserId(stored)
      setNickname(getStoredNickname())

      // Optimistic: if onboarding was just completed, the profile definitely
      // exists — skip the backend fetch to avoid a race where the fetch
      // overrides hasProfile=false and bounces the user back to /onboarding.
      if (typeof window !== "undefined" && localStorage.getItem("a3_onboarding_completed")) {
        setHasProfile(true)
        setIsLoading(false)
        return
      }

      // Verify the profile exists on the backend
      fetch(`http://localhost:8000/profile/${stored}`)
        .then((res) => {
          setHasProfile(res.ok)
          if (res.ok) {
            return res.json()
          }
          return null
        })
        .then((data) => {
          if (data?.nickname) {
            localStorage.setItem(NICKNAME_KEY, data.nickname)
            setNickname(data.nickname)
          }
        })
        .catch(() => setHasProfile(false))
        .finally(() => setIsLoading(false))
    } else {
      setIsLoading(false)
    }
  }, [])

  const startOnboarding = useCallback(() => {
    const uid = generateUserId()
    localStorage.setItem(USER_ID_KEY, uid)
    setUserId(uid)
    router.push("/onboarding")
  }, [router])

  const clearUser = useCallback(() => {
    localStorage.removeItem(USER_ID_KEY)
    localStorage.removeItem(NICKNAME_KEY)
    setUserId(null)
    setNickname(null)
    setHasProfile(false)
  }, [])

  return { userId, nickname, hasProfile, isLoading, startOnboarding, clearUser }
}
