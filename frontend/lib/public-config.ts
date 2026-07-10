export function requirePublicApiBaseUrl(): string {
  const value = process.env.NEXT_PUBLIC_API_URL?.trim()
  if (!value) throw new Error("NEXT_PUBLIC_API_URL is required")
  return value.replace(/\/$/, "")
}
