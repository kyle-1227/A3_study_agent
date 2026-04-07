# FE-REQ-01: SSE reconnection + error handling hardening

**Status:** DONE
**Date:** 2026-03-30

## Files Changed

| File | Change |
|------|--------|
| `frontend/app/page.tsx` | Extracted `API_BASE_URL` from env var; added 429/401 HTTP status handling with user-facing messages; attached `X-Access-Token` header from localStorage; added generic `!response.ok` guard |
| `frontend/.env.example` | **NEW** — documents `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_REQUIRE_TOKEN` |

## What Was Done

1. **NEXT_PUBLIC_API_URL env var** — replaced the single hardcoded `http://localhost:8000` with `process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"`.
2. **429 Too Many Requests** — returns an assistant message "⚠️ 服务繁忙，请稍后重试。" and a `[WARN]` log entry. Does not crash or retry.
3. **401 Unauthorized** — returns an assistant message "🔑 访问未授权，请检查访问令牌是否正确。", logs `[ERROR]`, and clears the stored token from localStorage so a future token gate (FE-REQ-10) can re-prompt.
4. **X-Access-Token header** — all `/stream` requests now include the header if a token exists in `localStorage("demo_access_token")`. This prepares for FE-REQ-10 (token gate UI).
5. **Generic HTTP error** — any other non-2xx response throws with status code + statusText, caught by the existing catch block.

## How to Verify

```bash
# Start backend
cd gaokao_tutor && python -m uvicorn app:app --reload --port 8000

# Start frontend
cd frontend && npm run dev
# Open http://localhost:3000

# Test 429: temporarily make backend return 429, or use curl to confirm frontend message
# Test 401: set X-Access-Token validation on backend, send without token
# Test env var: set NEXT_PUBLIC_API_URL=http://localhost:9999 in .env.local, restart dev server, confirm fetch goes to port 9999
```

## Suggested Commit

```bash
git add frontend/app/page.tsx frontend/.env.example
git commit -m "feat(frontend): FE-REQ-01 SSE error handling + NEXT_PUBLIC_API_URL env var"
```
