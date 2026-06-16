"use client"

import { Suspense, useCallback, useEffect, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  BrainCircuit,
  Check,
  Code,
  Database,
  GraduationCap,
  Monitor,
  Sigma,
  Sparkles,
  Target,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

// ── Types ────────────────────────────────────────────────────────────────────

type Step = "welcome" | "subjects" | "skills" | "goals"

interface OnboardingData {
  grade: string
  subjects: string[]
  skillLevels: Record<string, number>
  goals: string[]
  learningStyle: Record<string, number>
  dislikes: string[]
}

// ── Constants ────────────────────────────────────────────────────────────────

const GRADE_OPTIONS = ["大一", "大二", "大三", "大四", "研一", "研二", "研三", "高三", "其他"]

const SKILL_LEVELS = [
  { value: 0.25, label: "入门", desc: "刚开始接触" },
  { value: 0.5, label: "中等", desc: "有一定基础" },
  { value: 0.75, label: "熟练", desc: "掌握较好" },
]

const SUBJECT_META: Record<string, { zh: string; icon: string; desc: string }> = {
  python: { zh: "Python 编程", icon: "Code", desc: "Python 语言基础、数据分析、Web 开发" },
  computer: { zh: "计算机基础", icon: "Monitor", desc: "操作系统、计算机网络、数据结构" },
  big_data: { zh: "大数据", icon: "Database", desc: "数据处理、分布式系统、Spark/Hadoop" },
  machine_learning: { zh: "机器学习", icon: "BrainCircuit", desc: "ML 算法、深度学习、NLP" },
  math: { zh: "数学", icon: "Sigma", desc: "线性代数、概率论、微积分、优化" },
}

const STYLE_QUESTIONS: { dim: string; question: string }[] = [
  { dim: "prefer_examples", question: "你喜欢通过具体案例来学习吗？" },
  { dim: "prefer_visual", question: "你喜欢看图/图表/可视化吗？" },
  { dim: "prefer_step_by_step", question: "你喜欢一步步分步讲解吗？" },
  { dim: "prefer_concise", question: "你喜欢简短直接的答案吗？" },
  { dim: "prefer_theory", question: "你喜欢深入理解原理吗？" },
  { dim: "prefer_practice", question: "你喜欢动手练习而不是看理论吗？" },
  { dim: "prefer_analogy", question: "你喜欢用类比来理解吗？" },
]

const STYLE_OPTIONS = [
  { value: 0.2, label: "不太喜欢" },
  { value: 0.5, label: "一般" },
  { value: 0.8, label: "很喜欢" },
]

const STEP_LABELS: Record<Step, string> = {
  welcome: "欢迎",
  subjects: "学习方向",
  skills: "技能水平",
  goals: "目标与偏好",
}

function subjectIcon(iconName: string) {
  const icons: Record<string, React.ReactNode> = {
    Code: <Code className="h-5 w-5" />,
    Monitor: <Monitor className="h-5 w-5" />,
    Database: <Database className="h-5 w-5" />,
    BrainCircuit: <BrainCircuit className="h-5 w-5" />,
    Sigma: <Sigma className="h-5 w-5" />,
  }
  return icons[iconName] || <BookOpen className="h-5 w-5" />
}

// ── Inner component (uses searchParams) ──────────────────────────────────────

function OnboardingPageInner() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [step, setStep] = useState<Step>("welcome")
  const [availableSubjects, setAvailableSubjects] = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")

  const [data, setData] = useState<OnboardingData>({
    grade: "",
    subjects: [],
    skillLevels: {},
    goals: [""],
    learningStyle: {},
    dislikes: [],
  })

  const [customSubject, setCustomSubject] = useState("")
  const [dislikeInput, setDislikeInput] = useState("")

  // Fetch available subjects on mount
  useEffect(() => {
    fetch("http://localhost:8000/subjects")
      .then((r) => r.json())
      .then((d) => setAvailableSubjects(d.subjects || []))
      .catch(() => setAvailableSubjects(Object.keys(SUBJECT_META)))
  }, [])

  // Auto-generate userId if missing (so onboarding is self-sufficient)
  const [resolvedUserId, setResolvedUserId] = useState<string | null>(null)

  useEffect(() => {
    if (typeof window === "undefined") return
    const existing = localStorage.getItem("a3_user_id")
    if (existing) {
      setResolvedUserId(existing)
    } else {
      // Generate a new id just like the login flow does
      const t = Date.now().toString(36)
      const r = Math.random().toString(36).slice(2, 10)
      const uid = `u_${t}_${r}`
      localStorage.setItem("a3_user_id", uid)
      setResolvedUserId(uid)
    }
  }, [])

  // ── Step-specific handlers ──────────────────────────────────────────────

  const toggleSubject = (s: string) => {
    setData((prev) => {
      const next = prev.subjects.includes(s)
        ? prev.subjects.filter((x) => x !== s)
        : [...prev.subjects, s]
      return { ...prev, subjects: next }
    })
  }

  const addCustomSubject = () => {
    const trimmed = customSubject.trim()
    if (!trimmed || data.subjects.includes(trimmed)) return
    setData((prev) => ({ ...prev, subjects: [...prev.subjects, trimmed] }))
    setCustomSubject("")
  }

  const setSkillLevel = (subject: string, level: number) => {
    setData((prev) => ({
      ...prev,
      skillLevels: { ...prev.skillLevels, [subject]: level },
    }))
  }

  const setGoal = (index: number, value: string) => {
    setData((prev) => {
      const next = [...prev.goals]
      next[index] = value
      return { ...prev, goals: next }
    })
  }

  const addGoal = () => {
    setData((prev) => ({ ...prev, goals: [...prev.goals, ""] }))
  }

  const removeGoal = (index: number) => {
    setData((prev) => ({
      ...prev,
      goals: prev.goals.filter((_, i) => i !== index),
    }))
  }

  const setStylePreference = (dim: string, value: number) => {
    setData((prev) => ({
      ...prev,
      learningStyle: { ...prev.learningStyle, [dim]: value },
    }))
  }

  const addDislike = () => {
    const trimmed = dislikeInput.trim()
    if (!trimmed || data.dislikes.includes(trimmed)) return
    setData((prev) => ({ ...prev, dislikes: [...prev.dislikes, trimmed] }))
    setDislikeInput("")
  }

  const removeDislike = (item: string) => {
    setData((prev) => ({ ...prev, dislikes: prev.dislikes.filter((x) => x !== item) }))
  }

  // ── Navigation ──────────────────────────────────────────────────────────

  const stepOrder: Step[] = ["welcome", "subjects", "skills", "goals"]
  const currentIndex = stepOrder.indexOf(step)

  const goNext = () => {
    if (currentIndex < stepOrder.length - 1) {
      setStep(stepOrder[currentIndex + 1])
    }
  }

  const goBack = () => {
    if (currentIndex > 0) {
      setStep(stepOrder[currentIndex - 1])
    }
  }

  // ── Submit ──────────────────────────────────────────────────────────────

  const canSubmit =
    data.subjects.length > 0 &&
    data.grade.trim() !== ""

  const handleSubmit = async () => {
    if (!resolvedUserId) {
      setError("用户标识生成中，请稍后再试")
      return
    }
    setSubmitting(true)
    setError("")

    try {
      const nickname = localStorage.getItem("a3_nickname") || ""
      const payload = {
        user_id: resolvedUserId,
        nickname,
        subjects: data.subjects,
        skill_levels: data.skillLevels,
        goals: data.goals.filter((g) => g.trim()),
        learning_style: data.learningStyle,
        grade: data.grade,
        dislikes: data.dislikes,
      }
      console.log("[onboarding] submitting", { userId: resolvedUserId, subjects: data.subjects, grade: data.grade })

      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 30_000)

      const res = await fetch("http://localhost:8000/onboard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })
      clearTimeout(timeout)

      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || `HTTP ${res.status}`)
      }

      const result = await res.json()
      console.log("[onboarding] success", result)

      localStorage.setItem("a3_onboarding_completed", "true")
      router.push("/")
    } catch (e: any) {
      if (e.name === "AbortError") {
        setError("请求超时，请检查后端服务是否已启动")
      } else if (e.name === "TypeError" && e.message.includes("fetch")) {
        setError("无法连接后端服务，请确保 API 服务已启动 (localhost:8000)")
      } else {
        setError(e.message || "提交失败，请重试")
      }
      console.error("[onboarding] submit error", e)
    } finally {
      setSubmitting(false)
    }
  }

  // ── Progress bar ────────────────────────────────────────────────────────

  const progressPct = ((currentIndex + 1) / stepOrder.length) * 100

  if (!resolvedUserId) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-[var(--muted-foreground)]">正在初始化...</p>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--background)] p-4">
      <div className="w-full max-w-2xl">
        {/* Progress bar */}
        <div className="mb-8">
          <div className="mb-2 flex items-center justify-between text-sm text-[var(--muted-foreground)]">
            <span>
              步骤 {currentIndex + 1}/{stepOrder.length}：{STEP_LABELS[step]}
            </span>
            <span>{Math.round(progressPct)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-[var(--border)]">
            <div
              className="h-full rounded-full bg-[var(--primary)] transition-all duration-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-700 dark:bg-red-950 dark:text-red-300">
            {error}
          </div>
        )}

        {/* Step content */}
        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-sm">
          {/* ── Welcome ────────────────────────────────────────────────── */}
          {step === "welcome" && (
            <div className="space-y-6">
              <div className="text-center">
                <Sparkles className="mx-auto mb-3 h-10 w-10 text-[var(--primary)]" />
                <h1 className="text-2xl font-bold">欢迎来到高校学习 AI 助手</h1>
                <p className="mt-2 text-[var(--muted-foreground)]">
                  让我们花 2 分钟了解你，以便为你提供个性化的学习体验
                </p>
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium">
                  你目前所在的年级？
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {GRADE_OPTIONS.map((g) => (
                    <button
                      key={g}
                      onClick={() => setData((prev) => ({ ...prev, grade: g }))}
                      className={cn(
                        "rounded-lg border px-4 py-3 text-sm font-medium transition-colors",
                        data.grade === g
                          ? "border-[var(--primary)] bg-[var(--primary)] text-white"
                          : "border-[var(--border)] hover:border-[var(--primary)] hover:bg-[var(--muted)]"
                      )}
                    >
                      {g}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Subjects ───────────────────────────────────────────────── */}
          {step === "subjects" && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold">选择你希望学习的方向</h2>
                <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                  可多选，后续也可以随时更改
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {(availableSubjects.length > 0 ? availableSubjects : Object.keys(SUBJECT_META)).map((s) => {
                  const meta = SUBJECT_META[s] || { zh: s, icon: "BookOpen", desc: "" }
                  const selected = data.subjects.includes(s)
                  return (
                    <button
                      key={s}
                      onClick={() => toggleSubject(s)}
                      className={cn(
                        "flex items-start gap-3 rounded-lg border p-4 text-left transition-all",
                        selected
                          ? "border-[var(--primary)] bg-[var(--primary)]/10 ring-1 ring-[var(--primary)]"
                          : "border-[var(--border)] hover:border-[var(--primary)]/50 hover:bg-[var(--muted)]"
                      )}
                    >
                      <div
                        className={cn(
                          "mt-0.5 rounded-md p-2",
                          selected
                            ? "bg-[var(--primary)] text-white"
                            : "bg-[var(--muted)] text-[var(--muted-foreground)]"
                        )}
                      >
                        {subjectIcon(meta.icon)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{meta.zh}</span>
                          {selected && <Check className="h-4 w-4 text-[var(--primary)]" />}
                        </div>
                        <p className="mt-0.5 text-xs text-[var(--muted-foreground)]">
                          {meta.desc}
                        </p>
                      </div>
                    </button>
                  )
                })}
              </div>

              {/* Custom subject input */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={customSubject}
                  onChange={(e) => setCustomSubject(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && addCustomSubject()}
                  placeholder="+ 添加自定义方向"
                  className="flex-1 rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm placeholder:text-[var(--muted-foreground)]"
                />
                <Button variant="outline" size="sm" onClick={addCustomSubject}>
                  添加
                </Button>
              </div>

              {data.subjects.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {data.subjects.map((s) => (
                    <Badge key={s} variant="secondary" className="cursor-pointer" onClick={() => toggleSubject(s)}>
                      {SUBJECT_META[s]?.zh || s} ×
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── Skills ──────────────────────────────────────────────────── */}
          {step === "skills" && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold">你目前的掌握程度如何？</h2>
                <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                  请对你选择的方向进行自评
                </p>
              </div>

              {data.subjects.length === 0 && (
                <div className="rounded-lg border border-[var(--border)] bg-[var(--muted)] p-6 text-center text-sm text-[var(--muted-foreground)]">
                  你还没有选择学习方向，请返回上一步选择
                </div>
              )}

              {data.subjects.map((s) => {
                const meta = SUBJECT_META[s] || { zh: s, icon: "BookOpen" }
                const current = data.skillLevels[s]
                return (
                  <div key={s} className="rounded-lg border border-[var(--border)] p-4">
                    <div className="mb-3 flex items-center gap-2">
                      {subjectIcon(meta.icon)}
                      <span className="font-medium">{meta.zh}</span>
                    </div>
                    <div className="flex gap-2">
                      {SKILL_LEVELS.map((lv) => (
                        <button
                          key={lv.value}
                          onClick={() => setSkillLevel(s, lv.value)}
                          className={cn(
                            "flex-1 rounded-lg border px-3 py-3 text-sm transition-all",
                            current === lv.value
                              ? "border-[var(--primary)] bg-[var(--primary)] text-white"
                              : "border-[var(--border)] hover:border-[var(--primary)]/50"
                          )}
                        >
                          <div className="font-medium">{lv.label}</div>
                          <div className="mt-0.5 text-xs opacity-70">{lv.desc}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* ── Goals ───────────────────────────────────────────────────── */}
          {step === "goals" && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold">你的学习目标和偏好</h2>
                <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                  最后一步，告诉我们你的目标和学习偏好
                </p>
              </div>

              {/* Goals */}
              <div>
                <label className="mb-2 flex items-center gap-2 text-sm font-medium">
                  <Target className="h-4 w-4" /> 学习目标
                </label>
                {data.goals.map((g, i) => (
                  <div key={i} className="mb-2 flex gap-2">
                    <input
                      type="text"
                      value={g}
                      onChange={(e) => setGoal(i, e.target.value)}
                      placeholder={`目标 ${i + 1}，例如：准备408考研`}
                      className="flex-1 rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm placeholder:text-[var(--muted-foreground)]"
                    />
                    {data.goals.length > 1 && (
                      <Button variant="ghost" size="sm" onClick={() => removeGoal(i)}>
                        ×
                      </Button>
                    )}
                  </div>
                ))}
                {data.goals.length < 5 && (
                  <Button variant="outline" size="sm" onClick={addGoal}>
                    + 添加目标
                  </Button>
                )}
              </div>

              {/* Learning style */}
              <div>
                <label className="mb-3 flex items-center gap-2 text-sm font-medium">
                  <BookOpen className="h-4 w-4" /> 学习偏好
                </label>
                <div className="space-y-3">
                  {STYLE_QUESTIONS.map((sq) => (
                    <div key={sq.dim} className="flex items-center justify-between rounded-lg border border-[var(--border)] px-4 py-3">
                      <span className="text-sm">{sq.question}</span>
                      <div className="flex gap-1">
                        {STYLE_OPTIONS.map((opt) => (
                          <button
                            key={opt.value}
                            onClick={() => setStylePreference(sq.dim, opt.value)}
                            className={cn(
                              "rounded-md px-3 py-1 text-xs transition-all",
                              data.learningStyle[sq.dim] === opt.value
                                ? "bg-[var(--primary)] text-white"
                                : "bg-[var(--muted)] hover:bg-[var(--border)]"
                            )}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Dislikes */}
              <div>
                <label className="mb-2 block text-sm font-medium">
                  有什么你不喜欢的学习方式或主题？（可选）
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={dislikeInput}
                    onChange={(e) => setDislikeInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && addDislike()}
                    placeholder="例如：死记硬背、纯理论"
                    className="flex-1 rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm placeholder:text-[var(--muted-foreground)]"
                  />
                  <Button variant="outline" size="sm" onClick={addDislike}>
                    添加
                  </Button>
                </div>
                {data.dislikes.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {data.dislikes.map((d) => (
                      <Badge key={d} variant="outline" className="cursor-pointer" onClick={() => removeDislike(d)}>
                        {d} ×
                      </Badge>
                    ))}
                  </div>
                )}
              </div>

              {!canSubmit && (
                <p className="text-xs text-[var(--muted-foreground)]">
                  请至少选择一个学习方向和年级
                </p>
              )}
            </div>
          )}
        </div>

        {/* Navigation buttons */}
        <div className="mt-6 flex items-center justify-between">
          <div>
            {currentIndex > 0 && (
              <Button variant="outline" onClick={goBack}>
                <ArrowLeft className="mr-1 h-4 w-4" /> 上一步
              </Button>
            )}
          </div>
          <div className="flex gap-3">
            <Button
              variant="ghost"
              onClick={() => router.push("/")}
            >
              跳过，稍后再说
            </Button>
            {step === "goals" ? (
              <Button onClick={handleSubmit} disabled={!canSubmit || submitting}>
                {submitting ? "正在创建..." : (
                  <>
                    <Sparkles className="mr-1 h-4 w-4" /> 完成，开始学习
                  </>
                )}
              </Button>
            ) : (
              <Button onClick={goNext}>
                下一步 <ArrowRight className="ml-1 h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Page export with Suspense ────────────────────────────────────────────────

export default function OnboardingPage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-[var(--muted-foreground)]">加载中...</p>
      </div>
    }>
      <OnboardingPageInner />
    </Suspense>
  )
}
