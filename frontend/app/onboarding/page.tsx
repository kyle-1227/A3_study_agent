"use client"

import { Suspense, useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, ArrowRight, BookOpen, Check, Sparkles, Target } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  fetchLearningGuidanceCatalog,
  freezeOnboardingAttempt,
  getOrCreateOnboardingAttempt,
  OnboardingAttemptError,
  OnboardingClientError,
  submitOnboardingV2,
} from "@/lib/onboarding-client"
import {
  buildOnboardRequestV2,
  OnboardingContractError,
  parseOnboardRequestV2,
  type LearningGuidanceCatalogV1,
  type OnboardRequestV2,
  type OnboardingTopicInputV1,
  type PreferenceDimension,
} from "@/lib/onboarding-contracts"
import { requirePublicApiBaseUrl } from "@/lib/public-config"
import { cn } from "@/lib/utils"

const API_BASE_URL = requirePublicApiBaseUrl()

type Step = "welcome" | "topics" | "skills" | "goals"

interface TopicDraft {
  subject: string
  subjectTitle: string
  topic_id: string
  topicTitle: string
  level: number | null
  confidence: number | null
  goal: string
  importance: number | null
  progress: number | null
}

const GRADE_OPTIONS = ["大一", "大二", "大三", "大四", "研一", "研二", "研三", "高三", "其他"]
const SKILL_LEVEL_OPTIONS = [
  { value: 0.25, label: "入门" },
  { value: 0.5, label: "中等" },
  { value: 0.75, label: "熟练" },
  { value: 1, label: "精通" },
]
const CONFIDENCE_OPTIONS = [
  { value: 0.25, label: "不确定" },
  { value: 0.5, label: "一般" },
  { value: 0.75, label: "较确定" },
  { value: 1, label: "非常确定" },
]
const IMPORTANCE_OPTIONS = [
  { value: 0.25, label: "较低" },
  { value: 0.5, label: "一般" },
  { value: 0.75, label: "重要" },
  { value: 1, label: "最高" },
]
const PROGRESS_OPTIONS = [
  { value: 0, label: "未开始" },
  { value: 0.25, label: "刚开始" },
  { value: 0.5, label: "进行中" },
  { value: 0.75, label: "接近完成" },
  { value: 1, label: "已完成" },
]
const STYLE_OPTIONS = [
  { value: 0.2, label: "不太喜欢" },
  { value: 0.5, label: "一般" },
  { value: 0.8, label: "很喜欢" },
]
const STYLE_QUESTIONS: ReadonlyArray<{ dimension: PreferenceDimension; question: string }> = [
  { dimension: "prefer_examples", question: "你喜欢通过具体案例来学习吗？" },
  { dimension: "prefer_visual", question: "你喜欢图表或可视化吗？" },
  { dimension: "prefer_step_by_step", question: "你喜欢分步骤讲解吗？" },
  { dimension: "prefer_concise", question: "你喜欢简短直接的答案吗？" },
  { dimension: "prefer_theory", question: "你喜欢深入理解原理吗？" },
  { dimension: "prefer_practice", question: "你喜欢通过动手练习来学习吗？" },
  { dimension: "prefer_analogy", question: "你喜欢通过类比来理解吗？" },
]
const STEP_ORDER: Step[] = ["welcome", "topics", "skills", "goals"]
const STEP_LABELS: Record<Step, string> = {
  welcome: "基本信息",
  topics: "知识主题",
  skills: "掌握程度",
  goals: "目标与偏好",
}

function topicKey(subject: string, topicId: string): string {
  return `${subject}\u0000${topicId}`
}

function ChoiceGroup(props: {
  label: string
  value: number | null | undefined
  options: ReadonlyArray<{ value: number; label: string }>
  disabled: boolean
  onChange: (value: number) => void
  onClear?: () => void
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-sm font-medium">{props.label}</span>
        {props.onClear && props.value !== undefined && (
          <button
            type="button"
            className="text-xs text-[var(--muted-foreground)] underline-offset-2 hover:underline"
            disabled={props.disabled}
            onClick={props.onClear}
          >
            清除选择
          </button>
        )}
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {props.options.map((option) => (
          <button
            key={option.value}
            type="button"
            disabled={props.disabled}
            onClick={() => props.onChange(option.value)}
            className={cn(
              "rounded-lg border px-3 py-2 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60",
              props.value === option.value
                ? "border-[var(--primary)] bg-[var(--primary)] text-white"
                : "border-[var(--border)] hover:border-[var(--primary)]/60",
            )}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}

function hydrateTopicDrafts(
  request: OnboardRequestV2,
  catalog: LearningGuidanceCatalogV1,
): Record<string, TopicDraft> {
  const titles = new Map<string, { subjectTitle: string; topicTitle: string }>()
  for (const subject of catalog.subjects) {
    for (const topic of subject.topics) {
      titles.set(topicKey(subject.subject_id, topic.topic_id), {
        subjectTitle: subject.title,
        topicTitle: topic.title,
      })
    }
  }
  const goals = new Map(
    request.profile.goals.map((goal) => [topicKey(goal.subject, goal.topic_id), goal]),
  )
  return Object.fromEntries(
    request.profile.skills.map((skill) => {
      const key = topicKey(skill.subject, skill.topic_id)
      const title = titles.get(key)
      const goal = goals.get(key)
      if (!title || !goal) throw new OnboardingContractError("onboard_v2", "stored topic is invalid")
      return [
        key,
        {
          subject: skill.subject,
          subjectTitle: title.subjectTitle,
          topic_id: skill.topic_id,
          topicTitle: title.topicTitle,
          level: skill.level,
          confidence: skill.confidence,
          goal: goal.goal,
          importance: goal.importance,
          progress: goal.progress,
        },
      ]
    }),
  )
}

function hydratePreferences(
  request: OnboardRequestV2,
): Partial<Record<PreferenceDimension, number>> {
  const preferences: Partial<Record<PreferenceDimension, number>> = {}
  for (const item of request.profile.preferences) preferences[item.dimension] = item.strength
  return preferences
}

function errorMessage(error: unknown, operation: "catalog" | "submit" | "attempt"): string {
  if (error instanceof OnboardingClientError) {
    if (error.code === "onboarding_request_aborted") return "请求超时，请稍后重试。"
    if (error.code === "onboarding_http_failed") {
      if (error.status === 409) return "该用户的入门资料已存在且与本次提交冲突。"
      if (error.status === 422) return "提交内容未通过服务端严格校验，请检查所选主题。"
      if (error.status === 503) return "学习主题服务暂不可用，当前不能继续提交。"
      return `服务请求失败（HTTP ${error.status ?? "unknown"}）。`
    }
    if (error.code === "onboarding_transport_failed") return "无法连接学习主题服务。"
    if (error.code === "onboarding_client_configuration_invalid") return "前端 API 地址配置无效。"
    return "服务返回了不符合契约的数据。"
  }
  if (error instanceof OnboardingAttemptError) {
    if (error.code === "onboarding_attempt_payload_conflict") {
      return "本次提交已冻结；重试必须保持原始内容不变。"
    }
    return "本地幂等提交记录无效，当前不能安全提交。"
  }
  if (error instanceof OnboardingContractError) {
    if (error.contract === "onboard_result_v2") return "服务返回了不符合契约的数据。"
    return operation === "catalog"
      ? "学习主题目录不符合严格契约。"
      : "请完整填写每个主题的掌握程度、信心、目标、重要性和进度。"
  }
  return operation === "catalog" ? "学习主题目录加载失败。" : "提交失败，请稍后重试。"
}

function OnboardingPageInner() {
  const router = useRouter()
  const [step, setStep] = useState<Step>("welcome")
  const [catalog, setCatalog] = useState<LearningGuidanceCatalogV1 | null>(null)
  const [catalogLoading, setCatalogLoading] = useState(true)
  const [resolvedUserId, setResolvedUserId] = useState<string | null>(null)
  const [requestId, setRequestId] = useState<string | null>(null)
  const [frozenRequest, setFrozenRequest] = useState<OnboardRequestV2 | null>(null)
  const [grade, setGrade] = useState("")
  const [nickname, setNickname] = useState("")
  const [topicDrafts, setTopicDrafts] = useState<Record<string, TopicDraft>>({})
  const [preferences, setPreferences] = useState<
    Partial<Record<PreferenceDimension, number>>
  >({})
  const [dislikes, setDislikes] = useState<string[]>([])
  const [dislikeInput, setDislikeInput] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    try {
      const existing = localStorage.getItem("a3_user_id")
      const userId = existing ?? `u_${crypto.randomUUID()}`
      if (existing === null) localStorage.setItem("a3_user_id", userId)
      setNickname(localStorage.getItem("a3_nickname") ?? "")
      setResolvedUserId(userId)
    } catch {
      setError("无法建立本地用户身份，当前不能安全提交。")
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    let active = true
    setCatalogLoading(true)
    fetchLearningGuidanceCatalog({
      apiBaseUrl: API_BASE_URL,
      fetchImpl: fetch,
      signal: controller.signal,
    })
      .then((value) => {
        if (active) setCatalog(value)
      })
      .catch((caught: unknown) => {
        if (active && !controller.signal.aborted) setError(errorMessage(caught, "catalog"))
      })
      .finally(() => {
        if (active) setCatalogLoading(false)
      })
    return () => {
      active = false
      controller.abort()
    }
  }, [])

  useEffect(() => {
    if (!resolvedUserId || !catalog) return
    try {
      const attempt = getOrCreateOnboardingAttempt({
        storage: localStorage,
        userId: resolvedUserId,
        uuidFactory: () => crypto.randomUUID(),
      })
      setRequestId(attempt.request_id)
      if (attempt.payload !== null) {
        const request = parseOnboardRequestV2(attempt.payload, catalog)
        setFrozenRequest(request)
        setGrade(request.grade)
        setNickname(request.nickname)
        setDislikes([...request.dislikes])
        setTopicDrafts(hydrateTopicDrafts(request, catalog))
        setPreferences(hydratePreferences(request))
        setStep("goals")
      }
    } catch (caught: unknown) {
      setError(errorMessage(caught, "attempt"))
    }
  }, [catalog, resolvedUserId])

  const selectedTopics = useMemo(() => {
    if (!catalog) return []
    return catalog.subjects.flatMap((subject) =>
      subject.topics.flatMap((topic) => {
        const draft = topicDrafts[topicKey(subject.subject_id, topic.topic_id)]
        return draft ? [draft] : []
      }),
    )
  }, [catalog, topicDrafts])

  const locked = frozenRequest !== null
  const currentIndex = STEP_ORDER.indexOf(step)
  const skillsComplete =
    selectedTopics.length > 0 &&
    selectedTopics.every((topic) => topic.level !== null && topic.confidence !== null)
  const goalsComplete =
    selectedTopics.length > 0 &&
    selectedTopics.every(
      (topic) =>
        topic.goal.length > 0 &&
        topic.goal === topic.goal.trim() &&
        topic.importance !== null &&
        topic.progress !== null,
    )
  const canAdvance =
    (step === "welcome" && grade.length > 0 && grade === grade.trim()) ||
    (step === "topics" && selectedTopics.length > 0) ||
    (step === "skills" && skillsComplete)
  const canSubmit =
    Boolean(catalog && resolvedUserId && requestId) &&
    grade.length > 0 &&
    grade === grade.trim() &&
    skillsComplete &&
    goalsComplete

  const toggleTopic = (
    subject: LearningGuidanceCatalogV1["subjects"][number],
    topic: LearningGuidanceCatalogV1["subjects"][number]["topics"][number],
  ) => {
    if (locked) return
    const key = topicKey(subject.subject_id, topic.topic_id)
    setTopicDrafts((current) => {
      if (current[key]) {
        const next = { ...current }
        delete next[key]
        return next
      }
      return {
        ...current,
        [key]: {
          subject: subject.subject_id,
          subjectTitle: subject.title,
          topic_id: topic.topic_id,
          topicTitle: topic.title,
          level: null,
          confidence: null,
          goal: "",
          importance: null,
          progress: null,
        },
      }
    })
  }

  const updateTopic = (key: string, patch: Partial<TopicDraft>) => {
    if (locked) return
    setTopicDrafts((current) => {
      const existing = current[key]
      return existing ? { ...current, [key]: { ...existing, ...patch } } : current
    })
  }

  const addDislike = () => {
    if (locked) return
    const value = dislikeInput.trim()
    if (!value || dislikes.includes(value) || dislikes.length >= 50) return
    setDislikes((current) => [...current, value])
    setDislikeInput("")
  }

  const handleSubmit = async () => {
    if (!catalog || !resolvedUserId || !requestId || !canSubmit) {
      setError("请先完整填写所有必填信息。")
      return
    }
    setSubmitting(true)
    setError("")
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), 30_000)
    try {
      let request = frozenRequest
      if (request === null) {
        const topics: OnboardingTopicInputV1[] = selectedTopics.map((topic) => ({
          subject: topic.subject,
          topic_id: topic.topic_id,
          level: topic.level as number,
          confidence: topic.confidence as number,
          goal: topic.goal,
          importance: topic.importance as number,
          progress: topic.progress as number,
        }))
        request = buildOnboardRequestV2(
          {
            requestId,
            userId: resolvedUserId,
            nickname,
            grade,
            dislikes,
            topics,
            preferences,
          },
          catalog,
        )
        request = freezeOnboardingAttempt({
          storage: localStorage,
          userId: resolvedUserId,
          catalog,
          request,
        })
        setFrozenRequest(request)
      }
      const result = await submitOnboardingV2({
        apiBaseUrl: API_BASE_URL,
        catalog,
        request,
        fetchImpl: fetch,
        signal: controller.signal,
      })
      if (result.status !== "created" && result.status !== "replayed") {
        throw new OnboardingContractError("onboard_result_v2", "terminal status is invalid")
      }
      localStorage.setItem("a3_onboarding_completed", "true")
      router.replace("/")
    } catch (caught: unknown) {
      setError(errorMessage(caught, "submit"))
    } finally {
      window.clearTimeout(timeout)
      setSubmitting(false)
    }
  }

  if (!resolvedUserId) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-[var(--muted-foreground)]">正在建立用户身份...</p>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--background)] p-4">
      <div className="w-full max-w-3xl">
        <div className="mb-8">
          <div className="mb-2 flex items-center justify-between text-sm text-[var(--muted-foreground)]">
            <span>
              步骤 {currentIndex + 1}/{STEP_ORDER.length}：{STEP_LABELS[step]}
            </span>
            <span>{Math.round(((currentIndex + 1) / STEP_ORDER.length) * 100)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-[var(--border)]">
            <div
              className="h-full rounded-full bg-[var(--primary)] transition-all duration-300"
              style={{ width: `${((currentIndex + 1) / STEP_ORDER.length) * 100}%` }}
            />
          </div>
        </div>

        {error && (
          <div role="alert" className="mb-4 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-700 dark:bg-red-950 dark:text-red-300">
            {error}
          </div>
        )}
        {locked && (
          <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
            本次提交内容已冻结。再次提交会使用同一 request_id 和完全相同的内容安全重试。
          </div>
        )}

        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-sm">
          {step === "welcome" && (
            <div className="space-y-6">
              <div className="text-center">
                <Sparkles className="mx-auto mb-3 h-10 w-10 text-[var(--primary)]" />
                <h1 className="text-2xl font-bold">建立你的学习主题画像</h1>
                <p className="mt-2 text-[var(--muted-foreground)]">
                  所有主题身份都来自当前生产知识图谱，不会使用本地候选替代。
                </p>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">你目前所在的年级</label>
                <div className="grid grid-cols-3 gap-2">
                  {GRADE_OPTIONS.map((option) => (
                    <button
                      key={option}
                      type="button"
                      disabled={locked}
                      onClick={() => setGrade(option)}
                      className={cn(
                        "rounded-lg border px-4 py-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                        grade === option
                          ? "border-[var(--primary)] bg-[var(--primary)] text-white"
                          : "border-[var(--border)] hover:border-[var(--primary)]",
                      )}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {step === "topics" && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold">选择具体知识主题</h2>
                <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                  每个主题都需要单独填写掌握程度、目标和进度。
                </p>
              </div>
              {catalogLoading && <p className="text-sm text-[var(--muted-foreground)]">正在加载严格主题目录...</p>}
              {!catalogLoading && catalog &&
                catalog.subjects.map((subject) => (
                  <section key={subject.subject_id} className="rounded-lg border border-[var(--border)] p-4">
                    <h3 className="mb-3 font-semibold">{subject.title}</h3>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {subject.topics.map((topic) => {
                        const key = topicKey(subject.subject_id, topic.topic_id)
                        const selected = Boolean(topicDrafts[key])
                        return (
                          <button
                            key={topic.topic_id}
                            type="button"
                            disabled={locked}
                            onClick={() => toggleTopic(subject, topic)}
                            className={cn(
                              "flex items-center justify-between rounded-lg border px-3 py-3 text-left text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                              selected
                                ? "border-[var(--primary)] bg-[var(--primary)]/10"
                                : "border-[var(--border)] hover:border-[var(--primary)]/60",
                            )}
                          >
                            <span>{topic.title}</span>
                            {selected && <Check className="h-4 w-4 text-[var(--primary)]" />}
                          </button>
                        )
                      })}
                    </div>
                  </section>
                ))}
              {selectedTopics.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {selectedTopics.map((topic) => (
                    <Badge key={topic.topic_id} variant="secondary">
                      {topic.subjectTitle} / {topic.topicTitle}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          )}

          {step === "skills" && (
            <div className="space-y-5">
              <div>
                <h2 className="text-xl font-bold">逐主题填写掌握程度</h2>
                <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                  掌握水平与自评信心都必须由你明确选择。
                </p>
              </div>
              {selectedTopics.map((topic) => {
                const key = topicKey(topic.subject, topic.topic_id)
                return (
                  <section key={key} className="space-y-4 rounded-lg border border-[var(--border)] p-4">
                    <h3 className="font-semibold">{topic.subjectTitle} / {topic.topicTitle}</h3>
                    <ChoiceGroup
                      label="当前掌握水平"
                      value={topic.level}
                      options={SKILL_LEVEL_OPTIONS}
                      disabled={locked}
                      onChange={(value) => updateTopic(key, { level: value })}
                    />
                    <ChoiceGroup
                      label="对这次自评的信心"
                      value={topic.confidence}
                      options={CONFIDENCE_OPTIONS}
                      disabled={locked}
                      onChange={(value) => updateTopic(key, { confidence: value })}
                    />
                  </section>
                )
              })}
            </div>
          )}

          {step === "goals" && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-bold">逐主题填写目标与偏好</h2>
                <p className="mt-1 text-sm text-[var(--muted-foreground)]">
                  目标、重要性与当前进度不会由前端替你补默认值。
                </p>
              </div>
              {selectedTopics.map((topic) => {
                const key = topicKey(topic.subject, topic.topic_id)
                return (
                  <section key={key} className="space-y-4 rounded-lg border border-[var(--border)] p-4">
                    <label className="block">
                      <span className="mb-2 flex items-center gap-2 text-sm font-medium">
                        <Target className="h-4 w-4" /> {topic.subjectTitle} / {topic.topicTitle} 的目标
                      </span>
                      <input
                        type="text"
                        value={topic.goal}
                        disabled={locked}
                        maxLength={500}
                        onChange={(event) => updateTopic(key, { goal: event.target.value })}
                        placeholder="输入不带首尾空格的具体目标"
                        className="w-full rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-60"
                      />
                    </label>
                    <ChoiceGroup
                      label="目标重要性"
                      value={topic.importance}
                      options={IMPORTANCE_OPTIONS}
                      disabled={locked}
                      onChange={(value) => updateTopic(key, { importance: value })}
                    />
                    <ChoiceGroup
                      label="当前进度"
                      value={topic.progress}
                      options={PROGRESS_OPTIONS}
                      disabled={locked}
                      onChange={(value) => updateTopic(key, { progress: value })}
                    />
                  </section>
                )
              })}

              <section>
                <div className="mb-3 flex items-center gap-2 text-sm font-medium">
                  <BookOpen className="h-4 w-4" /> 学习偏好（可选）
                </div>
                <p className="mb-3 text-xs text-[var(--muted-foreground)]">
                  只有明确选择的维度才会写入，并会以相同强度绑定到每个已选主题。
                </p>
                <div className="space-y-4">
                  {STYLE_QUESTIONS.map(({ dimension, question }) => (
                    <ChoiceGroup
                      key={dimension}
                      label={question}
                      value={preferences[dimension]}
                      options={STYLE_OPTIONS}
                      disabled={locked}
                      onChange={(value) =>
                        setPreferences((current) => ({ ...current, [dimension]: value }))
                      }
                      onClear={() =>
                        setPreferences((current) => {
                          const next = { ...current }
                          delete next[dimension]
                          return next
                        })
                      }
                    />
                  ))}
                </div>
              </section>

              <section>
                <label className="mb-2 block text-sm font-medium">不喜欢的学习方式或主题（可选）</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={dislikeInput}
                    disabled={locked}
                    maxLength={500}
                    onChange={(event) => setDislikeInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault()
                        addDislike()
                      }
                    }}
                    placeholder="例如：死记硬背"
                    className="flex-1 rounded-lg border border-[var(--border)] bg-transparent px-3 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-60"
                  />
                  <Button type="button" variant="outline" size="sm" disabled={locked} onClick={addDislike}>
                    添加
                  </Button>
                </div>
                {dislikes.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {dislikes.map((item) => (
                      <Badge
                        key={item}
                        variant="outline"
                        className={cn(!locked && "cursor-pointer")}
                        onClick={() => {
                          if (!locked) setDislikes((current) => current.filter((value) => value !== item))
                        }}
                      >
                        {item} {!locked && "×"}
                      </Badge>
                    ))}
                  </div>
                )}
              </section>
            </div>
          )}
        </div>

        <div className="mt-6 flex items-center justify-between">
          <div>
            {currentIndex > 0 && (
              <Button type="button" variant="outline" onClick={() => setStep(STEP_ORDER[currentIndex - 1])}>
                <ArrowLeft className="mr-1 h-4 w-4" /> 上一步
              </Button>
            )}
          </div>
          {step === "goals" ? (
            <Button type="button" disabled={!canSubmit || submitting} onClick={handleSubmit}>
              <Sparkles className="mr-1 h-4 w-4" />
              {submitting ? "正在提交..." : locked ? "安全重试" : "完成并开始学习"}
            </Button>
          ) : (
            <Button
              type="button"
              disabled={!canAdvance || catalogLoading || !catalog}
              onClick={() => setStep(STEP_ORDER[currentIndex + 1])}
            >
              下一步 <ArrowRight className="ml-1 h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

export default function OnboardingPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center">
          <p className="text-[var(--muted-foreground)]">加载中...</p>
        </div>
      }
    >
      <OnboardingPageInner />
    </Suspense>
  )
}
