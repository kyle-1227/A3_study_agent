export const LEARNING_GUIDANCE_CATALOG_SCHEMA_VERSION =
  "learning_guidance_catalog_v1" as const
export const PROFILE_WRITE_REQUEST_SCHEMA_VERSION =
  "learning_guidance_profile_write_request_v1" as const
export const ONBOARD_REQUEST_SCHEMA_VERSION = "onboard_v2" as const
export const ONBOARD_RESULT_SCHEMA_VERSION = "onboard_result_v2" as const

export const PREFERENCE_DIMENSIONS = [
  "prefer_examples",
  "prefer_visual",
  "prefer_step_by_step",
  "prefer_concise",
  "prefer_theory",
  "prefer_practice",
  "prefer_analogy",
] as const

export type PreferenceDimension = (typeof PREFERENCE_DIMENSIONS)[number]

export interface LearningGuidanceCatalogTopicV1 {
  readonly topic_id: string
  readonly title: string
}

export interface LearningGuidanceCatalogSubjectV1 {
  readonly subject_id: string
  readonly title: string
  readonly topics: readonly LearningGuidanceCatalogTopicV1[]
}

export interface LearningGuidanceCatalogV1 {
  readonly schema_version: typeof LEARNING_GUIDANCE_CATALOG_SCHEMA_VERSION
  readonly data_version: string
  readonly artifact_fingerprint: string
  readonly subjects: readonly LearningGuidanceCatalogSubjectV1[]
}

export interface ProfileSkillWriteV1 {
  readonly subject: string
  readonly topic_id: string
  readonly level: number
  readonly confidence: number
}

export interface ProfileGoalWriteV1 {
  readonly subject: string
  readonly topic_id: string
  readonly goal: string
  readonly importance: number
  readonly progress: number
}

export interface ProfilePreferenceWriteV1 {
  readonly subject: string
  readonly topic_id: string
  readonly dimension: PreferenceDimension
  readonly strength: number
}

export interface LearningGuidanceProfileWriteRequestV1 {
  readonly schema_version: typeof PROFILE_WRITE_REQUEST_SCHEMA_VERSION
  readonly request_id: string
  readonly user_id: string
  readonly skills: readonly ProfileSkillWriteV1[]
  readonly goals: readonly ProfileGoalWriteV1[]
  readonly preferences: readonly ProfilePreferenceWriteV1[]
}

export interface OnboardRequestV2 {
  readonly schema_version: typeof ONBOARD_REQUEST_SCHEMA_VERSION
  readonly profile: LearningGuidanceProfileWriteRequestV1
  readonly nickname: string
  readonly grade: string
  readonly dislikes: readonly string[]
}

export interface OnboardResultV2 {
  readonly schema_version: typeof ONBOARD_RESULT_SCHEMA_VERSION
  readonly status: "created" | "replayed"
  readonly request_id: string
  readonly user_id: string
  readonly summary: string
  readonly skills_count: number
  readonly goals_count: number
  readonly preferences_count: number
}

export interface OnboardingTopicInputV1 {
  readonly subject: string
  readonly topic_id: string
  readonly level: number
  readonly confidence: number
  readonly goal: string
  readonly importance: number
  readonly progress: number
}

export interface OnboardingFormInputV1 {
  readonly requestId: string
  readonly userId: string
  readonly nickname: string
  readonly grade: string
  readonly dislikes: readonly string[]
  readonly topics: readonly OnboardingTopicInputV1[]
  readonly preferences: Readonly<Partial<Record<PreferenceDimension, number>>>
}

export interface ExpectedOnboardResultV2 {
  readonly requestId: string
  readonly userId: string
  readonly skillsCount: number
  readonly goalsCount: number
  readonly preferencesCount: number
}

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const IDENTITY_PATTERN = /^[A-Za-z0-9][A-Za-z0-9:._/-]{0,159}$/
const KNOWLEDGE_ID_PATTERN = /^[a-z0-9][a-z0-9._:-]{0,199}$/
const SHA256_PATTERN = /^[0-9a-f]{64}$/
const PREFERENCE_DIMENSION_SET = new Set<string>(PREFERENCE_DIMENSIONS)

export class OnboardingContractError extends Error {
  readonly contract: string

  constructor(contract: string, reason: string) {
    super(`${contract}: ${reason}`)
    this.name = "OnboardingContractError"
    this.contract = contract
  }
}

export function parseLearningGuidanceCatalogV1(value: unknown): LearningGuidanceCatalogV1 {
  const contract = LEARNING_GUIDANCE_CATALOG_SCHEMA_VERSION
  const data = record(value, contract)
  exactKeys(
    data,
    ["schema_version", "data_version", "artifact_fingerprint", "subjects"],
    contract,
  )
  if (data.schema_version !== contract) {
    fail(contract, `schema_version must equal ${contract}`)
  }
  const subjectsWire = boundedArray(data.subjects, "subjects", 1, 200, contract)
  const subjectIds = new Set<string>()
  const topicIds = new Set<string>()
  const subjects = subjectsWire.map((subjectValue, subjectIndex) => {
    const subjectContract = `${contract}.subjects[${subjectIndex}]`
    const subject = record(subjectValue, subjectContract)
    exactKeys(subject, ["subject_id", "title", "topics"], subjectContract)
    const subjectId = knowledgeId(subject.subject_id, "subject_id", 120, subjectContract)
    if (subjectIds.has(subjectId)) fail(contract, "subject_id values must be unique")
    subjectIds.add(subjectId)
    const topicsWire = boundedArray(subject.topics, "topics", 1, 500, subjectContract)
    const topics = topicsWire.map((topicValue, topicIndex) => {
      const topicContract = `${subjectContract}.topics[${topicIndex}]`
      const topic = record(topicValue, topicContract)
      exactKeys(topic, ["topic_id", "title"], topicContract)
      const topicId = knowledgeId(topic.topic_id, "topic_id", 160, topicContract)
      if (topicIds.has(topicId)) fail(contract, "topic_id values must be globally unique")
      topicIds.add(topicId)
      return {
        topic_id: topicId,
        title: normalizedString(topic.title, "title", 1, 240, topicContract),
      }
    })
    return {
      subject_id: subjectId,
      title: normalizedString(subject.title, "title", 1, 240, subjectContract),
      topics,
    }
  })

  const fingerprint = normalizedString(
    data.artifact_fingerprint,
    "artifact_fingerprint",
    64,
    64,
    contract,
  )
  if (!SHA256_PATTERN.test(fingerprint)) {
    fail(contract, "artifact_fingerprint must be lowercase SHA-256 text")
  }
  return deepFreeze({
    schema_version: LEARNING_GUIDANCE_CATALOG_SCHEMA_VERSION,
    data_version: normalizedString(data.data_version, "data_version", 1, 160, contract),
    artifact_fingerprint: fingerprint,
    subjects,
  })
}

export function parseOnboardRequestV2(
  value: unknown,
  catalog?: LearningGuidanceCatalogV1,
): OnboardRequestV2 {
  const contract = ONBOARD_REQUEST_SCHEMA_VERSION
  const data = record(value, contract)
  exactKeys(data, ["schema_version", "profile", "nickname", "grade", "dislikes"], contract)
  if (data.schema_version !== contract) {
    fail(contract, `schema_version must equal ${contract}`)
  }
  const profile = parseProfileWriteRequest(data.profile, catalog)
  const dislikes = boundedArray(data.dislikes, "dislikes", 0, 50, contract).map(
    (item, index) => normalizedString(item, `dislikes[${index}]`, 1, 500, contract),
  )
  if (new Set(dislikes).size !== dislikes.length) {
    fail(contract, "dislikes must be unique")
  }
  return deepFreeze({
    schema_version: ONBOARD_REQUEST_SCHEMA_VERSION,
    profile,
    nickname: normalizedString(data.nickname, "nickname", 0, 120, contract),
    grade: normalizedString(data.grade, "grade", 1, 120, contract),
    dislikes,
  })
}

export function buildOnboardRequestV2(
  input: OnboardingFormInputV1,
  catalogValue: LearningGuidanceCatalogV1,
): OnboardRequestV2 {
  const contract = "onboard_v2_builder"
  const catalog = parseLearningGuidanceCatalogV1(catalogValue)
  const inputRecord = record(input, contract)
  exactKeys(
    inputRecord,
    ["requestId", "userId", "nickname", "grade", "dislikes", "topics", "preferences"],
    contract,
  )
  const requestId = canonicalUuid(inputRecord.requestId, "requestId", contract)
  const userId = identity(inputRecord.userId, "userId", contract)
  const topicInputs = boundedArray(inputRecord.topics, "topics", 1, 200, contract).map(
    (topicValue, index) => parseTopicInput(topicValue, index, contract),
  )
  const topicInputBySlot = new Map<string, OnboardingTopicInputV1>()
  for (const item of topicInputs) {
    const key = slotKey(item.subject, item.topic_id)
    if (topicInputBySlot.has(key)) fail(contract, "topics must use unique topic slots")
    topicInputBySlot.set(key, item)
  }

  const orderedTopics: OnboardingTopicInputV1[] = []
  for (const subject of catalog.subjects) {
    for (const topic of subject.topics) {
      const selected = topicInputBySlot.get(slotKey(subject.subject_id, topic.topic_id))
      if (selected) orderedTopics.push(selected)
    }
  }
  if (orderedTopics.length !== topicInputs.length) {
    fail(contract, "every topic must reference the supplied catalog")
  }

  const preferenceRecord = record(inputRecord.preferences, `${contract}.preferences`)
  for (const dimension of Object.keys(preferenceRecord)) {
    if (!PREFERENCE_DIMENSION_SET.has(dimension)) {
      fail(contract, `unsupported preference dimension: ${dimension}`)
    }
  }
  const selectedPreferences = PREFERENCE_DIMENSIONS.flatMap((dimension) => {
    if (!Object.prototype.hasOwnProperty.call(preferenceRecord, dimension)) return []
    const strength = unitNumber(
      preferenceRecord[dimension],
      `preferences.${dimension}`,
      contract,
    )
    return orderedTopics.map((topic) => ({
      subject: topic.subject,
      topic_id: topic.topic_id,
      dimension,
      strength,
    }))
  })

  return parseOnboardRequestV2(
    {
      schema_version: ONBOARD_REQUEST_SCHEMA_VERSION,
      profile: {
        schema_version: PROFILE_WRITE_REQUEST_SCHEMA_VERSION,
        request_id: requestId,
        user_id: userId,
        skills: orderedTopics.map((topic) => ({
          subject: topic.subject,
          topic_id: topic.topic_id,
          level: topic.level,
          confidence: topic.confidence,
        })),
        goals: orderedTopics.map((topic) => ({
          subject: topic.subject,
          topic_id: topic.topic_id,
          goal: topic.goal,
          importance: topic.importance,
          progress: topic.progress,
        })),
        preferences: selectedPreferences,
      },
      nickname: inputRecord.nickname,
      grade: inputRecord.grade,
      dislikes: inputRecord.dislikes,
    },
    catalog,
  )
}

export function parseOnboardResultV2(
  value: unknown,
  expected?: ExpectedOnboardResultV2,
): OnboardResultV2 {
  const contract = ONBOARD_RESULT_SCHEMA_VERSION
  const data = record(value, contract)
  exactKeys(
    data,
    [
      "schema_version",
      "status",
      "request_id",
      "user_id",
      "summary",
      "skills_count",
      "goals_count",
      "preferences_count",
    ],
    contract,
  )
  if (data.schema_version !== contract) {
    fail(contract, `schema_version must equal ${contract}`)
  }
  if (data.status !== "created" && data.status !== "replayed") {
    fail(contract, "status must be created or replayed")
  }
  const result: OnboardResultV2 = {
    schema_version: ONBOARD_RESULT_SCHEMA_VERSION,
    status: data.status,
    request_id: canonicalUuid(data.request_id, "request_id", contract),
    user_id: identity(data.user_id, "user_id", contract),
    summary: normalizedString(data.summary, "summary", 1, 65_536, contract),
    skills_count: boundedInteger(data.skills_count, "skills_count", 1, 200, contract),
    goals_count: boundedInteger(data.goals_count, "goals_count", 1, 50, contract),
    preferences_count: boundedInteger(
      data.preferences_count,
      "preferences_count",
      0,
      200,
      contract,
    ),
  }
  if (
    expected &&
    (result.request_id !== expected.requestId ||
      result.user_id !== expected.userId ||
      result.skills_count !== expected.skillsCount ||
      result.goals_count !== expected.goalsCount ||
      result.preferences_count !== expected.preferencesCount)
  ) {
    fail(contract, "response identity or inventory does not match the submitted request")
  }
  return deepFreeze(result)
}

function parseProfileWriteRequest(
  value: unknown,
  catalogValue?: LearningGuidanceCatalogV1,
): LearningGuidanceProfileWriteRequestV1 {
  const contract = PROFILE_WRITE_REQUEST_SCHEMA_VERSION
  const data = record(value, contract)
  exactKeys(
    data,
    ["schema_version", "request_id", "user_id", "skills", "goals", "preferences"],
    contract,
  )
  if (data.schema_version !== contract) {
    fail(contract, `schema_version must equal ${contract}`)
  }
  const skills = boundedArray(data.skills, "skills", 1, 200, contract).map(
    (value, index) => parseSkill(value, index, contract),
  )
  const goals = boundedArray(data.goals, "goals", 1, 50, contract).map(
    (value, index) => parseGoal(value, index, contract),
  )
  const preferences = boundedArray(data.preferences, "preferences", 0, 200, contract).map(
    (value, index) => parsePreference(value, index, contract),
  )
  const skillSlots = skills.map((item) => slotKey(item.subject, item.topic_id))
  const goalSlots = goals.map((item) => slotKey(item.subject, item.topic_id))
  if (new Set(skillSlots).size !== skillSlots.length) {
    fail(contract, "skills must use unique topic slots")
  }
  if (new Set(goalSlots).size !== goalSlots.length) {
    fail(contract, "goals must contain exactly one goal per topic slot")
  }
  if (!sameSet(new Set(skillSlots), new Set(goalSlots))) {
    fail(contract, "skills and goals must cover the same topic slots")
  }
  if (new Set(goals.map((item) => item.goal)).size !== goals.length) {
    fail(contract, "goal text must be globally unique")
  }

  const preferenceSlots = new Set<string>()
  const preferenceCoverage = new Map<PreferenceDimension, Set<string>>()
  const preferenceStrengths = new Map<PreferenceDimension, number>()
  for (const item of preferences) {
    const topicSlot = slotKey(item.subject, item.topic_id)
    if (!skillSlots.includes(topicSlot)) {
      fail(contract, "preferences must reference a selected topic slot")
    }
    const logicalSlot = `${topicSlot}\u0000${item.dimension}`
    if (preferenceSlots.has(logicalSlot)) {
      fail(contract, "preferences must use unique topic and dimension slots")
    }
    preferenceSlots.add(logicalSlot)
    const strength = preferenceStrengths.get(item.dimension)
    if (strength !== undefined && strength !== item.strength) {
      fail(contract, "one preference dimension must use one strength")
    }
    preferenceStrengths.set(item.dimension, item.strength)
    const coverage = preferenceCoverage.get(item.dimension) ?? new Set<string>()
    coverage.add(topicSlot)
    preferenceCoverage.set(item.dimension, coverage)
  }
  const expectedSlots = new Set(skillSlots)
  for (const coverage of preferenceCoverage.values()) {
    if (!sameSet(coverage, expectedSlots)) {
      fail(contract, "each selected preference dimension must cover every topic")
    }
  }

  if (catalogValue) {
    const catalog = parseLearningGuidanceCatalogV1(catalogValue)
    const accepted = new Set(
      catalog.subjects.flatMap((subject) =>
        subject.topics.map((topic) => slotKey(subject.subject_id, topic.topic_id)),
      ),
    )
    if (skillSlots.some((slot) => !accepted.has(slot))) {
      fail(contract, "topic slots must reference the supplied catalog")
    }
  }
  return deepFreeze({
    schema_version: PROFILE_WRITE_REQUEST_SCHEMA_VERSION,
    request_id: canonicalUuid(data.request_id, "request_id", contract),
    user_id: identity(data.user_id, "user_id", contract),
    skills,
    goals,
    preferences,
  })
}

function parseSkill(value: unknown, index: number, contract: string): ProfileSkillWriteV1 {
  const itemContract = `${contract}.skills[${index}]`
  const data = record(value, itemContract)
  exactKeys(data, ["subject", "topic_id", "level", "confidence"], itemContract)
  return {
    subject: knowledgeId(data.subject, "subject", 120, itemContract),
    topic_id: knowledgeId(data.topic_id, "topic_id", 160, itemContract),
    level: unitNumber(data.level, "level", itemContract),
    confidence: unitNumber(data.confidence, "confidence", itemContract),
  }
}

function parseGoal(value: unknown, index: number, contract: string): ProfileGoalWriteV1 {
  const itemContract = `${contract}.goals[${index}]`
  const data = record(value, itemContract)
  exactKeys(data, ["subject", "topic_id", "goal", "importance", "progress"], itemContract)
  return {
    subject: knowledgeId(data.subject, "subject", 120, itemContract),
    topic_id: knowledgeId(data.topic_id, "topic_id", 160, itemContract),
    goal: normalizedString(data.goal, "goal", 1, 500, itemContract),
    importance: unitNumber(data.importance, "importance", itemContract),
    progress: unitNumber(data.progress, "progress", itemContract),
  }
}

function parsePreference(
  value: unknown,
  index: number,
  contract: string,
): ProfilePreferenceWriteV1 {
  const itemContract = `${contract}.preferences[${index}]`
  const data = record(value, itemContract)
  exactKeys(data, ["subject", "topic_id", "dimension", "strength"], itemContract)
  if (typeof data.dimension !== "string" || !PREFERENCE_DIMENSION_SET.has(data.dimension)) {
    fail(itemContract, "dimension is invalid")
  }
  return {
    subject: knowledgeId(data.subject, "subject", 120, itemContract),
    topic_id: knowledgeId(data.topic_id, "topic_id", 160, itemContract),
    dimension: data.dimension as PreferenceDimension,
    strength: unitNumber(data.strength, "strength", itemContract),
  }
}

function parseTopicInput(
  value: unknown,
  index: number,
  contract: string,
): OnboardingTopicInputV1 {
  const itemContract = `${contract}.topics[${index}]`
  const data = record(value, itemContract)
  exactKeys(
    data,
    ["subject", "topic_id", "level", "confidence", "goal", "importance", "progress"],
    itemContract,
  )
  return {
    subject: knowledgeId(data.subject, "subject", 120, itemContract),
    topic_id: knowledgeId(data.topic_id, "topic_id", 160, itemContract),
    level: unitNumber(data.level, "level", itemContract),
    confidence: unitNumber(data.confidence, "confidence", itemContract),
    goal: normalizedString(data.goal, "goal", 1, 500, itemContract),
    importance: unitNumber(data.importance, "importance", itemContract),
    progress: unitNumber(data.progress, "progress", itemContract),
  }
}

function record(value: unknown, contract: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(contract, "value must be an object")
  }
  return value as Record<string, unknown>
}

function exactKeys(data: Record<string, unknown>, fields: readonly string[], contract: string): void {
  const actual = Object.keys(data).sort()
  const expected = [...fields].sort()
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    fail(contract, "field inventory is invalid")
  }
}

function boundedArray(
  value: unknown,
  field: string,
  minimum: number,
  maximum: number,
  contract: string,
): unknown[] {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) {
    fail(contract, `${field} must be an array with ${minimum}..${maximum} items`)
  }
  return value
}

function normalizedString(
  value: unknown,
  field: string,
  minimum: number,
  maximum: number,
  contract: string,
): string {
  if (
    typeof value !== "string" ||
    value.length < minimum ||
    value.length > maximum ||
    value !== value.trim()
  ) {
    fail(contract, `${field} must be normalized text with length ${minimum}..${maximum}`)
  }
  return value
}

function canonicalUuid(value: unknown, field: string, contract: string): string {
  const parsed = normalizedString(value, field, 36, 36, contract)
  if (!UUID_PATTERN.test(parsed)) fail(contract, `${field} must be a canonical lowercase UUID`)
  return parsed
}

function identity(value: unknown, field: string, contract: string): string {
  const parsed = normalizedString(value, field, 1, 160, contract)
  if (!IDENTITY_PATTERN.test(parsed)) fail(contract, `${field} has an invalid identity shape`)
  return parsed
}

function knowledgeId(
  value: unknown,
  field: string,
  maximum: number,
  contract: string,
): string {
  const parsed = normalizedString(value, field, 1, maximum, contract)
  if (!KNOWLEDGE_ID_PATTERN.test(parsed)) fail(contract, `${field} has an invalid identifier shape`)
  return parsed
}

function unitNumber(value: unknown, field: string, contract: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    fail(contract, `${field} must be a finite number between zero and one`)
  }
  return value
}

function boundedInteger(
  value: unknown,
  field: string,
  minimum: number,
  maximum: number,
  contract: string,
): number {
  if (!Number.isInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    fail(contract, `${field} must be an integer between ${minimum} and ${maximum}`)
  }
  return value as number
}

function slotKey(subject: string, topicId: string): string {
  return `${subject}\u0000${topicId}`
}

function sameSet(left: Set<string>, right: Set<string>): boolean {
  return left.size === right.size && [...left].every((value) => right.has(value))
}

function deepFreeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) return value
  for (const nested of Object.values(value as Record<string, unknown>)) deepFreeze(nested)
  return Object.freeze(value)
}

function fail(contract: string, reason: string): never {
  throw new OnboardingContractError(contract, reason)
}
