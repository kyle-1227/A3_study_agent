# 志愿填报下线与 Agent 节点旧实现清理证据快照

日期：2026-07-13

审计基线：`ce64ee2`（`codex/streaming-context-v3`）

状态：替代与迁移实施前证据；不是“当前可全部删除”的证明

## 1. 范围、方法与停止条件

本报告只覆盖已批准的“志愿填报下线与 Agent 节点零旧实现清理”计划，保留
`docs/reports/dead_code_candidates.md`、`docs/reports/fallback_paths.md` 及其他历史报告。
本轮不删除运行时代码。

静态证据通过 `git grep ... HEAD` 固定在上述 commit，避免并行实施中的工作树变化
污染基线。动态入口额外交叉检查了 FastAPI 路由、LangGraph 构图、节点清单、配置、
prompt、测试、前端 App Router 与 checkpoint 接口。Vulture、Semgrep、import-linter、
Gitleaks、Bandit 当前均未安装；它们记为“未运行”，不记为通过。

总停止条件：正式应用仍由 `app.py` 调用旧 `get_compiled_graph()`，且生产 RAG 门未
满足。因此，志愿前端可按已批准范围独立下线，但当前正式图、旧节点 ID、旧
Resource Final、旧 memory prompt 和 fallback 表面均不得仅凭本报告直接删除。

## 2. 当前依赖结论

| 候选表面 | 基线定义/引用证据 | 动态风险与当前结论 | 权威替代 | 删除门 |
| --- | --- | --- | --- | --- |
| `/volunteer` 页面、入口和专属历史 | `frontend/app/volunteer/page.tsx`；`chat-area.tsx` 与 `left-sidebar.tsx` 导航；`VolunteerHistoryItem`、`getVolunteerHistory`、`saveVolunteerHistory`；键 `volunteer_chat_history`、`volunteer_chat_*` | App Router 前端闭环；后端/配置无志愿业务入口。基线页面每次只提交 `query/request_id`，不传 `thread_id/user_id`，不是持久会话记忆 | 整体产品下线；浏览器键一次性清理 | 页面、两个入口、类型/监听器/文案消失；路由 404；只删除目标键；前端测试/typecheck/lint/build 通过。清理 marker 保留一个发布周期后另行删 |
| 当前 `build_graph()` / `get_compiled_graph()` | `src/graph/builder.py` 定义；`app.py:645` 实际编译；checkpointer、manifest、PostgreSQL 测试引用 | **正在服务，禁止删除** | 资源证据 Parent-Child 图成为唯一 `build_graph(runtime)` | 生产启动严格装配新 runtime；生产 index/generation/gold/provider E2E 全通过；切换提交完成；checkpoint 迁移与零旧引用扫描通过 |
| Parent-Child P0 运行时 factory | `build_parent_child_graph` / `get_compiled_parent_child_graph` 被候选图测试和导出引用 | 不是正式服务图，但仍是 P0/消融运行入口 | P0/PG/PR/PGR 保留为离线评估器，不再作为产品运行图 | 离线评估可独立运行且不导入 runtime factory 后，删除 factory/导出；不得删除离线指标、数据校验和四变体基准 |
| 旧节点 ID `rag_retrieve`、`web_search` | 正式构图、事件白名单、活动轨迹、测试与 checkpoint 均引用 | 可能存在 pending task/interrupt，直接改名会破坏 resume | `parent_child_retrieve`、`web_research` | 一次迁移发布临时 alias；终态与中断态迁移通过；未知 ID 阻断；所有 checkpoint 中旧 ID 与 alias pending 数为零后删除 alias |
| `rag_generation_router` 与 `rag_generation_route` | 早期候选图曾注册；当前正式图和候选图均无该节点，planner 已写同一 runtime fingerprint | **已清理** factory/export/transient state/TypedDict/metadata；旧 pending candidate checkpoint 未被伪装为已迁移 | `resource_evidence_planner` 直接入口及 `evidence_orchestration_fingerprint` | **运行时替代已满足**：拓扑、planner fingerprint、registry/state absence 测试通过；未知旧 pending node 继续受 checkpoint migration 阻断门保护 |
| `joint_parent_hydration` / `parent_child_parent_hydration` | 早期候选图曾根据 `rag_generation_route` 复用节点；当前候选图已使用独立 hydration ID | 旧分支函数、状态判断与 metadata 已无活跃引用；旧 pending checkpoint 仍由迁移门处理 | `academic_parent_hydration`、`resource_parent_hydration` | **当前拓扑已满足**：两条路径具备独立节点与测试；最终生产切换仍等待 index/gold/provider/checkpoint 门 |
| `ALLOWED_NODES`、`TEXT_EMIT_NODES`、`GRAPH_NODES` | `app.py` 手写三份集合并参与状态更新、正文流与活动事件；`GRAPH_NODES` 还列出四个未注册节点 | 当前流式运行依赖，不能先删 | 编译图拓扑 + 节点元数据（正文流模式/活动能力） | manifest 与 runtime 元数据成为唯一来源；未知节点严格报错；SSE、状态更新、恢复与 manifest 测试覆盖后删除静态集合 |
| Supervisor 私有短语检测器 | 五组 marker、复数 detector 与单值 wrapper 已删除；生产 `supervisor_node` 只消费严格结构化输出 | LangGraph/export/FastAPI/prompt/config/importlib/getattr 扫描均无动态引用；仓外显式导入私有 helper 不构成公共 API | 严格 `SupervisorOutput` + `validate_supervisor_output` | **已满足**：强资源短语不能覆盖有效 `unknown/general` QA 或最终 `qa` 路由；结构化 runtime wiring、单/多资源、解释性请求和错误输出测试通过 |
| `_sanitize_valid_intents` | 对 `supervisor.valid_intents` 提供默认列表、类型 fallback，并静默剔除 `planning`；现配置已只含合法 intent | 与“无 silent default/无自动修复”冲突；模块导入时执行 | 严格配置 schema/启动校验 | 缺失、类型错误、非法 intent 都 fail-fast 的配置测试通过后删除 sanitizer 与 sanitize trace |

## 3. 学习路径、推荐与评估候选

| 候选 | 当前事实 | 必须先实现的替代 | 替代测试/删除门 |
| --- | --- | --- | --- |
| `curriculum_planner` | 定义于 `src/graph/academic.py`，写 `learning_path/curriculum_context`；未被 `builder.py` 导入，仅出现在手写清单、状态、CE/config/test 引用；异常时返回空字典 | 新图 `learner_path_planner`，显式使用真实 `user_id`、画像、知识图谱和学习记录，返回严格路径/不可用合约 | 路径业务验证、缺画像/历史明确不可用、study-plan/CE 消费测试通过后，删旧函数、旧状态/config 映射 |
| `recommendation_provider` | 未注册旧函数；无画像时返回空列表，异常也吞掉并返回空列表；手写清单、CE provider policy 与状态仍引用 | 新图 `resource_recommendation`，同时支持资源生成后自动轻量推荐和用户显式完整推荐 | 两入口、排序/理由合约、缺数据不可用、资源 final 绑定与恢复测试通过；随后删旧 wrapper/状态/config/UI 映射 |
| `assessment_result_handler` | 未注册；把生成的 exercise item 构造成 `user_answer=""`、`is_correct=True`，没有真实提交 | `assessment_attempt_v1` API；稳定 `question_id`；答案密钥只在服务端 checkpoint；严格身份/答案评估 | 正确/错误/越权/重复 request/未知题目/答案密钥不泄露/PostgreSQL 恢复测试通过后删除伪完成实现 |
| `adaptive_practice_responder` | 未注册；只格式化“将来会分析”的文本，不进行真实错因评估/新题生成 | `adaptive_practice_agent` + `assessment_final_v1` 权威终态 | 错因合约、完整新练习（题目/答案/解释/原因）、SSE journal 重放、单终态测试通过后删除旧节点与正文映射 |

当前 `src/assessment/practice_generator.py` 的占位题目也不能作为上述替代完成证明；
删除旧 assessment 前，新链路必须证明真实题卡提交与服务端答案评估，而非继续把“生成题目”
当成“用户答对”。

## 4. Resource Final 与 checkpoint 迁移候选

### 4.1 当前兼容表面

- `src/graph/resource_final.py` 固定 schema 2，但接受/归一化 legacy payload，并允许
  `terminal_status="unknown"`。
- `app.py` 仍定义 `_legacy_resource_final_payload()`，再交给
  `normalize_resource_final_payload()`；同时保留顶层 `review_doc_artifacts` 等字段。
- `frontend/lib/resource-final.ts` 接受 schema 1 或 2；schema 1 自动得到 `unknown`。
- `ThreadStatusResponse` 接受 `run_control_v1 | legacy`；主页面明确渲染 legacy checkpoint
  警告。
- `last_resource_final_payload` 持久化在 checkpoint/run-control 状态，不能只改前端 parser。

### 4.2 权威替代与删除门

唯一替代为严格 `ResourceFinalV3`：`resources[]` 判别联合、`recommendations[]`、
blocked resources、errors、validation、summary、terminal status、稳定 hash。完成以下全部门后，
才可删除 V1/V2/legacy reader、builder 和字段：

1. 终态 checkpoint 原位迁移为 V3 与 `run_control_v1`，默认 dry-run，显式 `--apply`。
2. 迁移命令通过 checkpointer API（`ObservableCheckpointer.alist` + graph
   `aupdate_state`），不直接写底层 SQL。
3. 同时校验旧/新 graph version、run-control schema、Resource Final schema 与精确 node ID
   映射；未知 pending node 必须阻断。
4. 中断态通过临时 alias 恢复，不重放已完成 provider/resource 工作。
5. 终态、user-stop、profile interrupt、未知节点阻断、PostgreSQL 重启恢复与 SSE
   resource replay 全部通过。
6. 全库扫描 `schema_version="legacy"`、Resource Final 1/2、`terminal_status="unknown"`
   和 pending alias task 均为零。

## 5. legacy memory prompt 候选

### 5.1 替代链路与独立旧层清理已实现

`generate_answer` 已停止动态导入 `src.context.context_builder.build_memory_context`，不再
重复检索 memory，也不再把 memory 文本前置到原始 system prompt 或追加到用户可见正文。
正式 provider-bound 输入现在只接受 Context Engineering 最终选中的 `rules/memory/profile`
项；`rules` 是无 memory 时的 required source，memory/profile 是 optional source。

`MemoryContextProvider` 只读取 state 中已有的 conversation summary、episodic 和 semantic
结果；显式 `ignore` 或待确认的 `ask_user` 会阻止 memory 注入。候选保留真实 thread/user
身份，错 thread 项由 strict source policy 拒绝；稳定 logical item ID 保证同一记忆的新版本
替换当前活跃版本，而不是让 Context Window V3 保留量持续累加。summary、episodic 与
semantic 使用公平限额选择，避免某一桶挤掉其余记忆类型。

替代快照 `ed953ac` 形成后，独立清理批次删除了整个 `src/context/`、仅供旧 builder
使用的 `MemoryContextInjection` 与 public export、旧 memory prompt/footer 常量、
`memory.token_budget`、`tests/test_context_builder.py` 和
`tests/test_token_budget_strict_config.py`。有效的 `src.memory` public import smoke 已迁到
独立测试，storage/retrieval/consolidation/embedding 均保留。

### 5.2 保留与替代

必须保留 Context Engineering 的 `MemoryContextProvider`、`ProfileContextProvider`、
`MessageContextProvider`、Provider Registry、packing/apply、Influence Ledger、Model View、
compaction、Context Window V3、完整 transcript/checkpoint、`LearningState.context` 与
`task_workspace`。`task_workspace` 被 input manifest、evidence/artifact providers、compaction
和运行状态直接消费，不是待删 telemetry。

本次删除依据以下已满足的等价门执行：

- 固定样本证明记忆、画像、消息、摘要、去重和 compact 后 retained token 均等价或更严格；
- provider-bound manifest 证明记忆只由 CE 最终选中项注入；
- Influence Ledger/活动面板能解释记忆影响且不泄露正文；
- 多轮、retry、resume、compact、Context Window V3 统计测试通过；
- 删除后全库不存在 `src.context` 或 `memory.token_budget` 生产引用。

替代验证已覆盖：实际 `generate_answer → invoke_plain_llm_fail_fast → CE → provider`
dispatch、无 memory 的 rules-only 调用、显式 ignore、跨 thread 拒绝、manifest descriptor、
session ledger source stats、Influence Ledger 安全来源计数、稳定 logical ID、三类 memory
公平选择、Model View CE block 去重，以及正式图
`episodic_memory_retriever → memory_use_decider → search_query_rewriter` 顺序。删除前已先
形成并提交全量通过的替代快照 `ed953ac`。

替代快照验证结果：相关回归 `397 passed`，全量后端
`2297 passed, 5 skipped`，前端 69 项测试/typecheck/ESLint/build、compileall、触及文件
Ruff、CE scoped mypy、security tests 和 diff check 均通过。全仓既有 Ruff/type debt 与
缺失的可选安全/死代码工具已在 Streaming V3 状态报告中单独记录，不作为通过项。

独立清理后的最终验证为 `161 passed` 聚焦回归与
`2279 passed, 5 skipped` 全量后端；前端 69 项测试/typecheck/ESLint/build、compileall、
触及文件 Ruff、retained memory scoped mypy、security tests、diff check 和活跃旧符号归零
扫描均通过。首次全量捕获的两条旧 budget 阶段守卫已改为防回归断言，没有删除测试。

## 6. fallback 与假产物候选

| 候选 | 基线引用事实 | 替代/删除门 |
| --- | --- | --- |
| `get_fallback_llm`、`invoke_with_fallback`、`async_invoke_with_fallback` | 定义仅在 `src/graph/llm.py`；调用只在 `tests/test_llm_fallback.py`；manifest 测试还禁止其他生产文件调用 | 证明全部生产 provider transport 走 manifest-guarded 单 provider 路径；保留同 provider 有界 retry；删 helper、导出与只测旧 helper 的测试 |
| `FALLBACK_MODEL/API_KEY/BASE_URL` | `.env.example` 与旧 LLM helper 引用 | helper 删除且配置/secret 扫描无引用后同步删除；不得转成隐藏默认 |
| `fallback_modes` | 实施前 26 个 `config/settings.yaml` 条目均为空，但 structured result/trace/API 与多个节点仍透传字段 | 已改为每节点单一显式 output mode，并从 config、API、result、trace、调用点和 tests 删除旧契约；strict Pydantic/business validation、同模式 semantic retry 与同 provider transport retry 均保留 |
| OpenRouter 专属旧 structured-output 路径 | 位于受保护 `src/llm/structured_output.py` 的历史 provider 分支 | 官方 provider 边界与真实协议 E2E 已覆盖；无生产配置/测试依赖后删除。不得影响独立 RAG embedding/rerank 的生产配置取舍 |
| mindmap fallback | `_build_fallback_mindmap_artifact` 在 structured/provider failure 后生成并通过本地结构检查 | 失败返回 typed resource error；无假 artifact；成功路径与 bundle partial-success 测试通过后删除 |
| review document fallback | fallback markdown、fallback 标志和 reviewer 放行逻辑仍参与 artifact/bundle | typed error；真实生成成功测试和失败不落产物测试替代 |
| code practice fallback | `_fallback_code_practice_markdown`，provider failure 后生成；reviewer failure 可被 deterministic local check 批准 | provider/reviewer failure typed error；禁止“本地检查通过即视作模型审阅成功”测试通过后删除 |
| video script fallback | planner/agent/reviewer/质量失败均可能生成 fallback outline/markdown 或批准 | typed error；完整脚本严格验证，失败不落产物 |
| video animation fallback | `_fallback_animation_spec` 被 planner/agent 空结果和异常路径调用 | typed error；严格 spec/provider 测试，失败不落动画产物 |
| Context Apply fallback policy 字段 | 已证明三个字段没有替代执行分支；规则 fallback 仅为 telemetry 声明 | **已删除**字段、配置解析、settings、trace/SSE 投影与 fixture；预算裁剪、dropped reason、importance observe-only 和同 provider retry 保留 |

多资源 `partial_success` 不是 fallback：至少一个真实资源通过 provider 和业务验证时才允许；
零真实成功必须为 typed failure/controlled stop。字符串 sanitizer 的 `fallback` 参数名若仅提供
安全展示占位，不代表业务替代执行，不在本次批量删除范围。

## 7. 当前生产门状态（2026-07-13）

| 门 | 证据 | 状态 |
| --- | --- | --- |
| 正式构图已切换 | `app.py` 基线仍调用 `get_compiled_graph(checkpointer=...)` | **未满足** |
| 生产 index | `config/rag/index.yaml` 不存在；`index.local.yaml` 存在但不是生产批准文件；本地 `index.runtime.yaml` 未跟踪 | **未满足** |
| 激活 generation | `config/rag/rollout.yaml` 为 `activation_enabled: false`、`shadow_enabled: false`；未发现本地 generation registry | **未满足** |
| 生产语料/独立来源 | 既有 readiness 报告显示五个主学科均缺达到门槛的独立来源 | **未满足** |
| 真实 gold | 本地未跟踪 `human_gold_v2.jsonl` 有 100 行，但 `historical_annotated_v2.jsonl` 为空；未有通过资格/数据所有者确认 | **未满足** |
| P0/PG/PR/PGR 结果 | 只有 evaluator/config/门定义，未发现四变体真实 gold result bundle | **未满足** |
| 真实 provider E2E | 现有报告没有给出新正式图从启动到 QA/资源/重连/评估的真实 provider 端到端通过证据 | **未满足** |
| checkpoint migration | 已实现严格、默认 dry-run 的依赖注入迁移核心与 `--apply` 全批预验证；生产 checkpointer/graph/projector/schema validator adapter、旧节点 alias 周期、逐 checkpoint 写失败幂等恢复和零旧 checkpoint 证明仍未完成 | **未满足** |

因此，当前允许推进“替代链路实现、志愿前端下线、迁移工具与测试”；禁止删除当前正式图、
旧 checkpoint reader/alias 前置能力或任何仍被新链路实际消费的状态。

## 8. 分阶段删除门与回滚边界

1. **证据提交**：本报告独立提交；不包含运行时删除。
2. **志愿下线提交**：页面、入口、专属历史与定向 storage purge；可独立回滚。
3. **替代实现提交**：唯一新图、路径/推荐/assessment、Resource Final V3、严格无 fallback；
   仍保留旧正式图。
4. **迁移提交**：dry-run/apply CLI、临时 alias、终态/中断态迁移与审计计数。
5. **生产切换提交**：只在第 7 节全部变为“满足”后切换；禁止 request-time 双图 fallback。
6. **零旧实现清理提交**：扫描为零后删除旧图/factory/node ID/legacy schema/memory/fallback；
   与切换分开，以便只回滚清理。
7. **迁移尾项提交**：一个部署周期且 checkpoint/storage marker 均清零后，删除 alias、迁移
   reader/script 和志愿 storage purge marker。

任一候选无法证明无动态引用时停止删除，在本报告或
`docs/reports/dead_code_candidates.md` 追加证据，不得靠删除测试获得绿色结果。

## 9. 验收扫描与测试索引

最终清理前至少执行并记录：

- 志愿：路由/入口不存在、目标 localStorage 定向清理、Vitest、ESLint、typecheck、Next build。
- 图：启动 fail-fast、唯一 topology/manifest、节点元数据、academic/resource hydration、推荐双入口。
- 评估：typed attempt API、答案不泄露、幂等、错因、自适应新题、`assessment_final` 单终态/重放。
- Resource Final：V3 Pydantic/business/hash、旧 checkpoint 迁移、前端唯一 parser、恢复与去重。
- checkpoint：终态、user-stop、profile interrupt、pending old node、unknown node block、PostgreSQL。
- CE：旧 prompt 等价样本、provider manifest、Influence Ledger、多轮/compact/Context Window V3。
- 资源失败：mindmap/review/code/video/provider/validation failure 不产生假 artifact；真实部分成功保持。
- 全局：`python -m compileall -q src tests app.py`、相关与全量 pytest、Ruff check/format、
  触及模块 mypy、`git diff --check`、旧符号/路由/config/prompt/dynamic entry 人工扫描。

Vulture、Semgrep、import-linter、Gitleaks、Bandit 仅在安装时运行；缺失必须继续记录为
“未运行”。

## 10. 明确保留边界

- Context Engineering contracts/providers、packing/apply、Influence Ledger、LLM Input Manifest。
- provider 输入预算与 `ContextUsageReport`。
- 完整 transcript/checkpoint、run control、activity timeline、`task_workspace`。
- `LearningState.context` 的检索证据与 Parent-Child 离线 P0/PG/PR/PGR 评估能力。
- 同 provider 有界 transport retry、明确 controlled stop、真实多资源 partial success。
- 与本计划无关的历史 dead-code 候选与报告。

## 11. Quiz replacement progress (2026-07-13)

- The Quiz producer now requires strict `ExerciseArtifact` and
  `ExerciseReviewVerdict` contracts. `question_type`, `choices`, canonical
  levels, complete level coverage, unique questions, and single-choice answer
  membership are validated without aliases or repair.
- `exercise_agent` and `exercise_reviewer` explicitly reject unsuccessful
  `StructuredLLMResult` values. Only an exact `approve` verdict may reach
  output; empty, unknown, rejected, or max-round states are typed failures.
- Document write failures and post-write renderability failures no longer
  produce a successful Quiz branch. The public Markdown, AI message, artifact,
  and Resource Final V3 quiz projection contain only `PublicExerciseCardV1`.
- Private answers are stored only in the durable
  `assessment_checkpoint_resources` projection. The projection is validated
  through strict JSON semantics so LangGraph `JsonPlusSerializer` list-shaped
  checkpoint recovery remains valid without key normalization or coercive
  compatibility adapters.
- The obsolete Quiz `quality_warning` field and the duplicate top-level
  `exercise.*` runtime configuration are removed. Quiz now requires explicit
  `llm.exercise.model`, `llm.exercise.temperature`, and
  `llm.exercise.max_generation_rounds` configuration.
- This is a replacement milestone, not the assessment deletion gate. The
  global Resource Final V3 runtime switch is now complete. The strict durable
  attempt journal and authoritative `assessment_final` stream terminal are now
  implemented as a tested foundation, but the
  `POST /threads/{thread_id}/assessment-attempts` route, real structured
  classifier/generator callbacks, checkpoint adapter, and PostgreSQL endpoint
  recovery are still required before deleting old assessment surfaces or the
  active graph.

## 12. Dead code/diff risk review

- Vulture run：未运行（未安装）。
- Candidates reported：本报告第 2 至 6 节；均附动态风险、替代和删除门。
- Code deleted：否；本证据阶段不删除运行时代码。
- Diff remains scoped：是；本文件为本阶段唯一预期改动。

## 13. Resource Final V3 authoritative cleanup progress (2026-07-14)

- Runtime and frontend no longer project or accept Resource Final V1/V2.
  A resource run without `resource_final_v3` fails closed with
  `resource_final_v3_missing`; non-resource evidence summaries terminate as a
  validated `qa_final`.
- Deleted compatibility surfaces include `src/graph/resource_final.py`, its
  V1/V2-only tests, frontend pseudo-completion/legacy result branches, and the
  unreferenced legacy SSE bubble comparison script.
- PostgreSQL reconstruction now persists only Resource Final V3. Activity safe
  details use V3 final identity and counts instead of a `resource:v1` fixture.
- Full backend regression passed with `2280 passed, 5 skipped`; frontend
  Vitest (69 tests), typecheck, full ESLint, and Next build passed.
- This satisfies only the Resource Final replacement slice. Production index,
  generation activation, real four-variant gold evidence, provider E2E, and
  zero-legacy checkpoint scans remain unsatisfied, so the current formal graph
  and migration readers were not deleted.

## 14. Context Apply fallback-field cleanup progress (2026-07-14)

- Removed all active source/config/test references to the three inert Context
  Apply fallback fields. There is no compatibility adapter or telemetry claim
  for a rule-based substitute.
- Context Apply failures remain typed and fail closed. Graceful whole-item
  budget trimming, required/optional source diagnostics, source and budget
  drop reasons, importance observe-only telemetry, and same-provider bounded
  transport retry remain intact.
- Validation: 303 focused Context Apply/importance/stream tests, 59 config and
  security tests, 4-file scoped mypy, full backend (`2280 passed, 5 skipped`),
  frontend 69-test Vitest/typecheck/ESLint/build, compileall, touched Ruff, and
  active-field scans passed.
- This is independent of the production graph switch. It does not satisfy or
  bypass the still-missing production index, activation, real gold variants,
  provider E2E, or zero-old-checkpoint gates.

## 15. Supervisor phrase-detector cleanup progress (2026-07-14)

- Deleted the five private phrase-marker collections and both deterministic
  resource-request detector helpers after repository, dynamic import, graph,
  route, prompt, and config scans proved they had no production consumer.
- Deleted only the tests that directly exercised those private tables. Their
  behavioral replacement proves a strongly worded resource query cannot
  override a Pydantic- and business-valid `unknown/general` QA result, and that
  the resulting graph route remains `qa`.
- Retained structured resource normalization, `SupervisorOutput`,
  `validate_supervisor_output`, single/multi-resource behavior, explanatory QA
  behavior, and all strict negative validation tests. `_sanitize_valid_intents`
  remains active and is deferred to its own fail-fast configuration spec.
- Validation: 75 focused Supervisor/Builder/Manifest tests and the full backend
  suite (`2273 passed, 6 skipped`) passed; frontend 69-test Vitest, typecheck,
  ESLint, and production build passed. Compileall, touched Ruff, scoped mypy,
  diff check, and old-symbol scans passed.
- This slice is independent of the production graph switch. It does not remove
  or weaken the current formal graph, checkpoint aliases/migration readers, or
  any Parent-Child RAG runtime work, and it does not satisfy the outstanding
  production index, generation activation, real gold, provider E2E, or
  zero-old-checkpoint gates.

## 16. Superseded RAG generation-router cleanup progress (2026-07-14)

- Removed `make_rag_generation_router_node`, its public module export,
  `rag_generation_route` transient/default and TypedDict fields, and the
  orphaned node metadata. Neither active graph registered the node before this
  cleanup.
- Retained the strict `resource_evidence_planner`, its Pydantic/business
  validation, direct candidate edge, runtime fingerprint, distinct academic
  and resource hydration nodes, evidence trace, and all bounded repair logic.
- Added absence coverage for module export, state/reset contract, registry
  metadata, and candidate topology. The existing mocked planner test now also
  proves the active replacement writes the runtime orchestration fingerprint.
- The early candidate node existed briefly in repository history. This cleanup
  does not rewrite checkpoints or declare old pending candidate tasks migrated;
  unknown pending nodes remain blocked by the migration gate.
- Focused verification passed with 51 evidence graph/trace/state/manifest
  tests, compileall, touched Ruff, three-file mypy, diff check, and active old
  symbol scans. Full backend (`2274 passed, 6 skipped`) and frontend 69-test
  Vitest/typecheck/ESLint/production build also passed. Production activation
  blockers remain unchanged.

## 17. Assessment durable-journal foundation progress (2026-07-14)

- Added `assessment_final` as an `agent_stream_v2` authoritative terminal with
  sequencer, journal capacity, session completion, replay, and single-terminal
  coverage.
- Added a strict, bounded thread checkpoint journal containing only request
  hashes, public finals, and UTC commit times. Original submitted answers and
  private answer-key material are not persisted in this journal.
- Added an injected checkpoint idempotency executor: duplicate identical
  requests execute once and replay; request ID/content conflicts fail; failed
  operations are not cached; append success must be proven by a strict reread.
- Focused assessment/state/stream tests passed (`97 passed`), together with
  compileall, touched Ruff, six-file mypy, security tests, and diff check.
- Full backend (`2288 passed, 6 skipped`) and frontend 69-test Vitest,
  typecheck, ESLint, and production build passed.
- This does not yet satisfy the deletion gate. The FastAPI endpoint, real
  structured provider callbacks, checkpoint read/write adapter, PostgreSQL
  recovery/concurrency proof, frontend submission, and final OpenAPI/E2E tests
  remain required.

## 18. Strict assessment endpoint replacement progress (2026-07-14)

- Added `POST /threads/{thread_id}/assessment-attempts` with the exact
  `assessment_attempt_v1` request schema. The route resolves the private answer
  key only from the thread checkpoint and emits exactly one
  `assessment_final` or safe `stream_error`, followed by session-owned
  `stream_done`.
- Replaced the placeholder callback path at the endpoint boundary with two
  provider-neutral strict runtimes: `error_classifier` and
  `practice_generator`. Both use one configured output mode, Pydantic with
  forbidden extras, explicit business validation, required rules-only Context
  Engineering, and no provider/model override in business code.
- Private evaluation data is wrapped in a provider-only envelope whose trace
  preview begins with a non-sensitive notice. The shared structured-output
  runtime now has an explicit `sensitive_trace=True` mode that preserves
  counts/stages/types while removing message preview content, raw output,
  provider error bodies, and validation text. Canary tests prove private
  submitted answers and answer explanations do not enter public finals,
  checkpoint journals, SSE errors, or structured-output trace payloads.
- The durable journal now writes an `in_progress` claim before any provider
  dispatch, then transitions the exact claim to `completed` or content-free
  `failed`. Identical completed/failed requests replay without dispatch;
  conflicting payloads fail; cancellation leaves a recovery-required claim and
  never automatically redispatches.
- Memory checkpointers use a process-local thread lock. PostgreSQL composition
  uses a connection-scoped advisory lock derived from a domain-separated
  thread hash. The SQL is parameterized and the DB URI is never traced. The
  real two-connection PostgreSQL test is present but skipped unless
  `A3_TEST_POSTGRES_URI` is explicitly configured.
- Adaptive practice is consistently limited to 1-3 complete tasks. The draft
  contract rejects repeated original questions, missing classification-specific
  task types, non-canonical whitespace/tags, duplicate questions, blank fields,
  schema drift, and unstable server-derived question identities.
- Cold-import coverage removed an assessment/structured-output circular import
  without deleting the legacy classifier/generator exports: the two old exports
  remain available through explicit lazy loading until their deletion gate is
  satisfied.
- Verification: 310 focused/config/security/stream tests passed with one real
  PostgreSQL test skipped; full backend passed with `2334 passed, 7 skipped,
  12 warnings`; frontend baseline passed with 23 Vitest files/69 tests,
  typecheck, full ESLint, and Next production build. Compileall, touched Ruff,
  8-file scoped mypy, and `git diff --check` passed. Whole-repo Ruff still has
  60 pre-existing lint errors and 65 pre-existing formatting files. Semgrep,
  import-linter, Gitleaks, Bandit, and Vulture are missing and were not run.
- This is not the old-assessment deletion gate. Frontend exercise submission,
  real Provider E2E, real PostgreSQL endpoint/restart recovery, and the global
  production graph/checkpoint gates remain outstanding; therefore
  `assessment_result_handler`, `adaptive_practice_responder`, the placeholder
  generator, and their state/config surfaces remain in place.

## 19. Strict frontend assessment replacement progress (2026-07-14)

- Resource Final V3 quiz cards are now exact-field parsed before message
  projection or browser persistence. The duplicated public card list and
  public artifact list must match exactly; private answer fields and unknown
  fields fail the whole contract. Stored messages are rebuilt from a freshly
  parsed V3 payload instead of trusting a persisted nested exercise object.
- Added a strict assessment client on the shared `agent_stream_v2` transport.
  It performs one POST with the caller's UUID, permits only explicit
  Last-Event-ID GET replay, rejects sequence gaps/conflicting duplicates and
  multiple or mismatched terminals, and binds thread/request/resource/question,
  elapsed time, hash shape, and terminal truth. It never reads an HTTP error
  body into diagnostics and never logs or persists the submitted answer.
- Added interactive free-text/single-choice cards with explicit
  idle/editing/submitting/correct/incorrect/failed/conflict states. The main
  chat stream and assessment stream are mutually excluded, other cards are
  disabled during assessment, and thread navigation aborts the active request.
- Verification passed: focused frontend 61 tests; full frontend 27 files/117
  tests, typecheck, full ESLint, and production build; compileall; full backend
  `2334 passed, 7 skipped`; and diff check. Whole-repo Ruff still reports the
  same 60 lint errors and 65 format files. Semgrep, import-linter, Gitleaks,
  Bandit, and Vulture remain unavailable and were not recorded as passing.
- The frontend-submission sub-gate is now satisfied. Deletion is still blocked
  by real Provider E2E, real PostgreSQL endpoint/restart recovery, production
  Parent-Child activation evidence, checkpoint migration, and a zero-legacy
  checkpoint scan. The old assessment nodes and migration readers therefore
  remain intentionally present.

## 20. Candidate safe learner-path, recommendation, and fan-in replacement progress (2026-07-14)

- The candidate Parent-Child graph now injects a strict `LearningGuidanceRuntime`
  and executes `learner_path_planner` before resource evidence planning. Both
  the evidence planner and study-plan planner revalidate the checkpoint-safe
  path schema and request/user/subject binding before consuming it. An explicit
  lowercase SHA-256 guidance runtime fingerprint is part of the candidate
  orchestration fingerprint; callable repr hashing is not used. Projection
  step/character policy and schema are also fingerprinted.
- The full learner path remains in the checkpoint, while Provider calls receive
  only an exact `learner_path_provider_projection_v1` reconstructed from it.
  The projection contains no request/user/profile/history identity fields and
  has explicit step and character ceilings. Oversize, stale, schema-drifted, or
  tampered projections fail before Provider dispatch; no truncation or fallback
  path exists.
- Path and recommendation checkpoint outputs atomically bind the guidance
  runtime fingerprint, projection-policy fingerprint, and concrete limits.
  Evidence and finalization consumers reject outputs from a changed runtime.
- Learner-path scope is derived again from `retrieval_plan[].subject`. A path
  engine is called only for one subject matching the graph's subject. Multi-
  subject and mismatched requests return `unsupported_subject_scope` before
  profile/history I/O, and both Provider consumers reject a stale single-
  subject available path in a multi-subject state.
- The candidate fan-in is now
  `resource_bundle_aggregator → resource_recommendation_auto →`
  `resource_bundle_output`. The aggregator never
  writes a message or terminal payload. It derives stable IDs only from a
  local, discarded strict projection and exposes no quiz answer material.
- Every post-planner evidence node and every candidate resource node validates
  the current orchestration fingerprint before continuing. The fingerprint
  covers Parent-Child handoff, evidence/profile policy, prompts/schemas,
  guidance runtime/projection policy, and Web timeout, so a resumed run cannot
  mix old checkpoint semantics with a new process runtime.
- The finalizer accepts only the automatic mode, revalidates recommendation
  identity and source-resource references, and builds one Resource Final V3.
  Missing profile/history/resources and unsupported multi-subject scope do not
  create neutral scores or fake recommendations.
- Automatic recommendation targets are now bound to an actual generated
  bundle `resource_id`, retained in Resource Final V3, and checked again by the
  strict frontend parser. A fabricated catalog/target ID cannot become an
  actionable-looking automatic recommendation.
- A compiled LangGraph test now proves real dual-`Send` fan-in: two workers join
  into one aggregator, one recommendation node, and one finalizer/Resource Final
  V3. The same graph-level fixture covers empty tasks and recommendation
  fail-fast behavior.
- The currently served graph still uses the original dispatcher and bundle
  finalizer. Its old `curriculum_context` branch is retained only because the
  production graph has not switched; the new candidate path does not write or
  depend on that field. This is a documented temporary dependency to delete
  with the formal graph, not a replacement implementation.
- No old node was deleted in this slice. Explicit recommendation still lacks a
  Supervisor action and a recommendation-only authoritative terminal. Public
  Resource Final V3 also lacks a machine-readable recommendation availability
  envelope; an empty array plus public summary is not sufficient for the final
  public-contract deletion gate. The recommended next contract is an independent
  `recommendation_final_v1` bound to a real catalog/KG candidate snapshot; this
  preserves Resource Final V3's invariant that success contains a generated
  resource. Extending V3 with recommendation-only success remains a user choice.
- Candidate implementation commit:
  `eee3d8e0b5042c16b1d975a2bea568652a6a2271`.
- Verification passed with 167 focused path/graph/study-plan/fan-in/Resource
  Final tests, 274
  expanded candidate/Resource Final/stream/manifest/security tests, full backend
  `2383 passed, 7 skipped`, frontend 118-test Vitest/typecheck/ESLint/build,
  compileall, touched Ruff,
  scoped mypy, cold imports, and diff checks. Ordinary dependency-following
  mypy timed out twice at 120 seconds and was not recorded as passing. Whole-
  repo Ruff remains at 60 lint findings and 65 formatting files. Semgrep,
  import-linter, Gitleaks,
  Bandit, and Vulture are unavailable and were not run.
- The production deletion gate remains closed at candidate commit `eee3d8e0`:
  the canonical production index and knowledge graph are absent,
  rollout activation is false, the latest local continuation is experimental,
  evaluation-ineligible, activation-disallowed and explicitly
  activation-prohibited,
  real Provider/PostgreSQL/four-variant evidence is incomplete, and no
  zero-legacy checkpoint scan exists. The formal graph and migration readers
  therefore remain required.

## 21. Retired volunteer guidance cleanup (2026-07-15)

The user selected D10-B: retain the targeted localStorage purge for one
migration release, then delete it. The purge component, marker, and focused
test therefore remain intentionally active in this batch.

Repository guidance no longer advertises the retired product. `PRODUCT.md` and
`DESIGN.md` now require retired routes and storage to be removed after their
migration window; `scripts/demo_profile.py` no longer instructs callers to
integrate the volunteer page. Historical audit reports remain unchanged as
evidence and are not runtime compatibility surfaces.

## 22. Checkpoint migration retirement after decision D7 (2026-07-15)

The user explicitly approved clearing existing checkpoints instead of
preserving them through a node/schema migration. Static scans found no runtime,
FastAPI, graph-builder, CLI-entrypoint, dynamic import, or configuration caller
outside the self-contained `src.checkpoint_migration` package, its deliberately
unwired script, and fake-only unit test. The script always stopped with a parser
error because no production adapter had ever been connected.

The unused package, script, and fake-only migration test were deleted. The
replacement is `docs/runbooks/checkpoint_clear_cutover.md`, which requires a
verified custom-format backup and stopped writers before clearing only
`checkpoints`, `checkpoint_blobs`, and `checkpoint_writes`. It explicitly
preserves `checkpoint_migrations`, forbids `CASCADE`, and requires a new-graph
write/reload and zero-legacy verification after restart.

A read-only probe of the existing Docker PostgreSQL instance observed `2,120`
checkpoint rows, `9,863` blob rows, `25,535` write rows, and `10` migration
version rows. No clear was executed in this batch: the old served backend may
still write until the new RAG graph reaches its cutover window. Counts must be
captured again immediately before backup and clear.

## 23. Real local PostgreSQL sub-gate (2026-07-15)

The existing healthy Docker PostgreSQL service was exercised with a connection
URI assembled only in the test process from the container environment. The URI
and password were neither printed nor written to `.env` or a report.

The first real run isolated a Windows-only test-runner defect: pytest used the
Proactor event loop, which psycopg async connections reject, while the formal
backend already uses `postgres_compatible_event_loop`. The opt-in integration
test now runs its scenario through an `asyncio.Runner` with that exact loop
factory; no runtime lock code, retry, or fallback changed.

After the fix, two independent `PostgresAssessmentExecutionLock` instances
serialized the same thread advisory lock, and the real LangGraph PostgreSQL
fixture wrote durable state, reopened a saver, reconstructed thread status, and
removed its random test checkpoint. Both real tests passed. This closes the
database availability, async-loop, lock, and saver-reopen sub-gates. It does not
yet prove a full assessment HTTP process-restart flow or the new RAG served
graph; those remain separate integration gates.

## 24. Real DeepSeek assessment runtime sub-gate (2026-07-15)

A fixed, non-user binary-search error sample was sent through the configured
DeepSeek assessment classifier and adaptive-practice generator. The existing
`DEEPSEEK_API_KEY` was loaded into one process only; no key, prompt body,
student answer, generated question, answer, or Provider body was printed or
persisted.

The first live run proved the classifier callback but exposed a real contract
bug in the generator: strict Pydantic fields declared as Python tuples cannot
accept JSON arrays after tool arguments are decoded. Unit tests had constructed
the draft with tuples and therefore hid the production failure. The two
Provider-only draft arrays (`tasks` and task `tags`) now require strict lists;
tuple input is covered by negative tests and remains rejected. The runtime then
performs one explicit list-to-tuple projection when building the final immutable
`AdaptivePracticeBatchV1`; structured-output parsing itself was not loosened.

The second live run passed both Provider calls. It returned a valid
`assessment_error_classification_v1`, a three-item
`adaptive_practice_batch_v1`, the required concept-error review task, and three
unique stable question IDs. Focused assessment regressions passed `58`; both
touched source files passed mypy, Bandit, Ruff, and Semgrep. This closes the
real classifier/generator callback sub-gate.

The same fixed sample was then exercised through the formal assessment SSE
endpoint and the real PostgreSQL saver. A correct-answer control emitted
`stream_start -> assessment_final -> stream_done`; after closing and reopening
the saver, the recorded final had the same payload hash and a Provider-failure
sentinel proved replay performed no new call. The incorrect-answer run then
completed the full chain with real DeepSeek classification/generation, three
adaptive tasks, and the same event sequence. After another saver reopen, the
incorrect terminal, three tasks, and payload hash were identical, again without
calling either Provider callback. Every random live thread was deleted in a
`finally` block. This closes the assessment HTTP, real Provider, PostgreSQL
restart, idempotent replay, and single-terminal sub-gate; it does not close any
new-RAG served-graph or RAG Provider gate.
