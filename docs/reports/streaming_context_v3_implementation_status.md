# Streaming + Context Window V3 实施状态

更新时间：2026-07-13

## 结论

本次统一重构已经在本地分支 `codex/streaming-context-v3` 落地。固定基线为
`9bf9950c002d71b7d184fadcd35993ab35306bd5`，代码、测试、前端构建和旧实现清理均已完成；
未 push、未 merge、未修改 main。

正式运行时现在只保留 `agent_stream_v2` 与 `thread_context_window_v3`。旧
`token/text` 流、`ThreadContextWindowV2`、V1-to-V2 SSE adapter 及其专属测试均已删除。

## 事件契约与实际交互

所有正式事件都包含：

- `schema_version=agent_stream_v2`
- `stream_id`、`event_id`、`sequence`
- `request_id`、`thread_id`、`created_at`

正式事件集合为：

- `stream_start`
- `content_block_start`、`content_block_delta`、`content_block_stop`
- `activity_update`、`tool_progress`、`artifact_progress`
- `qa_final`、`resource_final`
- `interrupt`、`stopped`、`stream_error`
- `stream_done`

权威终态只能是 `qa_final`、`resource_final`、`interrupt`、`stopped` 或
`stream_error` 中的一个；`stream_done` 只能由 session 在权威终态之后追加。producer 不再直接编码
SSE，也不再发送旧 `token/text/done` 形状。

交互链路：

1. `/stream`、`/resume` 和 continue 请求使用客户端 `request_id` 建立或附着运行。
2. producer 直接生成 `AgentStreamEventDraftV2`；session 负责身份、sequence、终态状态机和 SSE 编码。
3. 相同 `request_id` 与相同请求指纹只附着已有运行；冲突请求显式失败，不重复执行图。
4. `GET /streams/{stream_id}` 使用 `Last-Event-ID` 续传；journal 检查重复、gap、TTL 和容量。
5. DeepSeek strict tool-call 的 `function.arguments` 只投影 `QAResponse.answer` 增量，浏览器不接收原始 JSON。
6. provisional block 只进入 `LiveTurnState`；仅通过校验的 `qa_final/resource_final` 才提交正式消息。
7. 主页面和志愿填报页面共享同一个 SSE parser/client，支持 CRLF、多行 data、UTF-8 跨 chunk、EOF flush、重连和旧请求隔离。
8. 情绪支持与未知意图节点也通过 Pydantic、业务校验和 `qa_final` 终态提交，不会在 provisional 正文后落入 `stream_error`。

前端 Markdown 采用 stable prefix / unstable suffix，并使用 animation frame 批处理增长更新。

## 会话记忆账本与 Context Window V3

`SessionContextMemoryLedgerV1` 只记录真正 dispatch 给 Provider 的 Context Engineering 项；记录稳定
fingerprint、token、tokenizer/estimated 状态和 request/call/attempt 身份，不保存正文。

已验证的多轮统计示例：

- 同一 7-token 内容注入两次：`retained=7`、`lifetime_injected=14`、
  `lifetime_unique=7`、`injection_count=2`、`repeat_injection_count=1`。
- 同一 10-token 内容跨两个请求真实 dispatch：`request_count=2`、
  `lifetime_injected_tokens=20`；transient request reset 不清空账本。
- replay 同一个 record 保持幂等：`request_count=1`、`injection_count=1`。
- 同一逻辑项从 5 tokens 更新为 9 tokens：当前保留量为 9，历史累计与历史去重均为 14。
- 只有显式记忆清理会把 retained、lifetime 和 injection count 归零。

`ThreadContextWindowV3` 顶部百分比固定为：

`retained_memory_tokens / context_window_limit_tokens`

确定性验证示例：`10,000 / 1,000,000 = 1%`。V3 同时保留历史累计、历史去重、重复注入、请求数、
注入次数、各 source 类型统计、measurement、memory summary 和 compaction 状态。它不包含线程基线、
下一调用估算、target node、输出预留、预测增长、projected peak 或 headroom。分母必须由显式
`session_memory.window_model` 解析到 `model_limits`；配置缺失时显式报错。

新请求期间保留上一份快照并标记 updating；只有新 thread 或显式清理才归零。

## Model View 与压缩

完整 UI transcript 和 checkpoint messages 保持不变；Provider 调用使用独立
`ModelViewProjectionV1`。micro compaction 去重 CE block，并只在存在可信摘要 metadata 时替换旧工具结果；
system/schema/current query、最近 API rounds 和 tool-use/tool-result 配对均被保留。

full compaction 只依据最近一次真实、可触发的 provider dispatch 和显式阈值决定，不预测下一轮增长。
`CompactBoundaryV1`、`ConversationSummaryV2`、`CompactionResultV1`、账本和 V3 在同一次 checkpoint
更新中原子提交；摘要校验或提交失败会阻断图执行。

可复现实测（与 `tests/test_full_compaction_app.py` 同一 fixture）：

- 触发输入：`900,000 / 1,000,000 = 90%`
- model view：`12,884 → 132 tokens`
- compacted messages：2
- retained messages：5
- 该 fixture 无活跃 CE ledger：`ledger 0 → 0`

账本压缩的独立确定性 fixture：`retained 13 → 8 tokens`，
`lifetime_injected_tokens=13` 保持不变；尝试静默压缩为 0 会被拒绝。

设计方向参考 Claude Code 的 transcript/model-view 分离、实际 usage 更新和 compact boundary；仓库只吸收了
可审计的模式，没有安装或执行第三方脚本。

## 已删除的旧实现

删除文件：

- `src/context_engineering/thread_window.py`
- `src/streaming/adapter.py`
- `frontend/lib/thread-context-window.test.ts`
- `tests/test_thread_context_window_v2.py`
- `tests/test_agent_stream_adapter.py`

删除符号与运行时字段：

- `ThreadContextWindowV2`
- `thread_context_window_v2`
- `build_thread_context_window_v2`
- `adapt_legacy_sse_stream`
- `generate_sse`、`generate_resume_sse`、`generate_continue_sse`
- 后端旧 `token/text` emission 与前端消费分支
- 两个页面中的旧 Context Window V2 state/parser/fixture

保留项包括 Context Engineering collect/packing/apply、Influence Ledger、LLM Input Manifest、
`ContextUsageReport`、Provider 输入预算、完整 transcript/checkpoint、`task_workspace`、run control、activity
timeline 和正式 QA/resource contract。

清理前替代覆盖与人工引用证据见 `docs/reports/dead_code_candidates.md`。清理提交为
`b12142bef27460d0d3ecc97952b964ee2359f966`，可独立回滚。

## 质量门结果

通过：

- `python -m compileall -q src tests app.py`
- 全量 pytest：`2004 passed, 5 skipped, 9 warnings`，159.01 秒
- 清理相关合并回归：`521 passed, 3 warnings`
- 前端 Vitest：`21 files passed, 68 tests passed`
- `npm run typecheck`
- `npm run lint`
- `npm run build`，Next.js 生产构建与全部静态页面生成成功
- 本次触及文件 `ruff check`
- 本次触及文件 `ruff format --check`
- scoped mypy：9 个源文件 `Success: no issues found`
- `git diff --check`
- OpenAPI/thread status 扫描：无 Context Window V2，status 暴露 V3
- 运行时旧符号、旧路由、旧配置、旧前端消费和手写 `split("\\n\\n")` 扫描：无命中

未通过但属于本次范围外的既有仓库债务：

- `ruff check .`：61 条错误，分布于旧脚本、analytics/curriculum/profile 等模块及用户调试文件；本次触及范围无错误。
- `ruff format --check .`：76 个既有文件待格式化；未进行无计划的全仓格式化。

mypy 说明：默认依赖遍历缺少 `types-PyYAML`，并在 184 秒命令超时前发现了三个本次类型问题；三个问题均已修复。
最终 scoped 命令使用 `--no-incremental --follow-imports=skip --disable-error-code=import-untyped`，覆盖
`app.py`、streaming、QA/emotional/supervisor、run-control 和 schema，共 9 个源文件并通过。外部 PyYAML stub
缺失没有记为通过。

可选工具状态：

- Semgrep：缺失，未运行
- import-linter (`lint-imports`)：缺失，未运行
- Gitleaks：缺失，未运行
- Bandit：缺失，未运行
- Vulture：缺失，未运行

## 警告与剩余风险

- 全量 pytest 仍报告既有 `aiosqlite` worker 在线程结束后访问已关闭 event loop 的 warning；测试退出码为 0，但该 shutdown debt 未伪装为通过项。
- 未使用真实生产凭据执行 DeepSeek/provider 网络 E2E；strict arguments streaming、provider retry 和失败语义由 provider mock、graph mock 与契约测试覆盖。
- stream journal 是进程内、TTL/容量受限的恢复日志；进程重启后的跨进程续传不在本次合约范围内。
- 全仓 Ruff 和可选安全/死代码工具仍需单独治理，不应混入本次 streaming/context 重构提交。

## 原子提交

1. `b65be664247359a938ebb7c380dd6243d67c28f5` — profile completion 终态
2. `d11c597f6d7b56926f569eaa6b3285cae4a72554` — agent stream v2 contracts
3. `9d1842eed1f27bd5341e9594d764a37858ae5707` — provisional structured QA
4. `b5ac16b274108a52a72322cf6c1d7b719b1a867a` — LiveTurn 与 SSE parser
5. `14dfcef320e0c588fa6f408a813ba51871ab6a98` — streaming Markdown/progress UI
6. `ccf7f89b7f03d5023ebdcf827fabb6074c6ed436` — persistent injection ledger
7. `c1e4d738073218c872bf15ad8d6bdf3bb837f3e6` — Thread Context Window V3
8. `2bbfbe3ddfeff07df98e71a64c7d8c7513dfb2ad` — model view/micro compaction
9. `e4b7903202185ca93534f1a0de01d1e8c27a3c90` — full compaction/recovery
10. `f58ce516bd19d8dff34b483994249b42dfcab0fb` — replacement parity/report
11. `b12142bef27460d0d3ecc97952b964ee2359f966` — superseded code cleanup

实现代码与清理的最终 SHA 为 `b12142bef27460d0d3ecc97952b964ee2359f966`。本报告自身的提交 SHA
无法自嵌入而不改变 Git object ID，因此由最终交付消息和 `git log -1` 报告。

## 2026-07-14 Resource Final V3 权威化清理补充

本补充批次把流式终态中的 Resource Final 从“V3 优先、V1/V2 可投影”收紧为只接受
`resource_final_v3`。资源流程到达终态但没有严格 V3 时，运行状态写为失败并发送唯一
`stream_error(error_type=resource_final_v3_missing)`；不会发送
`completed_without_resource` 或伪完成。非资源 evidence controlled stop 现在构造严格
`QAResponse`，通过 Pydantic 与业务校验后以 `qa_final` 提交。

本批删除：

- `src/graph/resource_final.py` 及 V1/V2 normalizer、stable V1 ID/hash、legacy
  compatibility builder；
- `tests/test_resource_final_contract.py` 与其他仅验证 legacy projection 的断言；
- 前端 `completed_without_resource`、`resource_final_diagnostic`、
  `mindmap_result`、`review_doc_result` 分支；
- 无任何入口且仍模拟 `token/text/mindmap_result` 的
  `scripts/compare_sse_bubble_output.py`；
- activity fixture 中的 `resource:v1` 身份，改为
  `resource_final_id/resource_count/terminal_status`。

权威恢复证据包括 Resource Final V3 PostgreSQL fixture、stale-checkpoint terminal
output、request/thread 身份拒绝、资源缺 V3 fail-closed、非资源 `qa_final`、公开 quiz
不泄露答案、断线 journal/单终态测试。Phase-0 回归中最后一个 V1 quiz fixture 已用正式
V3 builder 重写，没有删除其 node activity、usage 与 final-event 共存断言。

本批质量门：

- `python -m compileall -q src tests app.py`：通过；
- 相关回归：`287 passed, 1 skipped`；
- 全量 pytest：`2280 passed, 5 skipped, 9 warnings`；
- 前端 Vitest：23 个文件、69 项测试通过；
- `npm run typecheck`、源码范围 ESLint、完整 `npm run lint`、Next production
  build：通过；构建路由无 `/volunteer`；
- 本批 Python 文件 Ruff check/format：通过；
- V3 schema/runtime/PostgreSQL fixture scoped mypy：3 个文件无问题；
- `tests/test_security.py`：`8 passed`；
- `git diff --check` 与活跃代码旧符号扫描：通过。

未伪装为通过的仓库债务：`ruff check .` 仍有 60 条既有问题，
`ruff format --check .` 仍有 66 个既有文件；`src/graph/academic.py` 全文件 mypy
有 28 个既有错误，均不在本批新增的 evidence-summary QA 区域。Semgrep、
import-linter、Gitleaks、Bandit、Vulture 缺失，未运行。全量 pytest 仍可见既有
`aiosqlite` event-loop shutdown 与 AsyncMock warning。

本补充不宣称“Agent 节点零旧实现”总目标完成。生产 `config/rag/index.yaml`、
generation 激活、真实 P0/PG/PR/PGR、真实 provider E2E 与零旧 checkpoint 仍未证明；
因此当前正式图、alias/迁移 reader 和回滚边界继续保留，未越过生产切换门。

## 2026-07-14 Context Apply 惰性 fallback 字段清理补充

已从 Context Apply policy、budget policy、importance policy/telemetry、配置解析、
`settings.yaml`、trace/SSE 投影和测试 fixture 删除三项没有真实替代分支的字段。importance
失败仍保留 typed reason、error type、sanitized warnings 与 elapsed time，但不再声明
rule-based fallback。预算按整项裁剪、source/budget drop reason、required source 失败、
observe-only importance scoring 与同 provider transport retry 均未改变。

验证结果：相关回归 `303 passed`，配置/安全 `59 passed`，全量后端
`2280 passed, 5 skipped, 11 warnings`；4 个触及模块 scoped mypy、compileall、
触及文件 Ruff check/format、前端 69 项 Vitest、typecheck、完整 ESLint 与 production
build 均通过。活跃代码、官方配置和测试中三项旧字段扫描为零。可选 Semgrep、
import-linter、Gitleaks、Bandit、Vulture 仍缺失，未记为通过。

## 2026-07-14 academic memory Context Engineering 替代进度

`generate_answer` 已从“节点内再次检索 + 旧 builder prepend + 正文 footer”切换为唯一的
Context Engineering provider-input 路径。正式策略为 active：required `rules`，optional
`memory/profile`；memory 最多 6 项/1600 tokens，按 thread 严格匹配，并允许无 relevance
的 conversation summary。显式 ignore/待确认状态不注入 memory，错 thread 候选被拒绝。

Memory/Profile ContextItem 使用稳定 logical ID，因此同一逻辑项内容更新会替换 V3 当前
活跃版本；历史累计仍按真实 dispatch 单调增加。实际 provider dispatch 测试证明
conversation summary、episodic、semantic、profile 与 rules 进入单一 CE block，dispatch
descriptor、SessionContextMemoryLedgerV1 和 Influence Ledger 只记录安全身份、token 与来源
计数，不保存正文。无 memory 时 required rules 路径仍正常调用，不模拟记忆或回退原消息。

本节是替代快照，不是旧层删除结论。`src/context`、`MemoryContextInjection`、旧 prompt
常量、`memory.token_budget` 和旧测试将在全量门通过后的独立提交中删除；当前正式图、
checkpoint migration reader 与生产门保护项继续保留。

替代快照质量门：相关 CE/academic/stream 回归 `397 passed`；全量后端
`2297 passed, 5 skipped, 10 warnings`；前端 23 个 Vitest 文件/69 项测试、typecheck、
完整 ESLint 和 Next production build 均通过；compileall、10 个触及 Python 文件的 Ruff
check/format、4 个 CE 文件 scoped mypy、8 项 security tests 与 `git diff --check` 通过。
`src/graph/academic.py` 全文件 mypy 仍报告 28 项既有错误，均不在本批 `generate_answer`
改动行。全仓 Ruff 仍为 60 项既有错误/66 个既有待格式化文件。Semgrep、import-linter、
Gitleaks、Bandit、Vulture 均缺失，未运行且未记为通过。全量 warning 仍是既有
`aiosqlite` event-loop、AsyncMock 与 pytest cache 权限债务。

## 2026-07-14 legacy memory prompt 独立清理

在替代快照 `ed953ac` 之后，已独立删除：

- `src/context/__init__.py`、`context_builder.py`、`token_manager.py`、`errors.py`；
- `MemoryContextInjection` 及其 `src.memory` public export；
- `MEMORY_CONTEXT_*` 与 `MEMORY_INFLUENCE_EXPLANATION_TEMPLATE` 旧 prompt/footer 常量；
- `memory.token_budget`；
- `tests/test_context_builder.py` 与 `tests/test_token_budget_strict_config.py`。

`src.memory` 的 Episodic/Semantic schema、storage、retrieval、embedding、consolidation 和
public API 均保留；原 public import smoke 已迁到 `tests/test_memory_public_api.py`。新增
absence guard 约束活跃 source/config/tests 不得重新出现旧 package、builder、schema、
prompt 常量或 token budget。该清理不修改正式图、checkpoint、CE ledger、Context Window
V3、compaction 或 provider retry，也不越过仍未满足的生产切换门。

清理质量门：memory/CE/academic 聚焦回归 `161 passed`；首次全量发现 2 条仍要求旧
`total_budget: 4096` 的阶段守卫，已改为验证“1M 模型窗口保留、旧 memory budget 不得
回归”，未删除测试；最终全量后端 `2279 passed, 5 skipped, 7 warnings`。前端 23 个
Vitest 文件/69 项测试、typecheck、完整 ESLint 与 Next production build 通过。compileall、
8 个触及 Python 文件的 Ruff check/format、3 个 retained memory 文件 scoped mypy、8 项
security tests、`git diff --check` 与活跃 source/config/tests 旧符号扫描均通过。

全仓 Ruff 仍为 60 项既有 lint debt；由于本次触及并格式化 `src/memory/schema.py`，全仓
待格式化文件由 66 降为 65。Semgrep、import-linter、Gitleaks、Bandit、Vulture 均缺失，
未运行且未记为通过。warning 仍为既有 AsyncMock、pytest cache 权限及偶发
`aiosqlite` event-loop shutdown 债务。

## 2026-07-14 Supervisor 旧短语路由器独立清理

`supervisor_node` 已由严格结构化链路独占路由语义：
`invoke_structured_llm → SupervisorOutput → validate_supervisor_output`。全仓静态与动态入口
扫描证明五组资源短语表、复数 detector 和单值 wrapper 没有生产调用、LangGraph 注册、
package export、FastAPI 路由、prompt/config 反射或 checkpoint node-ID 依赖，因此按已批准
的 Agent 节点零旧实现计划删除。

旧词表单元测试没有被简单丢弃来消除失败；替代回归使用包含明确资源生成短语的原始 query，
同时返回业务有效的 `unknown/general` QA 结构化结果，并验证 intent、mode、scope、空资源字段、
`needs_mindmap=false` 以及最终 `qa` 路由均不能被 query 二次覆盖。结构化运行时 wiring 测试还
锁定 `validate_supervisor_output` 必须作为 business validator。资源类型规范化、严格 schema、
业务校验与正式路由均保留。

本批质量门：Supervisor/Builder/Manifest `75 passed`；全量后端
`2273 passed, 6 skipped, 7 warnings`；前端 23 个 Vitest 文件/69 项测试、typecheck、完整
ESLint 与 Next production build 通过。compileall、两文件 Ruff check/format、Supervisor
scoped mypy、`git diff --check` 和七个旧符号的活跃代码归零扫描通过。warning 是既有
第三方 pkg_resources、AsyncMock 未 await 与 pytest cache 权限问题。Semgrep、
import-linter、Gitleaks、Bandit、Vulture 仍缺失，未运行且未记为通过。

该提交不删除仍在 import-time 使用的 `_sanitize_valid_intents`，也不触碰正式图、
Parent-Child 生产配置、checkpoint alias/migration reader 或并行 RAG 工作。生产 index、
generation 激活、真实 P0/PG/PR/PGR、真实 provider E2E 与零旧 checkpoint 门仍未满足。

## 2026-07-14 RAG generation-router 残留独立清理

当前候选图在 `search_query_rewriter` 后直接进入 `resource_evidence_planner`，planner 已写入
绑定 runtime 的 `evidence_orchestration_fingerprint`；academic/resource hydration 也已拆成
两个物理节点。此前遗留的 `make_rag_generation_router_node` 只返回固定 route 和相同
fingerprint，当前正式图和候选图都没有注册它。

本批删除 factory/export、`rag_generation_route` transient/TypedDict 字段与孤立 metadata，
并新增 module/state/reset/registry/topology absence 回归；已有 mock planner 测试增加真实替代
fingerprint 断言。严格 evidence schema/business validator、候选图拓扑、trace、repair loop、
完整 transcript/checkpoint 与当前正式图均未改变。

聚焦质量门为 51 项 evidence graph/trace/state/manifest 测试，compileall、4 个触及文件 Ruff
check/format、3 个源文件 scoped mypy、diff check 和旧符号归零扫描均通过；全量后端
`2274 passed, 6 skipped, 11 warnings`，前端 23 个 Vitest 文件/69 项测试、typecheck、完整
ESLint 与 production build 通过。warning 是既有第三方、aiosqlite event-loop、AsyncMock
与 pytest cache 权限债务。早期候选图曾短暂存在该 node ID，因此本代码清理不宣称
checkpoint 已迁移；未知旧 pending node 继续由迁移门阻断。生产 index、generation 激活、
真实四变体、provider E2E 与零旧 checkpoint 门仍未满足。全仓 Ruff 仍为 60 项既有 lint
debt 和 65 个既有待格式化文件；Semgrep、import-linter、Gitleaks、Bandit、Vulture 均缺失，
未运行且未记为通过。

## 2026-07-14 Assessment terminal 与 durable journal 底座

`assessment_final` 已加入 `agent_stream_v2` event/draft/authoritative-terminal 合约，完整支持
sequencer 单终态、容量预留、session completion、Last-Event-ID journal replay 与
`stream_done(terminal_type=assessment_final)`。QA、Resource、interrupt、stopped 和 error 的
既有终态不变。

新增 `AssessmentAttemptJournalV1` / `AssessmentAttemptRecordV1`：thread/request/hash/final/time
全部严格验证，extra 禁止，JSON restore 不做 key 修复。LangGraph durable state 只记录请求
内容 hash、公开 `AssessmentFinalV1` 和 UTC commit time；不记录 submitted answer、原题 answer
key、accepted answers、canonical answer、answer explanation、provider body 或异常正文。

`AssessmentCheckpointIdempotencyExecutor` 通过注入的 load/append callbacks 与存储解耦，按
`(thread_id, request_id)` 本地串行化：相同 hash 重放同一 final，不同 hash 抛显式 conflict，
operation 失败不写记录，每次 append 后必须复读并精确验证 durable record。锁使用引用计数，
最后一个调用结束后删除，避免请求数导致进程内常驻增长。

聚焦 journal/service/private-card/state/stream 回归 `97 passed`；compileall、10 个触及文件
Ruff check/format、6 个源文件 scoped mypy、8 项 security tests 与 diff check 通过。全量后端
`2288 passed, 6 skipped, 9 warnings`，前端 23 个 Vitest 文件/69 项测试、typecheck、完整
ESLint 与 production build 通过。本节仅是 endpoint 前置底座：FastAPI route、真实
structured classifier/generator、checkpoint callback adapter 和 PostgreSQL 多进程原子性/E2E
尚未完成，不能据此删除旧 assessment 节点。全仓 Ruff 仍有 60 项既有 lint debt 和 65 个
既有待格式化文件；Semgrep、import-linter、Gitleaks、Bandit、Vulture 仍缺失，未记为通过。

## 2026-07-14 Assessment attempt SSE endpoint 与严格恢复语义

已新增严格 `POST /threads/{thread_id}/assessment-attempts`。请求只接受
`assessment_attempt_v1`，`request_id` 必须为 canonical UUID；服务端从 thread checkpoint
校验 resource/question 身份并读取私有答案，浏览器与 journal 均不接收 submitted answer、
原题 accepted answers、canonical answer 或 answer explanation。正确结果直接形成
`assessment_final_v1`；错误结果必须经过严格错因分类和 1-3 道完整新练习的 Pydantic 与业务
校验。`assessment_final` 是唯一权威终态，`stream_done` 仍只由 session 追加。

Provider 输入使用 `assessment_private_provider_envelope_v1`。结构化调用显式启用
`sensitive_trace=True`：trace 保留 node/provider/model、阶段、计数、耗时和 error type，但删除
message preview 正文、raw output、provider error body、validation/parsing/business error 正文。
分类 business validator 还会阻断原题私密长文本或完整字段回显；canary 回归证明这些值不会进入
SSE、公开 final、checkpoint journal 或 structured-output trace。

幂等执行现为 pre-dispatch durable claim：`in_progress -> completed|failed`。相同 hash 的成功或
失败跨 service 重建后直接重放，不再次调用 Provider；不同 hash 显式 conflict；取消/崩溃留下
`in_progress` 并返回 recovery-required，禁止自动重做。memory checkpointer 使用进程锁；
PostgreSQL 使用 connection-scoped parameterized advisory lock。真实双连接测试已提供，但当前未
配置 `A3_TEST_POSTGRES_URI`，因此记为 skipped，不记为通过。

本批质量门：compileall、触及文件 Ruff check/format、8 个源文件 scoped mypy、`git diff --check`
通过；assessment/config/security/structured/stream 联合回归 `310 passed, 1 skipped`；全量后端
`2334 passed, 7 skipped, 12 warnings`；前端基线 23 个 Vitest 文件/69 项测试、typecheck、完整
ESLint 与 Next production build 通过。全仓 Ruff 仍有 60 项既有 lint debt 和 65 个既有待格式化
文件；Semgrep、import-linter、Gitleaks、Bandit、Vulture 缺失，均未运行。warning 仍是既有
aiosqlite event-loop、AsyncMock、第三方弃用与 pytest cache 权限债务。

旧 assessment 节点尚未删除：前端题卡提交、真实 Provider E2E、真实 PostgreSQL endpoint/restart
恢复与总生产切换门仍未满足。当前实现不能据此宣称“Agent 节点零旧实现”完成，也不会提前删除
`assessment_result_handler`、`adaptive_practice_responder` 或 placeholder generator。

## 2026-07-14 前端严格题卡与 Assessment SSE 接入

Resource Final V3 的 quiz payload 现在在进入消息与 localStorage 之前严格解析
`exercise_items` 和 `exercise_artifact.items`：两份公开题卡必须完全一致，题卡只允许
`schema_version/question_id/question_type/level/question/choices/tags`。答案、answer key、
accepted answers、原解释、pitfall、match mode 或任意未知字段都会使整个 Resource Final
被拒绝。恢复聊天时也会重新解析已保存的 Resource Final 并重建资源投影，不再信任旧的
`Message.exercise` 对象；没有严格 V3 payload 的旧 exercise 投影不会恢复。

新增前端 `assessment_attempt_v1` / `assessment_final_v1` / 公开题卡严格解析器和独立
assessment client。客户端保留调用方生成的 request ID，只 POST 一次；断线时只允许调用方
显式提供的 `GET /streams/{stream_id}` + `Last-Event-ID` 重放，不会重新 POST 或偷偷更换
request ID。共享 `agent_stream_v2` client/SSE parser 负责 UTF-8、CRLF、SSE id/type 与断线处理；
assessment 层另外校验 start → 单一 `assessment_final|stream_error` → 匹配 `stream_done`、
sequence gap、同 sequence 冲突、thread/request/resource/question/time 身份、payload-hash 格式和
终态业务真值。HTTP 409 显式进入 conflict；HTTP body、remote message 与 submitted answer
均不进入错误对象、日志或持久化。

题卡组件实现 `idle/editing → submitting → correct|incorrect|failed|conflict`，支持 free-text、
single-choice、错因与 1-3 道自适应练习。一次提交期间用同步 ref 阻止双击；失败后只能由用户
显式重试。主聊天请求与 assessment 请求互斥，其他题目在评估期间禁用；切换、清理或新建
thread 会 abort 当前 assessment。submitted answer 只存在于当前组件状态和一次 POST body，
不进入 `Message`、活动日志、console 或 localStorage；公开 adaptive task 的答案/解释按正式
`assessment_final_v1` 契约展示，不属于原题私有 answer key。

本批质量门：前端聚焦回归 7 个文件/61 项通过，全量 Vitest `27 files / 117 tests`、
typecheck、完整 ESLint 与 Next production build 通过；生产构建路由仅有 `/`、`/analytics`、
`/onboarding`、`/print/review-doc`，没有 `/volunteer`。后端 compileall 与全量 pytest
`2334 passed, 7 skipped, 10 warnings` 通过；首次 300 秒 pytest 命令仅超时、未记为通过，
随后以更长时限完整重跑通过。`git diff --check` 通过。全仓 Ruff 仍有 60 项既有 lint debt
和 65 个既有待格式化文件；本批没有 Python 源码改动。Semgrep、import-linter、Gitleaks、
Bandit、Vulture 均缺失，未运行且未记为通过。warning 仍是既有第三方弃用、aiosqlite
event-loop、AsyncMock 与 pytest cache 权限债务。

前端题卡提交门现已满足，但旧 assessment 节点删除门仍未满足：真实 Provider E2E、真实
PostgreSQL endpoint/restart recovery、生产 index/generation/四变体证据、checkpoint 迁移与
零旧 checkpoint 扫描仍缺失。因此本批没有删除 `assessment_result_handler`、
`adaptive_practice_responder`、placeholder generator、正式旧图或迁移 reader。

## 2026-07-14 候选 Parent-Child 图安全学习路径、自动推荐与真实 fan-in 闭环

本批只改候选图，没有切换 `app.py` 当前正式图。候选资源链现在为：

`search_query_rewriter → learner_path_planner → resource_evidence_planner →`
`证据检索/判定/分配 → resource_worker fan-out → resource_bundle_aggregator →`
`resource_recommendation_auto → resource_bundle_output → END`。

- `EvidenceOrchestrationRuntime` 现在必须显式注入 `LearningGuidanceRuntime`；builder
  不创建画像、历史、路径或推荐依赖，也没有默认 runtime。调用方还必须提供 lowercase
  SHA-256 runtime fingerprint。该值、Provider 投影步数/字符上限及投影 schema 均已
  纳入候选图 orchestration fingerprint，策略改变不会复用旧候选身份。
- `learner_path_planner_output_v1` 在进入证据 planner 和 study-plan planner 前都会重新
  校验 schema、`request_id/user_id/subject` 绑定。checkpoint JSON 解码后必须与输入的
  canonical JSON 完全一致；数字、日期或字段 coercion 会被拒绝，不存在 alias repair。
- path/recommendation checkpoint output 原子携带 guidance runtime fingerprint、投影
  policy fingerprint 与实际 steps/chars 上限。Evidence Planner 和候选 finalizer 必须
  与当前注入 runtime 完全匹配；runtime 或 policy 改变后不能消费旧输出。
- 完整路径继续留在 checkpoint 供恢复和审计；Provider 只能消费与完整路径逐字段重建并
  完全一致的 `learner_path_provider_projection_v1`。投影不含 `request_id`、`user_id`、
  `profile_signal_ids` 或 `history_ids`，并受显式 steps/chars 硬上限约束；超限或投影篡改
  会 typed fail-fast，不截断、不降级。
- 路径节点从已验证的 `retrieval_plan[].subject` 重新计算作用域。仅单科且等于主科目时
  才加载画像/历史和执行 path engine；多科或不匹配写入显式
  `unsupported_subject_scope`。证据 planner 与 study-plan 消费端会再次校验，不能把旧的
  单科 available 路径注入多科请求。
- 新路径不会写入旧 `curriculum_context`。该字段只保留给尚未切换的正式旧图；候选
  study-plan 直接消费严格 Provider-safe projection；完整输出与投影必须成对存在，旧
  curriculum context 不能掩盖半个新契约。
- `resource_bundle_aggregator` 只在内存中构造一次无推荐的严格 V3 投影以取得真实稳定
  `resource_id`，对外只写最小 `recommendation_resource_context`；它不写 Message、
  `resource_final_v3`、workspace、journal 或 SSE。
- planner 写入的 `evidence_orchestration_fingerprint` 现由 retrieval router 起的每个
  evidence 节点重新校验；资源 preflight、worker、aggregator、recommendation 与
  finalizer 也由候选 builder 的 runtime guard 包裹。该指纹包含 policy、prompt/schema、
  guidance runtime/policy、Parent-Child handoff、profile 与 web timeout，跨进程恢复时任一
  语义变化都会在继续 Provider/资源工作前阻断。
- 自动推荐只接收真实 success/partial-success 资源。失败与证据阻断资源不进入推荐
  上下文；多科目或主科目不一致返回显式 `unsupported_subject_scope`，不把首个科目
  静默套给所有资源。自动 recommendation item 的 `resource_id` 必须是已验证 bundle
  中的真实生成资源且出现在 source IDs；Resource Final V3 和前端严格保留该 ID，并校验
  ID 与资源类型，不能产生不可操作的伪推荐。
- 最终节点重新严格校验 recommendation output 的 mode、请求/用户/科目身份及资源
  引用，再一次性构造唯一 Resource Final V3。实测推荐前后单资源 `resource_id` 保持
  不变，最终 `payload_hash/resource_final_id` 按推荐内容变化；quiz 私有答案不进入聚合
  上下文或终态。
- 真实编译的最小 LangGraph 已覆盖双 `Send` fan-in：mindmap 成功、quiz 失败时 worker
  执行两次，但 aggregator、recommendation 与 finalizer 各一次且只有一个
  `partial_success` Resource Final V3；空 task 只产生一次 controlled stop，推荐异常在
  finalizer 前 fail-fast。
- 缺用户、科目、画像、历史或真实资源时不会生成中性分数或空成功。当前公共 V3 仍
  只有 `recommendations[]`，因此 unavailable reason 被前置写入公开 summary；机器可读
  `recommendation_outcome` 仍是生产切换前必须补齐的公共契约门。

本批不宣称显式推荐入口已完成。Supervisor 尚无严格 recommendation action，且当前
Resource Final V3 的 success 要求至少一个真实资源，不能诚实表达 recommendation-only
终态。显式入口必须在独立 spec 中同时决定路由和权威终态，不能伪装成资源成功。
当前建议选择独立 `recommendation_final_v1`：它可绑定真实 catalog/KG candidate snapshot，
不必削弱 Resource Final V3“成功必须含真实生成资源”的终态不变量；若选择扩展 V3，则需
显式批准 recommendation-only success 语义及相应迁移。

本阶段候选实现提交：`eee3d8e0b5042c16b1d975a2bea568652a6a2271`。

验证结果（完成安全投影、恢复绑定、推荐资源绑定与真实 fan-in 后重新运行）：

- 路径/候选图/study-plan/fan-in/Resource Final 聚焦：`167 passed`；
- 候选图、资源 V3、stream/session、manifest 与 security 扩大回归：`274 passed`；
- 全量后端：`2383 passed, 7 skipped, 7 warnings`；warning 是既有第三方弃用、
  AsyncMock coroutine 与 pytest cache 权限债务；
- `python -m compileall -q src tests app.py`、冷导入和 `git diff --check`：通过；
- 本批 14 个触及 Python 文件 Ruff check/format：通过；
- scoped mypy（`--follow-imports=skip --ignore-missing-imports`）：9 个触及源文件通过；
  普通完整依赖跟随命令两次在 120 秒超时，因此未记为通过；
- 前端：Vitest `27 files / 118 tests`、typecheck、完整 ESLint、Next production build
  均通过；build 路由仍无 `/volunteer`；
- 全仓 Ruff 仍为 60 项既有 lint debt，format check 仍为 65 个既有文件；
- Semgrep、import-linter、Gitleaks、Bandit、Vulture 缺失，均未运行且未记为通过。

生产门重新核对基于候选实现提交 `eee3d8e0`：`config/rag/index.yaml` 与
`data/knowledge_graph.yaml` 仍不存在，`activation_enabled=false`，当前进程没有
PostgreSQL/DeepSeek/Tavily/RAG reranker 凭据。`continuation_pc51` 仍标记
`status=completed_experimental`、`activation_allowed=false`、
`evaluation_eligible=false`、`experimental_only=true`、
`activation_prohibited=true`，且 reranker probe 的
`relevant_above_irrelevant=false`。真实 Provider E2E、真实 PostgreSQL restart、
P0/PG/PR/PGR 生产证据和零旧 checkpoint 扫描均未完成，因此正式旧图、迁移 reader、
alias 与旧节点实现继续保留，本批没有越过删除门。

## 2026-07-15 D8-A 显式推荐 served graph 与 Recommendation Final V1

显式推荐现已完成代码级端到端接入，不再复用 Resource Final V3 或伪装成资源生成：

`supervisor -> resource_recommendation_explicit -> recommendation_final_output -> END`

Supervisor 只接受 `academic/recommendation` 的严格组合，资源类型、`qa_scope` 必须为空，
且不得要求 live verification。当前 served graph、Parent-Child 候选图和资源证据候选图
都注册同一显式分支；`LearningGuidanceRuntime` 成为 served/P0 graph factory 的必填依赖，
没有默认 runtime 或 fallback。缺 `user_id` 优先返回 `missing_user_id`；零/多科目分别返回
`missing_subject`/`unsupported_subject_scope`；只有同 thread 且已严格绑定单科目的 workspace
continuation 才能补充当前空科目。

`recommendation_final` 已进入 `agent_stream_v2` 的单权威终态、容量预留、journal、断线重放、
active-run、checkpoint status、OpenAPI 与前端 LiveTurn/恢复链路。前端严格重算 final ID、
payload hash 与 catalog snapshot，最终只提交已验证推荐卡。五个 SSE 路由的 HTTP 200
OpenAPI content 现在只包含 `text/event-stream`。

提交前审查同时修复了两个与新 RAG 无关的旧终态漏洞：

- QA Final 现在重新验证 JSON schema、业务规则、runtime thread/request、payload hash 与 QA ID；
  malformed/tampered/wrong-thread QA 不能被忽略，也不能让另一个终态获胜。
- QA、Resource、Recommendation 任一完成态若 checkpoint 写入失败，均只发送
  `terminal_checkpoint_persist_failed`，不再发送 completed activity 或权威 final，也不会把
  未落盘 final 投影到 active-run。

实现提交为 `9d6d69f`，前端独立提交为 `225b084`，生产 KG artifact 提交为 `c8e20f8`。
验证：D8 聚焦 `654 passed`；全量后端 `2590 passed, 7 skipped`；前端 `31 files / 142 tests`、
typecheck、ESLint、Next build 通过；compileall、36 个触及 Python 文件 Ruff check/format、
12 个源文件 scoped mypy、import-linter（333 files / 2,051 dependencies，3 kept）、
Gitleaks staged scan 与 diff check 通过。全仓仍有 45 个既有 Ruff lint 与 57 个既有 format
文件；Bandit 的 2 个 scoped finding 和 Semgrep 的 55 个词法 finding 均来自既有代码，因此
未记为 clean pass。

当前仍不能宣称真实用户稳定获得 `available` 推荐：D5 adapters 已是严格 readers，但缺少
`profile.extra.learning_guidance_v1` 和 episodic `metadata.learning_guidance_v1` 的正式 writers，
且旧 episodic writer 仍优先用 `thread_id`。这部分可不等待新 RAG 并行修复。生产图整体切换、
checkpoint 清除和旧 served graph/节点删除仍受 `config/rag/index.yaml`、四变体 gold、真实 RAG
Provider E2E 与 rollout activation 门约束；当前 `activation_enabled=false`，本批没有越门。

## 2026-07-15 Profile/History Writers 与 Onboarding V2 生产收口

本批建立了可持久化的 profile/history 写链路，并把 onboarding 改为只消费生产
Learning Guidance catalog。公开请求保持 strict JSON `list`；Python tuple、set、generator、
字符串及被篡改的既有模型实例均拒绝。通过显式 compiler 后才投影为 frozen tuple 内部合同，
没有 `BeforeValidator`、alias normalization、schema repair 或 silent default。

- Profile writer 将完整命令 source hash、request receipt 与 topic binding 原子写入 SQLite。
  identical replay 不改时间戳；request/source/top-level drift 明确冲突。既有但未绑定的 profile
  只在 `onboard_v2` 来源下原子覆盖 nickname、grade、dislikes，同时保留原 skills、goals、
  learning style、behavior、observations、tags、created_at 与非保留 extra。
- History writer 只接受 journal 已完成的真实 assessment；普通聊天、资源生成与一般 behavior
  不会伪装成学习成效。提交答案、answer key、adaptive answer 与 request hash 不进入长期 history。
  history ID namespace 改为 insert-once protected fact；普通 upsert、consolidation、forgetting 与
  direct delete 均不能覆盖或删除。
- strict episodic insert 在同一 `BEGIN IMMEDIATE` 内完成 row decode、类型敏感 canonical JSON
  比较和 commit；任何验证失败先 rollback。`True/1/1.0`、`0.0/-0.0`、坏 JSON、重复 key、
  NaN/Infinity 与 `1e400` 均 fail-closed，内容派生异常不保留原始 JSON cause。
- Growth 查询先在 SQL 层以 case-sensitive protected prefix 过滤，再执行 500 条 limit；501 条
  更晚的普通记录与大小写近似 prefix 都不能挤掉权威 assessment。
- Assessment checkpoint 升为显式 `assessment_checkpoint_resources_v2` / resource V2；只对带
  V1 schema tag 的旧快照执行严格、可审计迁移并显式写入 `learning_guidance_binding=None`，
  不用缺字段默认值。history 持久化瞬态失败标记为 recoverable；同一 request 可在同进程新建
  recovery stream，assessment journal 只 replay，不重复分类或生成工作，成功后才发送
  `assessment_final`。
- generic episodic memory 的 user/hashed-thread owner 切换因缺少存量归属迁移而从本批撤回；
  继续使用既有 raw-thread key，并把缺失 thread 的旧 `unknown` sentinel 改为显式错误。该 owner
  cutover 必须另立 migration mini-spec，不能双读猜归属。

前端 onboarding 页的大差异是必要的数据模型替换：删除静态 subjects、`/subjects`、localhost
fallback 和自定义科目，改为 subject→KG topic catalog；每个 topic 显式填写 level、confidence、
goal、importance、progress，并持久化冻结 per-user request ID/payload 供安全重试。没有夹带视觉
redesign、组件体系迁移或无关页面重排。`useUser` 只把 404 判为 missing；5xx/transport 显示
unavailable，不再误重定向。`tsconfig.tsbuildinfo` 从版本库删除并由 `.gitignore` 排除；
`next-env.d.ts` 已恢复基线。

本批最终 focused 门：

- 后端 writers/onboarding/storage/assessment/stream 聚焦：`264 passed`；security：`8 passed`；
- 前端 onboarding/use-user：`5 files / 41 tests`；typecheck 与完整 ESLint 通过；本批代码状态的
  Next production build 已通过，路由仍为 `/`、`/analytics`、`/onboarding`、
  `/print/review-doc`；
- `python -m compileall -q src tests app.py`、changed-path Ruff check、`git diff --check`：通过；
- 23 个新合同/核心触及文件 scoped Ruff format-check：通过；为遵守 diff-risk 门，本批没有对
  既有未格式化的大型 `app.py`/graph 文件做全文件机械格式化；
- 两组 scoped mypy（contracts/writers/storage 9 files；assessment/stream 5 files）：通过；
- import-linter：338 files / 2,098 dependencies，3 kept / 0 broken；targeted Bandit：通过；
- diff-only secret-shaped additions 与 forbidden structured-output pattern 扫描：无命中；
- Semgrep、Gitleaks 未安装，明确记为未运行；未用其他扫描冒充通过。

rollout 继续保持 disabled。本批没有修改 Parent-Child RAG build/gold artifact，没有调用或输出
Provider secret/DB URI，也没有清空或迁移现有 PostgreSQL checkpoint。
