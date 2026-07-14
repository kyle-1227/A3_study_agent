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
