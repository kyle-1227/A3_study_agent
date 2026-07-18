# A3 Study Agent 比赛最终文档索引

本目录是赛题提交、演示验收和生产化复核的统一入口。比赛演示 runtime source / integration 为 `ca3960a`，已由 `main` 包含并发布；`707d79806364d95fd300b21d0cb93411f592d67a` 仅保留为两轮浏览器历史实测证据。
SSE `eed2139`、Evidence `4a91f68` 与 RAG `f53a710` 已分别以 `d7f5802`、`cde3e59`、`fa0f2dc` 集成，受控 fallback 治理以 `9cb929c` 集成。最终 Docker 镜像与基础 readiness 已复验；完整六场景和人工教育效果仍未验收。本文只把可复核证据写成已完成事实。

## 文档清单

- [系统开发说明书](system_development.md)：需求分析、架构、画像、多智能体、七类资源、路径推荐、辅导评估、防幻觉、安全与生产边界。
- [测试说明书](test_report.md)：测试分层、已记录门禁、复现命令、真实 canary 验收标准和未覆盖项。
- [部署说明书](deployment_guide.md)：Docker Compose 部署前置条件、严格配置、健康检查与恢复边界。
- [第三方软件与 AI 工具说明](third_party_notices.md)：直接依赖的名称、来源、许可证及需人工确认的高风险项。
- [生产部署运行手册](../runbooks/production_deployment.md)：运维级启动、PostgreSQL 重启/回放和浏览器 canary 操作规程。
- [Parent–Child RAG 运行手册](../runbooks/parent_child_rag_local_build.md)：索引构建、验证和发布边界。

## 赛题要求对照

| 赛题要求 | 当前实现与证据 | 结论 |
| --- | --- | --- |
| 对话式动态画像，不少于 6 个维度 | `src/profile/` 定义技能、学习风格、目标、行为、智能体观察、不喜欢内容、标签/扩展信息；对话抽取后按证据和置信度更新 | 已实现；演示需展示首次建档与后续更新 |
| 多智能体协同，至少 5 类资源 | 图编排包含监督、路径、证据规划、本地检索、网页研究、证据裁判、资源生成和 QA 角色；生产资源为学习计划、思维导图、测验、复习文档、代码练习、视频脚本、视频动画 7 类 | 已实现 |
| 个性化路径规划与资源推荐 | 路径规划把画像、历史/评估和 `KnowledgeGraphV1` 的 source-backed topic 绑定，再由资源证据规划和生成链路执行 | 已实现；推荐质量仍需真实用户评估 |
| 智能辅导与学习效果评估 | QA 路径、测验、学习历史、assessment 绑定和画像增量更新形成闭环 | 已实现基础闭环；不是教育效果的临床或统计学证明 |
| 流式输出与生成进度 | `agent_stream_v2` SSE、`EvidenceProgress`、序号、显式终态、Last-Event-ID 回放和 thread status 恢复 | 已实现 |
| 防幻觉与内容安全 | 本地/网页证据需求、证据裁判、有限修复、结构化输出和业务校验均采用失败即停止；敏感信息不得进入日志/报告 | 机制已实现；不能保证“零事实错误”，真实内容验收仍必需 |
| 开源项目及 AI 工具显著标注 | 本目录列出直接依赖、来源和许可证，并单列 PyMuPDF、Psycopg、课程资料和外部服务 | 文档已补齐；分发前仍需许可证负责人签字 |
| Docker 一键部署 | Compose 可统一启动前后端与 PostgreSQL，并执行严格 readiness | 有条件满足：必须另行提供获授权课程资料、密封索引和私密配置；纯 Git checkout 不自包含 |
| 系统开发、测试和部署文档 | 本目录及 `docs/runbooks/` 提供完整入口 | 已补齐 |

## 当前生产身份

- 已发布比赛演示 runtime source / integration：`ca3960a`；`main` 已包含该提交；浏览器历史实测 runtime：`707d79806364d95fd300b21d0cb93411f592d67a`。
- 已集成车道：SSE `eed2139 -> d7f5802`（36 files / 208 tests，ESLint/typecheck/build 通过）、Evidence `4a91f68 -> cde3e59`（64 passed）、RAG `f53a710 -> fa0f2dc`（总控 48 passed / 1 skipped；车道 50 passed / 1 skipped）。
- 唯一对外服务的检索/证据路径：resource-aware PGR。
- 密封 generation：`pc_20260715_98336c2_55`。
- generation manifest：`db579d40d1f4b79882f495277026e8fccfbfb816fbb150998e47753eec470218`。
- 课程知识图谱：`KnowledgeGraphV1`。
- 知识图谱数据版本：`2026.07.15-source-groups-v1`。
- 知识图谱 artifact：`c504e41ef2e481b30b940ac6cb04f661401f7907d1690efeafc1ed14680fa0b5`。
- Evidence orchestration：`9dec07d4f097bae80bbf815bd53494e4e8045b15e536d0fc38daa3b4da2e032b`。
- 最终 Docker：backend `sha256:6f7108ce1af9d5124c1e39a1c241d50eea7b55cb591ef784bc965bfe97247d48`、frontend `sha256:a650fd112b6469236def418b4ea136d702b46dbd572a3b389e829b4bf547de5e`；三容器 `healthy`，`/`、`/onboarding`、`/health/ready` 均为 HTTP 200。
- Evidence 补充策略为初始检索加最多 3 轮补搜，总任务 24、ledger 72；required evidence 仍须完整，partial 不会转为成功。
- code-practice 生成流式运行，严格 reviewer 使用独立 non-streaming 配置并保留结构化与业务校验。
- PostgreSQL checkpointer 使用严格配置、健康检查和重连预算明确的连接池；启动失败仍 fail-closed，运行中不会降级到 `MemorySaver`。
- 可变画像/记忆状态使用独立 `app_state:/app/.runtime_state` 卷；旧 `/app/data/*.db` 仅在新目标不存在时原子迁移，课程资料仍只读。
- Evidence `4a91f68` 仅用同一 Provider/模型对失败的 resource+subject partition 做有界 reask，继续执行完整校验，且 reask 不自行判断 blocked；聚焦测试 64 passed。
- RAG `f53a710` 仅对同一 rerank endpoint 做有界 complete-score batch split；所有候选必须有完整 score，RRF-only 与 partial scores 均禁止。
- SSE `eed2139` 仅在 transport 或 HTTP 410 后恢复一次同用户/线程/请求的权威 `completed`、`failed` 或 `stopped`；pending、legacy、identity drift、sequence gap 与合同错误显式失败，也不重新调度 Graph。
- P0、PG、PR、PGR 是离线评估变体，不是四条生产流量路径。
- 六场景数据是 smoke authoring，不是正式 Gold，也不代表完成人工评审。
- 真实 Docker/Provider/浏览器链路已连续完成两轮 code-practice，均为 `production_success=true`；完整六场景与人工学术/教育效果验收仍未完成。

## 提交前阻断项

以下事项不能靠补写文档自动消除：

1. **赛题第 49 行相关要求**：赛题“实现条件”要求开发过程中使用的其他 AI 辅助工具选用科大讯飞相关工具。当前可审计仓库记录能确认 OpenAI Codex 辅助开发，但不能证明使用过科大讯飞 AI Coding 工具。不得虚构使用记录；参赛负责人必须向组委会确认适用口径，并在提交前提供真实、可核验的合规证据或调整开发流程。
2. **PyMuPDF 许可**：该直接依赖采用 AGPL-3.0/商业双许可。项目根 MIT 许可证不会覆盖其义务；发布镜像、源代码或对外服务前必须由许可证负责人确定合规路径。
3. **课程资料权利**：课程文档和密封 Parent–Child 索引不随干净 checkout 自包含。提交、展示和分发前必须确认资料、派生索引、截图及生成内容的授权范围。
4. **真实验收**：两轮 code-practice 已通过，但仍须完成最终镜像与 PostgreSQL 重启回归、其余六场景覆盖和人工内容抽检。单一场景与单元测试不能替代这些证据。
5. **公网生产边界**：当前定位是可信本地演示。公网多租户鉴权、租户隔离、滥用防护和正式运维值守尚未闭环，不应直接暴露到互联网。
6. **静态安全基线**：本次全仓 Ruff 与 Bandit 门禁并非全绿；Bandit 报告 15 项 High、7 项 Medium、24 项 Low。详见[测试说明书](test_report.md)。公共生产声明前必须逐项复核、修复或提供可审计理由。
7. **演示材料**：当前 Git 提交不包含比赛演示 PPT 或 7 分钟内演示视频；它们由参赛负责人按最终演示版另行制作和验收。
8. **可分发知识库**：运行环境使用外部获授权课程资料与 sealed index；提交包能否携带至少一门完整课程知识库，必须由资料/许可证负责人确认后执行，不能直接复制私有挂载。

## 提交物口径

提交包可以包含源代码、无密钥配置模板、本文档集和演示材料；不得包含真实 `.env`、API key、Authorization、完整数据库 URI、Provider 请求/响应正文、未获授权课程资料或含敏感正文的调试产物。所有截图和报告在提交前必须再次脱敏。
