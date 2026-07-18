# A3 Study Agent 部署说明书

## 1. 部署结论

项目支持 Docker Compose 统一构建并启动 PostgreSQL、FastAPI 后端和 Next.js 前端。这里的“一键部署”是**满足外部资产和私密配置前置条件后的单命令启动**，不是纯 Git checkout 开箱即用。

`main` 已发布到 `b8f9504`；两轮 code-practice 真实 canary 只证明其历史 runtime。SSE `eed2139`、Evidence `4a91f68` 与 RAG `f53a710` 仍待治理和最终 Docker 重建，因而当前没有最终 integration SHA；PostgreSQL-only restart、完整六场景与人工内容验收仍未完成。本文不读取或展示真实 `.env`。

## 2. 环境与外部资产

### 软件

- Docker Desktop 或 Docker Engine；
- Docker Compose v2；
- 建议至少满足仓库本地开发基线：Python 3.11+、Node.js 20.12+（仅 Compose 使用者不必在宿主机运行应用进程）。

### 必须另行提供的资产

1. 具备提交、展示和处理授权的课程资料目录；
2. 已密封的 Parent–Child 索引目录，其中 registry primary 指向
   `pc_20260715_98336c2_55`，shadow 为空；
3. 与该 generation 一致的 manifest、`KnowledgeGraphV1` 和
   `2026.07.15-source-groups-v1` 身份，精确 fingerprint 为：
   - manifest：`db579d40d1f4b79882f495277026e8fccfbfb816fbb150998e47753eec470218`；
   - KG artifact：`c504e41ef2e481b30b940ac6cb04f661401f7907d1690efeafc1ed14680fa0b5`；
   - Evidence orchestration：`6274c8ac2b0e70828d7e5f64f72ed8f2b9ab36ae8683adcf0b274d60df277b01`；
4. 仅存放在忽略文件或部署平台 secret store 中的私密配置。

课程资料和密封索引可能因版权、体积与敏感性不进入 Git。部署负责人应通过受控渠道交付，校验来源和完整性；不得用空目录、旧 Flat 索引或临时生成内容冒充。

## 3. 配置

从模板创建本地忽略文件，但不要提交：

```powershell
if (-not (Test-Path -LiteralPath '.env')) {
  Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
$env:A3_ENV_FILE = (Resolve-Path '.env').Path
```

必须配置的名称如下；本文档故意不提供值：

| 名称 | 用途 |
| --- | --- |
| `A3_ENV_FILE` | shell 级 env 文件绝对路径；Compose 不静默选择其他文件 |
| `DEEPSEEK_API_KEY` | 严格配置引用的对话模型凭据 |
| `RAG_EMBEDDING_API_KEY` | embedding 凭据，名称必须精确一致 |
| `RAG_RERANKER_API_KEY` | reranker 凭据，名称必须精确一致 |
| `TAVILY_API_KEY` | 允许的网页研究凭据 |
| `POSTGRES_PASSWORD` | PostgreSQL 强密码 |
| `NEXT_PUBLIC_API_URL` | 浏览器访问的后端地址 |
| `COURSE_DATA_HOST_PATH` | 获授权课程资料宿主路径 |
| `PARENT_CHILD_INDEX_HOST_PATH` | 密封 Parent–Child 索引宿主路径 |
| `PARENT_CHILD_GENERATION_ID` | 必须为 `pc_20260715_98336c2_55` 并与 registry primary 一致 |

索引配置必须保持
`EMBEDDING_API_KEY_ENV=RAG_EMBEDDING_API_KEY` 和
`RERANKER_API_KEY_ENV=RAG_RERANKER_API_KEY`。不要把 API key、Authorization、完整数据库 URI 或 Provider body 放入命令行、截图、日志或报告。

## 4. 预检与单命令启动

从仓库根目录执行：

```powershell
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE config --quiet
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE up --detach --build --wait
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE ps
```

第二条命令是准备完成后的统一启动入口。Compose 对课程资料和密封索引都使用 long-syntax 只读 bind，并设置 `bind.create_host_path=false`；D:/E: 路径不存在时必须失败，不能自动创建空目录。运行时 Chroma 快照使用独立可写 volume，生成 artifact 使用持久化 volume。后端镜像包含 Chromium 和 ffmpeg，以支持视频动画资源。
课程资料和密封索引都保持只读；`app_state` 持久卷专门保存 `/app/.runtime_state` 下的画像与记忆 SQLite。启动迁移不会覆盖已存在的新数据库，迁移或 schema 初始化失败会直接阻断 backend readiness。


正式比赛镜像还应从干净 HEAD 重建，并在 backend/frontend 镜像上写入同一
`org.opencontainers.image.revision` 标签。精确命令与标签检查见
[生产部署运行手册](../runbooks/production_deployment.md)。

任何缺失变量、宿主路径、generation/manifest/KG 身份或 PostgreSQL readiness 都应使部署失败。不得为了“启动成功”改用静默默认值、旧 RAG、其他 Provider 或其他模型。

## 5. 服务验收

先执行不触发 Provider 的检查：

```powershell
Invoke-WebRequest http://localhost:8000/health/live -UseBasicParsing
Invoke-WebRequest http://localhost:8000/health/ready -UseBasicParsing
Invoke-WebRequest http://localhost:8000/graph/manifest -UseBasicParsing
Invoke-WebRequest http://localhost:8000/subjects -UseBasicParsing
Invoke-WebRequest http://localhost:3000 -UseBasicParsing
```

`/health/live` 只证明进程可响应。`/health/ready` 还必须返回：

- `health_ready_v3` 与 `status=ready`；
- `checkpointer_type=postgres`；
- `deployment_mode=active`；
- `rollout_activation_enabled=true`；
- `rollout_shadow_enabled=false`；
- 与 production config 一致的 graph、KnowledgeGraph、generation manifest 和 evidence orchestration 身份。

当前实测身份必须精确匹配 generation 55、manifest
`db579d40d1f4b79882f495277026e8fccfbfb816fbb150998e47753eec470218`、KG
`c504e41ef2e481b30b940ac6cb04f661401f7907d1690efeafc1ed14680fa0b5` 和 Evidence
`6274c8ac2b0e70828d7e5f64f72ed8f2b9ab36ae8683adcf0b274d60df277b01`。

生产 checkpointer 必须保持 PostgreSQL-only：连接池会在借出连接前做健康检查并在预算内替换失效连接，初始化或重连失败必须显式暴露，不能转用内存状态。readiness 恢复只是第一层检查；数据库单独重启时 backend/frontend 容器 ID 必须保持不变，并继续验证历史 thread/status、SSE journal、Context 注入和 artifact 下载。
三条生产级受控恢复均须单独验收且不得降低质量：Evidence `4a91f68` 只以同一 Provider/模型对失败的 resource+subject partition 有界 reask，并且不自行判断 blocked；RAG `f53a710` 只在同一 endpoint 做 complete-score batch split，禁止 RRF-only 与 partial scores；SSE `eed2139` 只在 transport 或 HTTP 410 后读取一次身份匹配的权威终态。任一路径都不能切 Provider、模型、generation 或旧 Flat RAG，也不能把 partial evidence 或 pending status 写成成功。


`/subjects` 只应暴露五个生产学科：大数据、计算机、机器学习、数学和 Python；内部目录不能被当作学科。

以上仍只是 readiness。两轮历史 code-practice 已验证 Last-Event-ID 回放、刷新恢复、请求漂移冲突和 DOCX/Markdown/Python 下载，但最终比赛/部署验收还要按[生产部署运行手册](../runbooks/production_deployment.md)完成 PostgreSQL-only restart、其余六场景覆盖并人工抽检学术内容。未覆盖项必须写“未完成”，不能由单一场景外推为通过。

## 6. 停止与恢复

常规停止：

```powershell
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE down
```

常规停止不要附加 `--volumes`，否则会删除 PostgreSQL 和 artifact volume。保留 generation 55、registry、Flat 53、根 `chroma_store`、成功报告和 Gold authoring checkpoint；它们的清理需要单独审批。

当前请求路径只服务 PGR。请求失败不会自动切换到 Flat 或其他 generation。恢复必须在停止写入后显式还原经过校验的外部索引/registry 备份并重启；不得把 Candidate/PGR 失败伪装成旧 RAG 成功。

## 7. 生产边界

该部署当前只面向可信本地比赛演示。开放公网前至少需要完成多租户认证与授权、租户数据隔离、速率/滥用控制、secret 管理、备份恢复演练、监控告警和值守流程。详细操作与证据保存规则见[生产部署运行手册](../runbooks/production_deployment.md)。
