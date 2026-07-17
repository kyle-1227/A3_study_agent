# A3 Study Agent 部署说明书

## 1. 部署结论

项目支持 Docker Compose 统一构建并启动 PostgreSQL、FastAPI 后端和 Next.js 前端。这里的“一键部署”是**满足外部资产和私密配置前置条件后的单命令启动**，不是纯 Git checkout 开箱即用。

本次纯文档提交没有读取真实 `.env`，也没有执行 Docker、Provider 请求或真实浏览器 canary，因此不构成已部署或已验收证明。

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
   `2026.07.15-source-groups-v1` 身份；
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

第二条命令是准备完成后的统一启动入口。Compose 将密封索引只读挂载，把运行时 Chroma 快照放入独立可写 volume，并把生成 artifact 放入持久化 volume。后端镜像包含 Chromium 和 ffmpeg，以支持视频动画资源。

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

`/subjects` 只应暴露五个生产学科：大数据、计算机、机器学习、数学和 Python；内部目录不能被当作学科。

以上仍只是 readiness。最终比赛/部署验收还要按[生产部署运行手册](../runbooks/production_deployment.md)执行六场景真实网页 canary、PostgreSQL 重启、Last-Event-ID 回放、刷新恢复、请求漂移冲突和 artifact 下载检查，并人工抽检学术内容。未执行时必须写“未执行”，不能写“通过”。

## 6. 停止与恢复

常规停止：

```powershell
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE down
```

常规停止不要附加 `--volumes`，否则会删除 PostgreSQL 和 artifact volume。保留 generation 55、registry、Flat 53、根 `chroma_store`、成功报告和 Gold authoring checkpoint；它们的清理需要单独审批。

当前请求路径只服务 PGR。请求失败不会自动切换到 Flat 或其他 generation。恢复必须在停止写入后显式还原经过校验的外部索引/registry 备份并重启；不得把 Candidate/PGR 失败伪装成旧 RAG 成功。

## 7. 生产边界

该部署当前只面向可信本地比赛演示。开放公网前至少需要完成多租户认证与授权、租户数据隔离、速率/滥用控制、secret 管理、备份恢复演练、监控告警和值守流程。详细操作与证据保存规则见[生产部署运行手册](../runbooks/production_deployment.md)。
