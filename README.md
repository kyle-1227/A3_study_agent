# A3 Study Agent

[English](README_en.md)

A3 Study Agent 是面向高校学习场景的多智能体学习系统：它提供严格用户画像、学习路径、课程知识图谱、证据约束问答，以及学习资料生成和可恢复的 SSE 网页交互。

## 当前运行时

- Web：Next.js + FastAPI；agent_stream_v2 支持进度、断线重放和线程恢复。
- 学习：用户画像、学习路径、KnowledgeGraphV1、测评和七类资源生成。
- 检索：唯一的 Parent–Child primary（向量检索 + BM25 + RRF + reranker + 父块回填）。
- 证据：本地课程证据与网页证据经过 requirement/evidence judge；不足时只能在同一条严格流程中进行有界补搜。
- 持久化：PostgreSQL checkpoint，启动与 /health/ready 都 fail-closed。
- 部署边界：仅面向本机 Docker 网页交互；不宣称已完成真实六场景 Canary 或人工教学验收。

## Parent–Child primary

网页后端只读取：

~~~text
indexes/parent_child/
  primary/
    primary_state.json
    revisions/r<revision>/
      primary_metadata.json
      primary_validation.json
      chroma_children/
      parents.sqlite
      bm25/
      policy_manifest.json
      subject_manifest.json
~~~

primary_state.json 是唯一运行时指针。它记录 revision、更新时间、配置指纹和成功的结构校验状态；不使用 sealed marker、READY、generation registry、shadow、previous 或回滚指针。

每次构建先写入 primary/.staging/<build-id>。只有 Chroma、BM25、父库、集合维度、provider 身份、科目和 chunk 策略都通过严格校验后，才会原子更新 state。缺失或损坏 primary 时后端拒绝启动，绝不回退到根目录 chroma_store。

已有 Parent–Child 产物可迁移为 revision 1：

~~~powershell
python scripts/migrate_parent_child_primary.py --project-root . --index-config config/rag/index.production.yaml --source-artifact-identity pc_20260715_98336c2_55 --build-id migrate-primary-r1
~~~

后续用新 chunk 策略直接重建 primary：

~~~powershell
python scripts/build_parent_child_primary.py --project-root . --index-config config/rag/index.production.yaml --build-id rebuild-20260719 --artifact-identity pc-primary-20260719
~~~

请先阅读 [primary 本地运行手册](docs/runbooks/parent_child_primary_local.md)。在真实 Docker 浏览器 Canary 和人工网页交互通过前，不要删除旧 chroma_store、历史 generation 或 registry 文件。

## Docker 一键部署

前置条件：Docker Desktop / Compose v2、一个显式但不提交的环境文件、获授权的课程数据目录，以及已构建的 primary 索引。

~~~powershell
Copy-Item .env.example .env
# 编辑 .env：填入密钥、数据库密码、COURSE_DATA_HOST_PATH 和 PARENT_CHILD_INDEX_HOST_PATH
$env:A3_ENV_FILE = (Resolve-Path .env).Path
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE config --quiet
docker compose --project-name a3_study_agent --env-file $env:A3_ENV_FILE up --detach --build --wait --wait-timeout 420
Invoke-WebRequest http://localhost:8000/health/ready -UseBasicParsing
~~~

必要变量包括 RAG_EMBEDDING_API_KEY、RAG_RERANKER_API_KEY、POSTGRES_PASSWORD、COURSE_DATA_HOST_PATH 和 PARENT_CHILD_INDEX_HOST_PATH。Compose 不挂载 /app/chroma_store；运行时 Chroma 快照位于独立的 rag_runtime_chroma 卷。

/health/ready 必须返回 health_ready_v4，并包含 parent_child_primary_revision、parent_child_primary_updated_at 和 parent_child_primary_config_fingerprint。浏览器 Canary 会在交互前后读取两次该端点，拒绝 primary 身份漂移。

## 质量检查

~~~powershell
python -m py_compile app.py src/schemas.py
ruff check .
ruff format --check .
python -m pytest tests/test_primary_index.py tests/test_app_health.py tests/test_production_browser_canary.py -q
~~~

可用时还应执行 lint-imports、类型检查、Semgrep、Bandit 和 Gitleaks。缺失工具不是通过。

## 项目资料

- [竞赛文档索引](docs/competition/README.md)
- [系统开发说明](docs/competition/system_development.md)
- [测试说明](docs/competition/test_report.md)
- [部署说明](docs/competition/deployment_guide.md)
- [第三方软件与 AI 工具说明](docs/competition/third_party_notices.md)
- [Primary RAG 本地运行手册](docs/runbooks/parent_child_primary_local.md)

## License

代码采用 [MIT License](LICENSE)。课程资料、模型服务和第三方组件须按各自授权使用。
