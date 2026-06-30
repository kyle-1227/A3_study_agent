# A3 Study Agent

高校个性化学习资源生成智能体。

<p align="center">
  <a href="README_en.md">English README</a> |
  <a href="docs/architecture/v0.3.0/diagram_design.md">Architecture Diagrams</a> |
  <a href="CHANGELOG.md">Changelog</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v0.3.0-orange?style=flat-square" alt="version" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="python" />
  <a href="https://github.com/langchain-ai/langgraph">
    <img src="https://img.shields.io/badge/langgraph-v1.1.1-7C3AED?style=flat-square&logo=diagram-next&logoColor=white" alt="langgraph" />
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license" />
  </a>
</p>

## 关于项目

A3 Study Agent 是一个面向高校课程学习场景的多智能体系统。它围绕学习者的问题、目标和资源需求，生成课程答疑、分层练习、思维导图、复习文档和学习计划等个性化学习资源。

系统结合本地课程资料 RAG、BM25、reranker、Tavily Web Research、Evidence Judge V2、DeepSeek strict structured output、SSE 流式输出和 OpenTelemetry 可观测性，支持真实交互链路中的检索、证据判断、生成和诊断。

当前 React 前端主要用于演示复杂 Agent 交互、SSE 流式输出、资源生成和运行轨迹。外部 LangGraph/SSE 节点名仍保留 `web_search`，内部语义已经统一为 Web Research V2。

## 核心能力

- **课程答疑**：基于本地课程材料和 Web Research evidence 的双源证据，生成面向高校学习者的解释、示例和学习建议。
- **个性化资源生成**：生成分层练习题、思维导图、复习文档、项目案例和学习材料摘要。
- **学习计划**：通过计划起草、审查和人工反馈，支持阶段化学习安排。
- **学习支持**：以高校学习导师 / 学业支持导师的语气，提供温暖且可执行的建议。
- **稳定结构化输出**：小型结构化节点使用 DeepSeek official strict tool calling；结构化失败通过 re-ask retry 提升恢复能力。
- **可观测性**：通过 A3_TRACE、OpenTelemetry、SSE 节点事件和结构化诊断日志排查真实交互链路。
- **配置驱动**：通过 YAML settings 和 XML prompts 管理运行参数、模型行为和提示词。

## 系统架构

```mermaid
graph TD
  START([学习者输入]) --> supervisor[意图识别]

  supervisor --> search_query_rewriter[查询改写]
  supervisor -->|emotional| emotional_response[学业支持回应]
  supervisor -->|unknown| handle_unknown[未知意图处理]

  search_query_rewriter --> academic_router[学习路由]
  academic_router --> rag_retrieve[Local RAG]
  academic_router --> web_search[Web Research V2]
  rag_retrieve --> evidence_judge[Evidence Judge V2]
  web_search --> evidence_judge

  evidence_judge --> generate_answer[回答生成]
  evidence_judge --> mindmap_planner[思维导图规划]
  evidence_judge --> exercise_planner[练习规划]
  evidence_judge --> review_doc_planner[复习文档规划]
  evidence_judge --> study_plan_emotional_intel[学习计划上下文]

  generate_answer --> evaluate_hallucination[忠实性评估]
  evaluate_hallucination -->|通过| END_A([结束])
  evaluate_hallucination -->|重试| rewrite_query[查询重写]
  rewrite_query --> academic_router

  mindmap_planner --> mindmap_agent[思维导图生成]
  mindmap_agent --> mindmap_reviewer[思维导图审查]
  mindmap_reviewer -->|通过| mindmap_output[导图导出]
  mindmap_reviewer -->|打回| mindmap_rewrite[导图修订]
  mindmap_rewrite --> mindmap_agent

  exercise_planner --> exercise_agent[题目生成]
  exercise_agent --> exercise_reviewer[题目审查]
  exercise_reviewer -->|通过| exercise_output[练习输出]
  exercise_reviewer -->|打回| exercise_rewrite[题目修订]
  exercise_rewrite --> exercise_agent

  review_doc_planner --> review_doc_agent[复习文档生成]
  review_doc_agent --> review_doc_reviewer[复习文档审查]
  review_doc_reviewer -->|通过| review_doc_output[文档导出]
  review_doc_reviewer -->|打回| review_doc_rewrite[文档修订]
  review_doc_rewrite --> review_doc_agent

  study_plan_emotional_intel --> study_plan_planner[学习计划规划]
  study_plan_planner --> study_plan_agent[学习计划生成]
  study_plan_agent --> study_plan_reviewer_academic[学术审查]
  study_plan_agent --> study_plan_reviewer_emotional[负荷审查]
  study_plan_reviewer_academic --> study_plan_consensus[共识检查]
  study_plan_reviewer_emotional --> study_plan_consensus
  study_plan_consensus -->|通过| study_plan_output[计划输出 + HIL]
  study_plan_consensus -->|打回| study_plan_rewrite[计划修订]
  study_plan_rewrite --> study_plan_agent

  emotional_response --> END_E([结束])
  handle_unknown --> END_U([结束])
```

更多图示见 [`docs/architecture/v0.3.0/diagram_design.md`](docs/architecture/v0.3.0/diagram_design.md)。

## 技术栈

| 层级 | 组件 |
| ---- | ---- |
| 前端 | Next.js 16、React、Tailwind CSS、React Flow |
| 后端 API | FastAPI、Uvicorn、SSE |
| 编排 | LangGraph |
| 本地知识库 | ChromaDB、BM25、reranker |
| Web Research | Tavily |
| 结构化输出 | DeepSeek official strict tool calling、Pydantic validation、re-ask retry |
| 证据判断 | Evidence Judge V2 item grader + sufficiency judge |
| 状态快照 | LangGraph Checkpointer，默认 MemorySaver，可选 PostgreSQL |
| 可观测性 | A3_TRACE、OpenTelemetry、Jaeger、SQLite fallback |
| 配置 | YAML settings、XML prompts |

## 快速启动

### Docker Compose

```bash
git clone https://github.com/kyle-1227/A3_study_agent.git
cd A3_study_agent

cp .env.example .env
# 编辑 .env，填入模型、检索和观测配置。

docker compose up -d

# 可选：启用 Jaeger tracing
docker compose --profile observability up -d
```

前端：`http://localhost:3000`
后端 API：`http://localhost:8000`
Jaeger：`http://localhost:16686`

### 本地开发

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# 编辑 .env，填入 API keys。
```

#### 构建知识库

将 PDF / MD / TXT 课程资料放入以下一个或多个目录：

- `data/big_data`
- `data/computer`
- `data/machine_learning`
- `data/math`
- `data/python`

然后运行：

```bash
python scripts/build_index.py
```

#### 启动服务

后端和前端需要分别打开两个终端运行。

**终端 1：后端**

```bash
uvicorn app:app --reload --port 8000
```

**终端 2：前端**

```bash
cd frontend
npm install
npm run dev
```

注意：`pytest tests/test_security.py -q` 是后端测试命令，不要放在前端启动终端里。

## 项目结构

```text
A3_study_agent/
|-- app.py                         # FastAPI SSE endpoints + lifespan
|-- docker-compose.yml             # Backend + PostgreSQL + Jaeger
|-- config/
|   |-- settings.yaml              # Runtime parameters
|   `-- prompts/                   # XML prompt templates
|-- src/
|   |-- graph/                     # LangGraph nodes and state flow
|   |-- rag/                       # Local retrieval and indexing
|   |-- llm/                       # LLM factory and structured output runtime
|   |-- database/                  # Checkpointer management
|   |-- tracing/                   # OpenTelemetry setup
|   `-- tools/                     # Web research and resource tools
|-- frontend/                      # Next.js UI
|-- data/                          # University course materials
|-- scripts/                       # Indexing and debug scripts
`-- tests/                         # Test suite
```

## 测试

后端测试：

```bash
python -m pytest tests/test_config.py tests/test_app.py tests/test_rag.py tests/test_tracing.py -v
python -m pytest tests/test_security.py -q
```

如果环境允许，可以运行完整后端测试：

```bash
python -m pytest -q
```

前端检查：

```bash
cd frontend
npm run lint
.\node_modules\.bin\tsc.cmd --noEmit
npm run build
```

## License

[MIT](./LICENSE)
