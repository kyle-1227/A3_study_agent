# 高考辅导 AI

<p align="center">
  <a href="README_en.md">English README</a> ·
  <a href="docs/architecture/v0.3.0/diagram_design.md">架构图</a> ·
  <a href="CHANGELOG.md">更新日志</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v0.3.0-orange" alt="version" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license" />
</p>

一个面向生产场景的高考备考多智能体对话 AI 系统，基于 **LangGraph**（有状态编排）、**FastAPI**（SSE 流式传输）和 **Next.js**（响应式前端）构建。轻量级 Qwen2.5-7B 路由 Agent 将用户问题分发给三个专项 Agent：学科辅导、学习规划和情绪疏导，每个分支都具备完整的可观测性和容错机制。

---

## 效果演示

<!-- TODO: 替换为 v0.3.0 截图 -->
<!-- <img src="./assets/v0.3.0/screenshot-chat.png" alt="聊天界面" style="zoom:40%;" /> -->
<!-- <img src="./assets/v0.3.0/screenshot-hil.png" alt="HIL计划审阅" style="zoom:40%;" /> -->
<!-- <img src="./assets/v0.3.0/screenshot-dag.png" alt="DAG图视图" style="zoom:40%;" /> -->

<!-- TODO: 替换为演示视频链接 -->
<!-- [演示视频](./assets/v0.3.0/demo.mp4) -->

---

## v0.3.0 新特性

- **对抗式计划生成**：学习计划引入多智能体博弈——"起草者"生成计划，"学术审查员"和"情绪审查员"并行审阅，全票通过才放行，否则打回重写
- **人工介入 (HIL) 计划审批**：对抗循环收敛后，图执行挂起（LangGraph `interrupt`），将草稿推送到前端，用户可直接编辑或提供自然语言反馈
- **反馈路由器**：智能判断用户反馈需要"微调"还是"重写"——微调仅局部修改（快速），重写清空草稿从头规划（完整对抗循环）
- **单摘要防膨胀**：多轮反馈只保留一条压缩摘要，避免上下文无限增长
- **交互式 DAG 视图**：React Flow 替换静态 SVG，支持拖拽平移、滚轮缩放，实时显示 19 个节点的运行状态
- **计划导出**：一键下载学习计划为 Markdown 文件
- **`text` SSE 事件**：非流式节点的完整输出通过 `text` 事件推送，解决了计划和兜底回复不可见的问题
- **`done` SSE 事件**：流完成标记，前端可准确判断流结束
- **输入校验**：Pydantic `max_length` 防御超长输入

---

## 核心功能

- **学科问答** — 混合 RAG（向量 + BM25 + Reranker）并行 Fan-out/Fan-in 检索，幻觉评估 + 自动重试闭环
- **学习规划** — 对抗式多智能体起草 + 审查循环，结合实时高考政策搜索，支持人工反馈迭代
- **情绪支持** — 以经验丰富的班主任身份，提供温暖而实用的回应
- **意图路由** — Qwen2.5-7B Supervisor 低延迟分类用户意图，精准分发
- **LLM 容灾** — DeepSeek 主 API 超时或 5xx 时，自动切换到 SiliconFlow（Qwen2.5-7B）
- **分布式追踪** — OpenTelemetry 全链路埋点，导出到 Jaeger（OTLP）+ SQLite 兜底
- **状态持久化** — PostgreSQL 驱动的 LangGraph Checkpointer；无数据库时自动降级为无状态运行
- **配置驱动** — YAML 运行参数 + XML 提示词注册表，修改行为无需动代码
- **实时可观测** — SSE 驱动的推理路径（节点列表或交互式 DAG）、节点耗时、错误流和 Token 用量
- **Markdown 渲染** — 完整 GFM 支持：表格、代码块、LaTeX 公式、列表

---

## 系统架构

```text
用户 ──► Next.js (SSE) ──► FastAPI /stream ──► LangGraph StateGraph
                                                       │
                    ┌──────────────────────────────────┼───────────────────────┐
                    ▼                                  ▼                       ▼
               [学科辅导]                          [学习规划]              [情绪支持]
             ┌──────┴──────┐                          │                       │
             ▼             ▼                          ▼                       ▼
      rag_retrieve    web_search              search_policy        emotional_response
      (向量+BM25       (DuckDuckGo)                  │
       +Reranker)          │                         ▼
             └──────┬──────┘                   gather_intel
                    ▼                                │
              generate_answer                       ▼
                    │                            drafter ◄── adv_rewrite (循环)
                    ▼                           ┌───┴───┐
         evaluate_hallucination          reviewer_academic  reviewer_emotional
               ┌────┴────┐                     └───┬───┘
            [重试]      [结束]              consensus_check
               │                                   │
          rewrite_query                       plan_output ─── interrupt (HIL)
               │                               ┌───┴───┐
          academic_router (循环)         [确认→END] [反馈→feedback_router]
                                                        ├─ tweak → plan_tweak ─┐
                                                        └─ rewrite → drafter   │
                                                              plan_output ◄────┘
```

横切关注点：所有节点上的 `@traced_node` → OpenTelemetry → Jaeger UI / SQLite

详细架构图见 [`docs/architecture/v0.3.0/diagram_design.md`](docs/architecture/v0.3.0/diagram_design.md)

---

## 技术选型

| 层级 | 组件 | 说明 |
| ---- | ---- | ---- |
| 前端 | Next.js 16 + Tailwind CSS 4 + React Flow | 响应式聊天 UI、SSE 消费端、交互式 DAG、Markdown 渲染 |
| 后端 API | FastAPI + Uvicorn | SSE 端点（`/stream`、`/resume`）、CORS、OTel 自动埋点 |
| 编排 | LangGraph | StateGraph + `interrupt()` HIL + 条件边 + Fan-out/Fan-in |
| 路由 LLM | Qwen2.5-7B（SiliconFlow） | 轻量意图分类 + 反馈路由（temperature=0.0） |
| 生成 LLM | DeepSeek-V3 | 学科解答、学习计划、情绪支持 |
| LLM 容灾 | Qwen2.5-7B（SiliconFlow） | 跨厂商故障转移 |
| 向量数据库 | ChromaDB | 本地知识库检索（L2→相关度归一化） |
| 文本嵌入 | BAAI/bge-m3（SiliconFlow） | RAG 向量化 |
| 关键词检索 | rank-bm25 + jieba | 中文感知 BM25 检索 |
| 重排序 | BAAI/bge-reranker-v2-m3（SiliconFlow） | 合并候选集的精排 |
| 网络搜索 | DuckDuckGo | 学习规划及学科问答的在线补充 |
| 状态持久化 | PostgreSQL（psycopg） | LangGraph Checkpointer 多轮对话记忆 + HIL 中断恢复 |
| 可观测性 | OpenTelemetry + Jaeger + SQLite | 所有图节点的分布式链路追踪 |
| 配置管理 | YAML + XML | 运行参数与提示词模板 |

---

## 快速启动

### 方式一：Docker Compose（推荐）

```bash
git clone https://github.com/chipfighter/gaokao_tutor.git
cd gaokao_tutor

cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 SILICONFLOW_API_KEY

# 启动（后端 + 前端 + PostgreSQL）
docker compose up -d

# 可选：启用 Jaeger 追踪
docker compose --profile observability up -d
```

前端地址：`http://localhost:3000` · 后端 API：`http://localhost:8000` · Jaeger：`http://localhost:16686`

### 方式二：本地开发

#### 环境要求

- Python 3.11+
- Node.js 18+ 和 npm
- PostgreSQL（可选，用于状态持久化和 HIL；不配置时自动降级为无状态）

#### 后端

```bash
conda create -n gaokao_tutor python=3.11 -y
conda activate gaokao_tutor

pip install -e ".[dev]"

cp .env.example .env
# 编辑 .env 填入 API 密钥
```

#### 构建知识库

将高考试卷的 `.txt` / `.pdf` 文件放入 `data/chinese/` 或 `data/math/` 目录，然后：

```bash
python scripts/build_index.py
```

#### 前端

```bash
cd frontend
npm install
```

#### 启动

```bash
# 终端 1 — 后端
uvicorn app:app --reload --port 8000

# 终端 2 — 前端
cd frontend
npm run dev
```

---

## 项目结构

```text
gaokao_tutor/
├── app.py                        # FastAPI SSE 端点 + lifespan
├── dockerfile                    # 多阶段构建（前端 + 后端）
├── docker-compose.yml            # 一键部署（后端 + PostgreSQL + Jaeger）
├── config/
│   ├── settings.yaml             # 运行参数（温度、超时、重试上限）
│   └── prompts/                  # XML 提示词模板
├── src/
│   ├── graph/
│   │   ├── builder.py            # 图构建与编译（19 个节点）
│   │   ├── state.py              # TutorState TypedDict（26 个字段）
│   │   ├── supervisor.py         # 意图路由 + 关键词提取（Qwen2.5-7B）
│   │   ├── academic.py           # 并行检索、答案生成、幻觉评估
│   │   ├── planner.py            # 政策搜索 + 情报收集
│   │   ├── plan_adversarial.py   # 对抗式起草/审查 + HIL 反馈路由
│   │   ├── emotional.py          # 情绪支持
│   │   └── llm.py                # 统一 LLM 工厂 + 容灾降级
│   ├── rag/                      # 混合检索：向量 + BM25 + Reranker
│   ├── config/                   # YAML 配置加载 + XML 提示词缓存
│   ├── database/                 # PostgreSQL Checkpointer 管理
│   ├── tracing/                  # OTel 初始化、@traced_node、SQLite 导出
│   └── schemas.py                # Pydantic 请求模型
├── frontend/
│   ├── app/page.tsx              # 主页面：SSE 消费、HIL 反馈
│   └── components/
│       ├── chat-area.tsx         # 消息气泡 + Markdown 渲染
│       ├── plan-review.tsx       # HIL 计划审阅组件（编辑/反馈/导出）
│       ├── right-panel.tsx       # 交互式 DAG + 节点轨迹 + 日志
│       └── left-sidebar.tsx      # 对话历史
├── data/                         # 高考试卷（语文、数学）
├── scripts/                      # 索引构建脚本
└── tests/                        # 测试套件（全部 Mock）
```

---

## SSE 事件协议

| 事件类型 | 描述 | 示例载荷 |
| -------- | ---- | -------- |
| `thread_id` | 会话 ID（流开始时） | `{"type":"thread_id","thread_id":"abc..."}` |
| `node_event` | 节点生命周期 | `{"type":"node_event","node":"drafter","status":"start"}` |
| `token` | 流式 Token | `{"type":"token","content":"..."}` |
| `text` | 非流式节点完整输出 | `{"type":"text","content":"...","node":"plan_output"}` |
| `usage` | Token 用量 | `{"type":"usage","node":"drafter","input_tokens":500}` |
| `interrupt` | HIL 中断 | `{"type":"interrupt","draft":"...","thread_id":"..."}` |
| `done` | 流完成 | `{"type":"done"}` |
| `error` | 错误 | `{"type":"error","message":"..."}` |

---

## 测试

```bash
# 单元测试（无需在线 API）
OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short

# 前端构建检查
cd frontend && npm run build
```

---

## 许可证

MIT
