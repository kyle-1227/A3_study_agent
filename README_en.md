# A3 Study Agent

A3 Study Agent 鈥?AI-powered personalized learning resource generation assistant for university students.

<p align="center">
  <a href="README.md">涓枃 README</a> |
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

## About

A3 Study Agent is a multi-agent system for university learning scenarios. It helps learners generate personalized learning resources such as course Q&A, layered exercises, mind maps, project examples, and study plans.

The system combines local course-material RAG, BM25, reranking, Tavily Web Search, structured LLM output, evidence judging, SSE streaming, and OpenTelemetry tracing. It is designed for diagnosable end-to-end learning workflows rather than exam-prep-specific learning support.

The current React frontend is a lightweight reference implementation for demonstrating complex agent interaction, streaming output, generated resources, and runtime traces. Additional planning modules may evolve later.

## Core Capabilities

- **Course Q&A**: Answer university course questions using local course materials and judged web evidence.
- **Personalized Resource Generation**: Generate exercises, mind maps, project cases, and learning-material summaries.
- **Study Planning**: Draft and review staged study plans with multi-agent review and human feedback.
- **Academic Support**: Respond with the tone of a university learning mentor or academic support advisor.
- **Observability**: Use A3_TRACE, OpenTelemetry, SSE node events, and structured diagnostics to inspect real interactions.
- **Configuration Driven**: Control behavior through YAML runtime settings and XML prompt templates.

## Architecture

```mermaid
graph TD
  START([Learner Input]) --> supervisor[Intent Classification]

  supervisor --> search_query_rewriter[Query Rewriter]
  search_query_rewriter --> academic_router[Academic Router]
  supervisor -->|emotional| emotional_response[Academic Support]
  supervisor -->|unknown| handle_unknown[Unknown Intent]

  academic_router --> rag_retrieve[Local RAG]
  academic_router --> web_search[Tavily Web Search]
  rag_retrieve --> evidence_judge[Evidence Judge]
  web_search --> evidence_judge

  evidence_judge --> generate_answer[Generate Answer]
  evidence_judge --> study_plan_emotional_intel[Emotional / Workload Intel]
  evidence_judge --> mindmap_planner[Mindmap Planner]
  evidence_judge --> exercise_planner[Exercise Planner]
  evidence_judge --> review_doc_planner[Review Doc Planner]

  generate_answer --> evaluate_hallucination[Faithfulness Eval]
  evaluate_hallucination -->|Pass| END_A([End])
  evaluate_hallucination -->|Retry| rewrite_query[Query Rewrite]
  rewrite_query --> academic_router

  study_plan_emotional_intel --> study_plan_planner[Study Plan Planner]
  study_plan_planner --> study_plan_agent[Study Plan Generator]
  study_plan_agent --> study_plan_reviewer_academic[Academic Reviewer]
  study_plan_agent --> study_plan_reviewer_emotional[Emotional Reviewer]
  study_plan_reviewer_academic --> study_plan_consensus[Consensus Check]
  study_plan_reviewer_emotional --> study_plan_consensus
  study_plan_consensus -->|Pass| study_plan_output[Study Plan Output]
  study_plan_consensus -->|Reject| study_plan_rewrite[Plan Revision]
  study_plan_rewrite --> study_plan_agent
  study_plan_output --> END_P([End])

  emotional_response --> END_E([End])
  handle_unknown --> END_U([End])
```

See [`docs/architecture/v0.3.0/diagram_design.md`](docs/architecture/v0.3.0/diagram_design.md) for more diagrams.

## Tech Stack

| Layer | Components |
| ----- | ---------- |
| Frontend | Next.js 16, React, Tailwind CSS, React Flow |
| Backend API | FastAPI, Uvicorn, SSE |
| Orchestration | LangGraph |
| Local Knowledge | ChromaDB, BM25, reranker |
| Web Search | Tavily |
| State Snapshots | LangGraph Checkpointer, MemorySaver by default, optional PostgreSQL |
| Observability | A3_TRACE, OpenTelemetry, Jaeger, SQLite fallback |
| Configuration | YAML settings, XML prompts |

## Quick Start

### Docker Compose

```bash
git clone https://github.com/kyle-1227/A3_study_agent.git
cd A3_study_agent

cp .env.example .env
# Edit .env and fill in model, search, and observability settings.

docker compose up -d

# Optional: enable Jaeger tracing
docker compose --profile observability up -d
```

Frontend: `http://localhost:3000`
Backend API: `http://localhost:8000`
Jaeger: `http://localhost:16686`

### Local Development

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Fill in API keys.
```

#### Build the Knowledge Base

Place PDF / MD / TXT course materials under one or more subject directories:

- `data/big_data`
- `data/computer`
- `data/machine_learning`
- `data/math`
- `data/python`

Then run:

```bash
python scripts/build_index.py
```

#### Run

```bash
# Terminal 1: backend
uvicorn app:app --reload --port 8000

# Terminal 2: frontend
cd frontend
npm install
npm run dev
```

## Project Structure

```text
A3_study_agent/
鈹溾攢鈹€ app.py                         # FastAPI SSE endpoints + lifespan
鈹溾攢鈹€ docker-compose.yml             # Backend + PostgreSQL + Jaeger
鈹溾攢鈹€ config/
鈹?  鈹溾攢鈹€ settings.yaml              # Runtime parameters
鈹?  鈹斺攢鈹€ prompts/                   # XML prompt templates
鈹溾攢鈹€ src/
鈹?  鈹溾攢鈹€ graph/                     # LangGraph nodes and state flow
鈹?  鈹溾攢鈹€ rag/                       # Local retrieval and indexing
鈹?  鈹溾攢鈹€ llm/                       # LLM factory and structured output runtime
鈹?  鈹溾攢鈹€ database/                  # Checkpointer management
鈹?  鈹溾攢鈹€ tracing/                   # OpenTelemetry setup
鈹?  鈹斺攢鈹€ tools/                     # Web search and resource tools
鈹溾攢鈹€ frontend/                      # Next.js UI
鈹溾攢鈹€ data/                          # University course materials
鈹溾攢鈹€ scripts/                       # Indexing and debug scripts
鈹斺攢鈹€ tests/                         # Test suite
```

## Testing

```bash
python -m pytest tests/test_config.py tests/test_app.py tests/test_rag.py tests/test_tracing.py -v

# If the environment allows:
python -m pytest -q
cd frontend && npm run build
```

## License

[MIT](./LICENSE)

