# A3 Study Agent

A3 Study Agent — AI-powered personalized learning resource generation assistant for university students.

<p align="center">
  <a href="README.md">中文 README</a> |
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

The system combines local course-material RAG, BM25, reranking, Tavily Web Search, structured LLM output, evidence judging, SSE streaming, and OpenTelemetry tracing. It is designed for diagnosable end-to-end learning workflows rather than exam-prep-specific tutoring.

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

  supervisor -->|academic| academic_router[Academic Router]
  supervisor -->|planning| gather_planning_context[Planning Context Retrieval]
  supervisor -->|emotional| emotional_response[Academic Support]
  supervisor -->|unknown| handle_unknown[Unknown Intent]

  academic_router --> rag_retrieve[RAG / Web Evidence Retrieval]
  rag_retrieve --> generate_answer[Generate Answer]
  generate_answer --> evaluate_hallucination[Faithfulness Eval]
  evaluate_hallucination -->|Pass| END_A([End])
  evaluate_hallucination -->|Retry| rewrite_query[Query Rewrite]
  rewrite_query --> academic_router

  gather_planning_context --> gather_intel[Intel Gathering]
  gather_intel --> drafter[Plan Drafter]
  drafter --> reviewer_academic[Academic Reviewer]
  drafter --> reviewer_emotional[Emotional Reviewer]
  reviewer_academic --> consensus_check[Consensus Check]
  reviewer_emotional --> consensus_check
  consensus_check -->|Pass| plan_output[Plan Output + HIL]
  consensus_check -->|Reject| adv_rewrite[Plan Revision]
  adv_rewrite --> drafter

  plan_output -->|Confirm| END_P([End])
  plan_output -->|Feedback| feedback_router[Feedback Classification]
  feedback_router -->|Tweak| plan_tweak[Plan Fine-tuning]
  feedback_router -->|Rewrite| drafter
  plan_tweak --> plan_output

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
conda create -n a3_study_agent python=3.11 -y
conda activate a3_study_agent

pip install -e ".[dev]"

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
├── app.py                         # FastAPI SSE endpoints + lifespan
├── docker-compose.yml             # Backend + PostgreSQL + Jaeger
├── config/
│   ├── settings.yaml              # Runtime parameters
│   └── prompts/                   # XML prompt templates
├── src/
│   ├── graph/                     # LangGraph nodes and state flow
│   ├── rag/                       # Local retrieval and indexing
│   ├── llm/                       # LLM factory and structured output runtime
│   ├── database/                  # Checkpointer management
│   ├── tracing/                   # OpenTelemetry setup
│   └── tools/                     # Web search and resource tools
├── frontend/                      # Next.js UI
├── data/                          # University course materials
├── scripts/                       # Indexing and debug scripts
└── tests/                         # Test suite
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
