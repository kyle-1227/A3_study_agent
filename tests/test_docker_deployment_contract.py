from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_supervised_single_process_targets() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim AS backend" in dockerfile
    assert "FROM node:20-alpine AS frontend" in dockerfile
    assert '["uvicorn", "app:app"' in dockerfile
    assert '["node", "server.js"]' in dockerfile
    assert 'CMD ["sh", "-c"' not in dockerfile
    assert "& uvicorn" not in dockerfile
    assert dockerfile.index("COPY src/ ./src/") < dockerfile.index(
        "RUN pip install --no-cache-dir ."
    )
    assert "python -m playwright install --with-deps chromium" in dockerfile


def test_compose_requires_secrets_and_persists_runtime_artifacts() -> None:
    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)

    assert set(compose["services"]) == {"postgres", "backend", "frontend", "jaeger"}
    assert "${POSTGRES_PASSWORD:?" in compose_text
    assert "${NEXT_PUBLIC_API_URL:?" in compose_text
    assert "${POSTGRES_PASSWORD:-" not in compose_text
    assert "${NEXT_PUBLIC_API_URL:-" not in compose_text
    assert "${COURSE_DATA_HOST_PATH:?" in compose_text
    assert "${PARENT_CHILD_INDEX_HOST_PATH:?" in compose_text
    assert "${PARENT_CHILD_GENERATION_ID:?" in compose_text
    assert ":/app/indexes/parent_child:ro" in compose_text
    assert (
        "rag_runtime_chroma:/app/indexes/parent_child/.runtime_chroma" in compose_text
    )
    assert "artifacts:/app/artifacts" in compose_text
    assert compose["services"]["backend"]["build"]["target"] == "backend"
    assert compose["services"]["frontend"]["build"]["target"] == "frontend"
    assert compose["services"]["backend"]["environment"]["CHECKPOINTER_ENABLED"] == (
        "true"
    )
    assert compose["services"]["backend"]["environment"]["CHECKPOINTER_TYPE"] == (
        "postgres"
    )
    backend_healthcheck = compose["services"]["backend"]["healthcheck"]["test"]
    assert any("/health/ready" in part for part in backend_healthcheck)
    assert all("/openapi.json" not in part for part in backend_healthcheck)
    frontend_healthcheck = compose["services"]["frontend"]["healthcheck"]["test"]
    assert any("http://127.0.0.1:3000" in part for part in frontend_healthcheck)


def test_next_build_does_not_ignore_typescript_errors() -> None:
    config = (ROOT / "frontend" / "next.config.mjs").read_text(encoding="utf-8")

    assert "ignoreBuildErrors" not in config


def test_environment_example_uses_dedicated_rag_secret_names() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "RAG_EMBEDDING_API_KEY=replace_with_rag_embedding_api_key" in env_example
    assert "RAG_RERANKER_API_KEY=replace_with_rag_reranker_api_key" in env_example
    assert "EMBEDDING_API_KEY_ENV=RAG_EMBEDDING_API_KEY" in env_example
    assert "RERANKER_API_KEY_ENV=RAG_RERANKER_API_KEY" in env_example
    assert env_example.count("CHROMA_PERSIST_DIR=") == 1
    assert "CONTEXT_POLICY_MODE=strict" in env_example
