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


def test_compose_mounts_only_parent_child_primary_runtime_storage() -> None:
    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)

    for required in (
        "POSTGRES_PASSWORD",
        "NEXT_PUBLIC_API_URL",
        "COURSE_DATA_HOST_PATH",
        "PARENT_CHILD_INDEX_HOST_PATH",
        "A3_ENV_FILE",
    ):
        assert required in compose_text
    assert "PARENT_CHILD_GENERATION_ID" not in compose_text
    assert "chroma_store" not in compose_text
    assert "chroma_data" not in compose_text

    volumes = compose["services"]["backend"]["volumes"]
    parent_child_mount = next(
        item
        for item in volumes
        if isinstance(item, dict) and item.get("target") == "/app/indexes/parent_child"
    )
    assert parent_child_mount["read_only"] is True
    assert "PARENT_CHILD_INDEX_HOST_PATH" in parent_child_mount["source"]
    assert "rag_runtime_chroma:/app/indexes/parent_child/.runtime_chroma" in (
        compose_text
    )
    assert "rag_runtime_chroma" in compose["volumes"]
    assert compose["services"]["backend"]["healthcheck"]["start_period"] == "240s"


def test_environment_example_has_primary_rag_secrets_without_legacy_flat_rag() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "RAG_EMBEDDING_API_KEY=replace_with_rag_embedding_api_key" in env_example
    assert "RAG_RERANKER_API_KEY=replace_with_rag_reranker_api_key" in env_example
    assert "EMBEDDING_API_KEY_ENV=RAG_EMBEDDING_API_KEY" in env_example
    assert "RERANKER_API_KEY_ENV=RAG_RERANKER_API_KEY" in env_example
    for forbidden in (
        "CHROMA_PERSIST_DIR=",
        "PARENT_CHILD_GENERATION_ID=",
        "RAG_SPLITTER_MODE=",
        "INDEX_ADD_BATCH_SIZE=",
        "INDEX_MAX_RETRIES=",
    ):
        assert forbidden not in env_example


def test_docker_context_excludes_generated_primary_indexes() -> None:
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {
        "artifacts",
        "indexes/parent_child",
        ".runtime_state",
        "frontend/.next",
        "frontend/node_modules",
    } <= ignored
