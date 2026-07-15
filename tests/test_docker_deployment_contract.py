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
    assert "./indexes/parent_child:/app/indexes/parent_child:ro" in compose_text
    assert "artifacts:/app/artifacts" in compose_text
    assert compose["services"]["backend"]["build"]["target"] == "backend"
    assert compose["services"]["frontend"]["build"]["target"] == "frontend"


def test_next_build_does_not_ignore_typescript_errors() -> None:
    config = (ROOT / "frontend" / "next.config.mjs").read_text(encoding="utf-8")

    assert "ignoreBuildErrors" not in config
