# ── Stage 1: Frontend build ──────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./

ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

RUN npm run build


# ── Stage 2: Python backend ─────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg (PostgreSQL driver)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get purge -y curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Backend source
COPY app.py ./
COPY src/ ./src/
COPY config/ ./config/

# Knowledge base data
COPY data/ ./data/

# Scripts
COPY scripts/ ./scripts/

# Frontend standalone output from Stage 1
COPY --from=frontend-builder /app/frontend/.next/standalone ./frontend-standalone/
COPY --from=frontend-builder /app/frontend/.next/static ./frontend-standalone/frontend/.next/static

EXPOSE 8000 3000

# Start both frontend and backend
CMD ["sh", "-c", "node frontend-standalone/frontend/server.js & uvicorn app:app --host 0.0.0.0 --port 8000"]