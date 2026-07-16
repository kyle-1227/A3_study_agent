# syntax=docker/dockerfile:1

FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

ARG NEXT_PUBLIC_API_URL
RUN test -n "$NEXT_PUBLIC_API_URL"
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

RUN npm run typecheck && npm run build


FROM python:3.11-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md README_en.md ./
COPY src/ ./src/
RUN --mount=type=cache,id=a3-pip-cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 .
RUN python -m playwright install --with-deps chromium

COPY app.py ./
COPY config/ ./config/
COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


FROM node:20-alpine AS frontend

ENV HOSTNAME=0.0.0.0 \
    PORT=3000

WORKDIR /app

COPY --from=frontend-builder /app/frontend/.next/standalone ./
COPY --from=frontend-builder /app/frontend/.next/static ./.next/static

EXPOSE 3000

CMD ["node", "server.js"]
