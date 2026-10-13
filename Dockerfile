# syntax=docker/dockerfile:1

# ---- build: resolve dependencies with uv into a self-contained virtualenv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
# EXTRAS=speech adds faster-whisper (local transcription); the default image runs on gold transcripts.
ARG EXTRAS=""
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $([ -n "$EXTRAS" ] && echo "--extra $EXTRAS")

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $([ -n "$EXTRAS" ] && echo "--extra $EXTRAS")

# ---- runtime: slim image, non-root user ----
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app data/visits ./data/visits
COPY --chown=app:app data/phi_challenge.yaml data/phi_challenge_2.yaml ./data/
COPY --chown=app:app results ./results

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NOTEWELL_DATA_DIR=/home/app/var \
    NOTEWELL_DATASET_DIR=/app/data/visits \
    NOTEWELL_AUDIO_DIR=/app/data/audio \
    NOTEWELL_CODES_DIR=/home/app/codes \
    NOTEWELL_RESULTS_DIR=/app/results \
    NOTEWELL_LLM_LEDGER=/home/app/var/calls.jsonl \
    NOTEWELL_LLM_CACHE_DIR=/home/app/var/llm-cache \
    NOTEWELL_LOG_FORMAT=json

USER app
RUN mkdir -p /home/app/var /home/app/codes

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "notewell.api.app:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
