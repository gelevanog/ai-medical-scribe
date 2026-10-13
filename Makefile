.DEFAULT_GOAL := help
.PHONY: help install serve web test lint format web-lint web-build codes models audio eval-offline eval-asr eval-real eval-local report calls docker-build docker-up clean

FREE_MODEL ?= nvidia/nemotron-3-super-120b-a12b:free
FREE_FALLBACKS ?= nvidia/nemotron-3-ultra-550b-a55b:free

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install:  ## Python dependencies (uv, with the speech extra) and the web app's (npm)
	uv sync --all-extras
	cd web && npm ci --no-audit --no-fund

serve:  ## API on http://localhost:8000 (offline fake model unless NOTEWELL_LLM_PROVIDER is set)
	uv run notewell serve --port 8000

web:  ## Next.js dev server on http://localhost:3000
	cd web && npm run dev

test:  ## Python tests (no keys, no model downloads)
	uv run pytest -m "not speech"

lint:  ## Ruff lint + format check + mypy (strict)
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:  ## Auto-format and fix lint issues
	uv run ruff format src tests
	uv run ruff check --fix src tests

web-lint:  ## ESLint + TypeScript type check
	cd web && npm run lint && npm run typecheck

web-build:  ## Production build of the web app
	cd web && npm run build

codes:  ## Download the full ICD-10-CM FY2027 code list from CDC (checksum verified)
	uv run notewell codes download

models:  ## Kokoro-82M and faster-whisper small.en/base.en weights (~0.9 GB, CPU)
	uv run notewell models download --whisper small.en --whisper base.en

audio:  ## Speak the 32 synthetic visits with two Kokoro voices (clean + noisy WAV), ~35 min on 16 cores
	uv run notewell data synth-audio

eval-offline:  ## Notes, canaries and guards with the offline fake model (no API calls)
	uv run notewell eval notes fake --provider fake
	uv run notewell eval phi

eval-asr:  ## WER, key terms and diarization on the synthetic audio (local, CPU)
	uv run notewell eval asr --model small.en --condition clean
	uv run notewell eval asr --model small.en --condition noisy
	uv run notewell eval asr --model base.en --condition clean
	uv run notewell eval asr --model small.en --condition clean --no-prompt

eval-real:  ## Notes from gold and ASR transcripts + the canary leak test with free models (needs OPENROUTER_API_KEY)
	uv run notewell eval notes main --provider openrouter --model $(FREE_MODEL) --fallbacks $(FREE_FALLBACKS)
	uv run notewell eval notes asr --provider openrouter --model $(FREE_MODEL) --fallbacks $(FREE_FALLBACKS) --transcripts results/asr/transcripts/small.en-clean
	uv run notewell eval canary --provider openrouter --model $(FREE_MODEL) --fallbacks $(FREE_FALLBACKS)
	uv run notewell eval guards --run main

report:  ## results/summary.json and results/report.md, plus the call-ledger summary
	uv run notewell calls
	uv run notewell eval report

docker-build:  ## Build the API and web images
	docker compose build

docker-up:  ## Everything in Docker: web :3000, API :8000 (with the mock FHIR server)
	docker compose up --build

clean:  ## Remove caches (keeps results/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache web/.next
