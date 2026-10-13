"""Builds the configured chat model. The guards run first: nothing is constructed (no client, no key read) for a
refused model. Every model, the offline fake included, is wrapped so the egress check always runs."""

from __future__ import annotations

from notewell.config import DEFAULT_MODELS, LLMProviderKind, Settings
from notewell.llm.base import ChatModel, LLMError
from notewell.llm.fake import FakeScribe
from notewell.llm.guards import ensure_free_models, ensure_paid_allowed
from notewell.llm.openai_compat import OpenAICompatibleModel
from notewell.llm.resilient import CallLedger, DiskCache, ResilientModel, Throttle


def build_chat_model(
    settings: Settings,
    *,
    provider: LLMProviderKind | None = None,
    model: str | None = None,
    fallback_models: list[str] | None = None,
) -> ChatModel:
    kind: LLMProviderKind = provider or settings.llm_provider
    name = model or (settings.llm_model if provider is None else "") or DEFAULT_MODELS[kind]
    fallbacks = settings.llm_fallback_models if fallback_models is None else fallback_models
    if kind == "fake":
        return FakeScribe()
    if kind == "openrouter":
        if settings.require_free_models or not settings.allow_paid_models:
            ensure_free_models([name, *fallbacks])
        return OpenAICompatibleModel(
            kind="openrouter",
            base_url=settings.openrouter_base_url,
            model=name,
            api_key=settings.openrouter_api_key,
            fallback_models=fallbacks,
            require_free=settings.require_free_models or not settings.allow_paid_models,
            timeout_seconds=settings.llm_timeout_seconds,
            reasoning_effort=settings.llm_reasoning_effort,
        )
    if kind == "openai_compatible":
        ensure_paid_allowed(kind, name, allowed=settings.allow_paid_models)
        if not settings.llm_base_url:
            raise LLMError("NOTEWELL_LLM_BASE_URL is required for the openai_compatible provider")
        return OpenAICompatibleModel(
            kind="openai_compatible",
            base_url=settings.llm_base_url,
            model=name,
            api_key=settings.llm_api_key,
            require_free=False,
            timeout_seconds=settings.llm_timeout_seconds,
            send_temperature=not name.startswith("claude"),
        )
    raise LLMError(f"unknown LLM provider {kind!r}")


def resilient(model: ChatModel, settings: Settings, *, tag: str = "", cache: bool | None = None) -> ResilientModel:
    use_cache = settings.llm_cache if cache is None else cache
    return ResilientModel(
        model,
        ledger=None if model.is_local else CallLedger(settings.llm_ledger, settings.llm_max_calls),
        cache=DiskCache(settings.llm_cache_dir) if use_cache and not model.is_local else None,
        throttle=None if model.is_local else Throttle(settings.llm_min_seconds_between_requests),
        max_retries=settings.llm_max_retries,
        tag=tag,
    )
