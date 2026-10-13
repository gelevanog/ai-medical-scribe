"""OpenAI-compatible chat completions over HTTP: OpenRouter (free models by default) or any OpenAI-compatible endpoint
(a BAA-covered gateway in production, vLLM, a budget proxy)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import httpx

from notewell.llm.base import BudgetExceededError, Completion, LLMError, Message, RetryableLLMError
from notewell.llm.guards import ensure_free_body, ensure_free_models, ensure_served_free

Kind = Literal["openrouter", "openai_compatible"]

OPENROUTER_HEADERS = {"HTTP-Referer": "https://github.com/gelevanog/ai-medical-scribe", "X-Title": "Notewell"}


class OpenAICompatibleModel:
    def __init__(
        self,
        *,
        kind: Kind,
        base_url: str,
        model: str,
        api_key: str | None = None,
        fallback_models: Sequence[str] = (),
        require_free: bool = True,
        timeout_seconds: float = 180.0,
        reasoning_effort: str = "",
        send_temperature: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if kind == "openrouter" and not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        if not base_url:
            raise LLMError(f"no base URL configured for the {kind} provider")
        if not model:
            raise LLMError(f"no model configured for the {kind} provider")
        self.kind = kind
        self.model = model
        self.fallback_models = list(fallback_models) if kind == "openrouter" else []
        self.require_free = require_free and kind == "openrouter"
        if self.require_free:
            ensure_free_models([model, *self.fallback_models])
        self.reasoning_effort = reasoning_effort
        self.send_temperature = send_temperature
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if kind == "openrouter":
            self._headers.update(OPENROUTER_HEADERS)
        self._timeout = timeout_seconds
        self._transport = transport

    @property
    def label(self) -> str:
        return f"{self.kind}/{self.model}"

    @property
    def is_local(self) -> bool:
        return False

    def body(self, messages: Sequence[Message], *, max_tokens: int, temperature: float) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": list(messages), "max_tokens": max_tokens}
        if self.send_temperature:
            body["temperature"] = temperature
        if self.kind == "openrouter":
            if self.fallback_models:
                body["models"] = [self.model, *self.fallback_models]
            if self.reasoning_effort in {"none", "off"}:
                body["reasoning"] = {"enabled": False}
            elif self.reasoning_effort:
                body["reasoning"] = {"effort": self.reasoning_effort}
            if self.require_free:
                ensure_free_body(body)
        return body

    def complete(self, messages: Sequence[Message], *, max_tokens: int, temperature: float = 0.0) -> Completion:
        body = self.body(messages, max_tokens=max_tokens, temperature=temperature)
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(self._url, json=body, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise RetryableLLMError(f"timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise RetryableLLMError(f"connection error: {exc}") from exc
        data = _json_or_error(response)
        served = data.get("model")
        if self.require_free:
            ensure_served_free(served if isinstance(served, str) else None)
        choices = data.get("choices") or []
        if not choices:
            raise RetryableLLMError("no choices in the response")
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        finish = choices[0].get("finish_reason")
        if not content.strip():
            if finish == "length":
                raise RetryableLLMError("empty answer: max_tokens reached before any output (hidden reasoning)")
            if finish == "content_filter":
                raise LLMError("the model declined to answer (finish_reason=content_filter)")
            raise RetryableLLMError(f"empty answer (finish_reason={finish})")
        usage = data.get("usage") or {}
        return Completion(
            text=content,
            model=str(served or self.model),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            extra={"finish_reason": finish},
        )


def _error_message(data: dict[str, Any]) -> str:
    error: Any = data.get("error", data)
    if isinstance(error, dict):
        metadata = error.get("metadata")
        raw = metadata.get("raw") if isinstance(metadata, dict) else None
        return str(raw or error.get("message") or error)[:300]
    return str(error)[:300]


def _json_or_error(response: httpx.Response) -> dict[str, Any]:
    status = response.status_code
    try:
        data = response.json()
    except ValueError:
        data = {"error": {"message": response.text[:300]}}
    if not isinstance(data, dict):
        data = {"error": {"message": str(data)[:300]}}
    error = data.get("error")
    if isinstance(error, dict) and status < 400:  # OpenRouter reports some upstream failures inside a 200
        code = error.get("code")
        status = code if isinstance(code, int) and code >= 400 else 502
    if status < 400:
        return data
    message = _error_message(data)
    if status == 429 and "budget" in message.lower():
        raise BudgetExceededError(f"gateway budget exhausted: {message}")
    if status == 429:
        retry_after = response.headers.get("retry-after")
        try:
            delay = float(retry_after) if retry_after else None
        except ValueError:
            delay = None
        raise RetryableLLMError(f"rate limited: {message}", retry_after=delay)
    if status >= 500 or status in {408, 409}:
        raise RetryableLLMError(f"upstream {status}: {message}")
    raise LLMError(f"upstream {status}: {message}")
