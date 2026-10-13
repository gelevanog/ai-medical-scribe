from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from notewell.config import Settings
from notewell.llm.base import BudgetExceededError, Completion, LLMError, PolicyViolationError, RetryableLLMError
from notewell.llm.factory import build_chat_model, resilient
from notewell.llm.fake import FakeScribe
from notewell.llm.guards import ensure_free_body, ensure_free_models, ensure_paid_allowed, ensure_served_free
from notewell.llm.openai_compat import OpenAICompatibleModel
from notewell.llm.resilient import CallLedger, DiskCache, ResilientModel

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]


def _key() -> str:
    return "sk-or-" + "test" * 4  # built at runtime: no key-shaped literal in the repository


def test_free_guard_accepts_free_ids() -> None:
    ensure_free_models(["nvidia/nemotron-3-super-120b-a12b:free", "google/gemma-4-31b-it:free"])


@pytest.mark.parametrize("model", ["openai/gpt-5", "anthropic/claude-sonnet-5", "x/y:free:online", "x/y:online", "x/y"])
def test_free_guard_refuses_paid_and_online(model: str) -> None:
    with pytest.raises(PolicyViolationError):
        ensure_free_models([model])


def test_free_guard_checks_fallbacks_and_plugins() -> None:
    with pytest.raises(PolicyViolationError):
        ensure_free_body({"model": "a/b:free", "models": ["a/b:free", "c/d"]})
    with pytest.raises(PolicyViolationError, match="plugins"):
        ensure_free_body({"model": "a/b:free", "plugins": [{"id": "web"}]})


def test_served_model_must_be_free() -> None:
    ensure_served_free("a/b:free")
    ensure_served_free(None)
    with pytest.raises(PolicyViolationError):
        ensure_served_free("a/b")


def test_paid_opt_in() -> None:
    ensure_paid_allowed("openai_compatible", "m", allowed=True)
    with pytest.raises(PolicyViolationError, match="ALLOW_PAID_MODELS"):
        ensure_paid_allowed("openai_compatible", "m", allowed=False)


def test_factory_refuses_before_constructing(tmp_path: Path) -> None:
    s = Settings(llm_provider="openrouter", llm_model="openai/gpt-5", openrouter_api_key=None, llm_ledger=None)
    with pytest.raises(PolicyViolationError):  # refused by the guard, not by the missing key
        build_chat_model(s)
    s2 = Settings(llm_provider="openai_compatible", llm_model="claude-haiku", llm_base_url="http://x", llm_ledger=None)
    with pytest.raises(PolicyViolationError):
        build_chat_model(s2)
    s3 = s2.model_copy(update={"allow_paid_models": True})
    assert build_chat_model(s3).label == "openai_compatible/claude-haiku"


def test_factory_fake() -> None:
    model = build_chat_model(Settings(llm_provider="fake", llm_ledger=None))
    assert model.is_local and model.label == "fake/fake-scribe"


def _transport(handler) -> httpx.MockTransport:  # type: ignore[no-untyped-def]
    return httpx.MockTransport(handler)


def _model(handler, **kw) -> OpenAICompatibleModel:  # type: ignore[no-untyped-def]
    return OpenAICompatibleModel(kind="openrouter", base_url="https://openrouter.test/api/v1", model="a/b:free",
                                 api_key=_key(), transport=_transport(handler), **kw)  # fmt: skip


def test_openrouter_body_and_answer() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"model": "a/b:free", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 2}})  # fmt: skip

    c = _model(handler, reasoning_effort="none", fallback_models=["c/d:free"]).complete(MSGS, max_tokens=50)  # type: ignore[arg-type]
    assert c.text == "{}" and c.input_tokens == 10
    assert seen["reasoning"] == {"enabled": False}
    assert seen["models"] == ["a/b:free", "c/d:free"]
    assert str(seen["auth"]).startswith("Bearer ")


def test_openrouter_rejects_paid_served_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "a/b", "choices": [{"message": {"content": "x"}}]})

    with pytest.raises(PolicyViolationError):
        _model(handler).complete(MSGS, max_tokens=5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status", "payload", "error"),
    [
        (429, {"error": {"message": "rate limited"}}, RetryableLLMError),
        (503, {"error": {"message": "overloaded"}}, RetryableLLMError),
        (400, {"error": {"message": "bad"}}, LLMError),
        (429, {"error": {"message": "budget exhausted"}}, BudgetExceededError),
        (200, {"error": {"code": 502, "message": "upstream"}}, RetryableLLMError),
    ],
)
def test_error_mapping(status: int, payload: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        _model(lambda r: httpx.Response(status, json=payload)).complete(MSGS, max_tokens=5)  # type: ignore[arg-type]


def test_empty_answer_after_hidden_reasoning_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"model": "a/b:free", "choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        )

    with pytest.raises(RetryableLLMError, match="max_tokens"):
        _model(handler).complete(MSGS, max_tokens=5)  # type: ignore[arg-type]


class Flaky:
    label = "openrouter/a/b:free"
    is_local = False

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def complete(self, messages, *, max_tokens, temperature=0.0):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls <= self.failures:
            raise RetryableLLMError("rate limited", retry_after=0.01)
        return Completion(text="ok", model="a/b:free", input_tokens=3, output_tokens=1)


def test_retries_ledger_and_cache(tmp_path: Path) -> None:
    ledger = CallLedger(tmp_path / "calls.jsonl", max_calls=10)
    inner = Flaky(failures=2)
    model = ResilientModel(
        inner, ledger=ledger, cache=DiskCache(tmp_path / "cache"), max_retries=3, retry_base_seconds=0.01, tag="t"
    )
    assert model.complete(MSGS, max_tokens=5).text == "ok"  # type: ignore[arg-type]
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["retryable_error", "retryable_error", "ok"]
    assert all("hello" not in json.dumps(r) for r in rows)  # the ledger never stores prompts
    assert rows[-1]["payload_sha256"]
    again = model.complete(MSGS, max_tokens=5)  # type: ignore[arg-type]
    assert again.cached and inner.calls == 3


def test_budget_is_enforced(tmp_path: Path) -> None:
    ledger = CallLedger(tmp_path / "calls.jsonl", max_calls=1)
    model = ResilientModel(Flaky(failures=5), ledger=ledger, max_retries=3, retry_base_seconds=0.01)
    with pytest.raises(BudgetExceededError):
        model.complete(MSGS, max_tokens=5)  # type: ignore[arg-type]


def test_egress_check_runs_before_sending_even_for_the_fake() -> None:
    sent: list[object] = []

    def refuse(messages):  # type: ignore[no-untyped-def]
        raise PolicyViolationError("egress check: identifiers found")

    model = ResilientModel(FakeScribe(), egress_check=refuse, observer=lambda label, m: sent.append(m))
    with pytest.raises(PolicyViolationError):
        model.complete(MSGS, max_tokens=5)  # type: ignore[arg-type]
    assert sent == []


def test_resilient_wrapper_from_settings(tmp_path: Path) -> None:
    s = Settings(llm_provider="fake", llm_ledger=tmp_path / "c.jsonl")
    model = resilient(build_chat_model(s), s)
    assert model.ledger is None  # the fake is local: no ledger rows
