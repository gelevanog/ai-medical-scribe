"""The free-only guard (on by default) and the paid-model opt-in.

* Every OpenRouter model id (and every fallback) must end in ``:free``; ``:online`` variants are refused (web search is
  a paid plugin and would send the prompt to a search provider), and so is any request body that carries ``plugins``.
* An answer that OpenRouter reports as served by a non-free model is rejected.
* The ``openai_compatible`` provider (any metered endpoint) and paid OpenRouter ids need NOTEWELL_ALLOW_PAID_MODELS=true.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from notewell.llm.base import PolicyViolationError


def ensure_free_models(model_ids: Iterable[str]) -> None:
    ids = [str(m) for m in model_ids]
    online = [m for m in ids if ":online" in m]
    if online:
        raise PolicyViolationError(f"free-only guard: refusing :online model variant(s): {', '.join(online)}")
    paid = [m for m in ids if not m.endswith(":free")]
    if paid:
        raise PolicyViolationError(
            f"free-only guard: refusing non-free OpenRouter model id(s): {', '.join(paid)} "
            "(set NOTEWELL_ALLOW_PAID_MODELS=true and NOTEWELL_REQUIRE_FREE_MODELS=false to allow paid models)"
        )


def ensure_free_body(body: Mapping[str, Any]) -> None:
    if body.get("plugins"):
        raise PolicyViolationError("free-only guard: request plugins (web search, file parsing) are refused")
    ensure_free_models([str(body.get("model", "")), *[str(m) for m in body.get("models", []) or []]])


def ensure_served_free(served_model: str | None) -> None:
    if served_model and not served_model.endswith(":free"):
        raise PolicyViolationError(f"free-only guard: OpenRouter served non-free model {served_model!r}; rejected")


def ensure_paid_allowed(provider: str, model: str, *, allowed: bool) -> None:
    if not allowed:
        raise PolicyViolationError(
            f"paid-model guard: the {provider} provider calls a metered endpoint (model {model!r}); "
            "set NOTEWELL_ALLOW_PAID_MODELS=true to opt in"
        )
