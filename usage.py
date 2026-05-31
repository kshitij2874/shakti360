"""
usage.py — Per-request token + cost accounting.

call_llm() records every model call into a request-scoped accumulator (a
ContextVar, so concurrent requests don't mix). The orchestrator reads the
total at the end of a turn and reports it to the dashboard metrics and Langfuse.

Pricing is USD per 1M tokens, env-overridable. Defaults reflect DeepSeek V4
public pricing (cache-miss input / cached input / output) as of 2026-05:
  V4 Flash: $0.14 in / $0.0028 cached / $0.28 out
  V4 Pro:   $0.435 in / $0.003625 cached / $0.87 out
DeepSeek prompt caching makes repeated prefixes ~50x cheaper on input, so we
use cache-hit token counts when the API reports them.
"""
from __future__ import annotations

import contextvars
import logging
import os

logger = logging.getLogger("shakti.usage")

_usage_ctx: contextvars.ContextVar = contextvars.ContextVar("shakti_usage", default=None)


def _price(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, default))
    except (TypeError, ValueError):
        return default


# USD per 1M tokens. {in_miss, in_hit, out}
PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {
        "in_miss": _price("PRICE_V4FLASH_IN", 0.14),
        "in_hit":  _price("PRICE_V4FLASH_IN_CACHED", 0.0028),
        "out":     _price("PRICE_V4FLASH_OUT", 0.28),
    },
    "deepseek-v4-pro": {
        "in_miss": _price("PRICE_V4PRO_IN", 0.435),
        "in_hit":  _price("PRICE_V4PRO_IN_CACHED", 0.003625),
        "out":     _price("PRICE_V4PRO_OUT", 0.87),
    },
    "gemini-2.5-flash": {
        "in_miss": _price("PRICE_GEMINI_FLASH_IN", 0.15),
        "in_hit":  _price("PRICE_GEMINI_FLASH_IN", 0.15),
        "out":     _price("PRICE_GEMINI_FLASH_OUT", 0.60),
    },
}
_DEFAULT_PRICE = {"in_miss": 0.20, "in_hit": 0.20, "out": 0.60}


def _pricing_for(model: str) -> dict[str, float]:
    m = (model or "").lower()
    for key, price in PRICING.items():
        if key in m:
            return price
    return _DEFAULT_PRICE


def _blank() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "calls": 0,
        "by_model": {},
    }


def start_usage() -> None:
    """Reset the accumulator for the current request."""
    _usage_ctx.set(_blank())


def record_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> None:
    """Add one model call's usage to the request accumulator and compute cost."""
    u = _usage_ctx.get()
    if u is None:
        return  # tracking not started for this request — ignore

    price = _pricing_for(model)
    miss = max(prompt_tokens - cached_tokens, 0)
    cost = (
        cached_tokens * price["in_hit"]
        + miss * price["in_miss"]
        + completion_tokens * price["out"]
    ) / 1_000_000

    u["prompt_tokens"] += prompt_tokens
    u["completion_tokens"] += completion_tokens
    u["cached_tokens"] += cached_tokens
    u["total_tokens"] += prompt_tokens + completion_tokens
    u["cost_usd"] += cost
    u["calls"] += 1

    bm = u["by_model"].setdefault(model, {"tokens": 0, "cost_usd": 0.0, "calls": 0})
    bm["tokens"] += prompt_tokens + completion_tokens
    bm["cost_usd"] += cost
    bm["calls"] += 1


def get_usage() -> dict:
    """Return the current request's accumulated usage (zeros if not started)."""
    u = _usage_ctx.get()
    return dict(u) if u else _blank()
