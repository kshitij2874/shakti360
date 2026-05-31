"""
llm.py — Unified, tiered LLM client for ShaktiAgent.

Two tiers:
  - "fast"      → DeepSeek V4 Flash in NON-thinking mode for routing,
                  classification, clarifying questions, follow-up detection,
                  next-steps, memory, greeting. Quick + economical.
  - "reasoning" → DeepSeek V4 Pro in THINKING mode for the main answer and
                  difficult situations. Highest quality.

Primary provider is DeepSeek; falls back to Google Gemini (Vertex AI) if the
DeepSeek key is missing or a call fails.

Model IDs are env-configurable:
  DEEPSEEK_MODEL_FAST       (default "deepseek-v4-flash")
  DEEPSEEK_MODEL_REASONING  (default "deepseek-v4-pro")
Gemini fallback tiers:
  GEMINI_MODEL              (default "gemini-2.5-flash")
  GEMINI_MODEL_REASONING    (default = GEMINI_MODEL)

DeepSeek V4 notes (https://api-docs.deepseek.com/guides/thinking_mode):
  - Thinking mode toggled via extra_body={"thinking": {"type": ...}}.
  - Thinking mode ignores temperature/top_p and returns chain-of-thought in
    reasoning_content; the final answer is still in message.content.
  - Legacy deepseek-chat / deepseek-reasoner deprecate 2026/07/24.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger("shakti.llm")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Backwards-compatible single var still respected as the fast default.
_DEEPSEEK_DEFAULT = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MODEL_FAST = os.getenv("DEEPSEEK_MODEL_FAST", _DEEPSEEK_DEFAULT)
DEEPSEEK_MODEL_REASONING = os.getenv("DEEPSEEK_MODEL_REASONING", "deepseek-v4-pro")

# reasoning_effort sent when thinking is enabled ("high" | "max")
DEEPSEEK_REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "high")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL_REASONING = os.getenv("GEMINI_MODEL_REASONING", GEMINI_MODEL)


def _deepseek_model_for(tier: str) -> str:
    return DEEPSEEK_MODEL_REASONING if tier == "reasoning" else DEEPSEEK_MODEL_FAST


def _gemini_model_for(tier: str) -> str:
    return GEMINI_MODEL_REASONING if tier == "reasoning" else GEMINI_MODEL


async def call_llm(
    prompt: str,
    max_tokens: int = 1200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model_name: Optional[str] = None,  # explicit Gemini override (fallback only)
    tier: str = "fast",                # "fast" | "reasoning"
) -> str:
    """
    Route to the tier's model. Primary DeepSeek, Gemini fallback on missing key/error.
    """
    if os.getenv("DEEPSEEK_API_KEY", ""):
        try:
            return await _call_deepseek(prompt, max_tokens, temperature, top_p, tier)
        except Exception as e:
            logger.warning(f"DeepSeek ({tier}) call failed ({e}) — falling back to Gemini")

    gemini_name = model_name or _gemini_model_for(tier)
    return await _call_gemini(prompt, max_tokens, temperature, top_p, gemini_name)


# ── DeepSeek ──────────────────────────────────────────────────────────────────

async def _call_deepseek(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    tier: str = "fast",
) -> str:
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed — run: pip install openai")

    model = _deepseek_model_for(tier)
    thinking = (tier == "reasoning")

    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )

    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    if thinking:
        # Thinking mode: no sampling params; effort + thinking flag via extra_body.
        kwargs["extra_body"] = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": DEEPSEEK_REASONING_EFFORT,
        }
    else:
        # Non-thinking mode: fast, supports sampling params.
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    response = await client.chat.completions.create(**kwargs)

    # Record token usage + cost (cache-hit tokens billed cheaper when reported)
    try:
        from usage import record_usage
        u = response.usage
        if u:
            extra = getattr(u, "model_extra", None) or {}
            cached = (
                getattr(u, "prompt_cache_hit_tokens", None)
                or extra.get("prompt_cache_hit_tokens", 0)
                or 0
            )
            record_usage(model, u.prompt_tokens or 0, u.completion_tokens or 0, cached)
    except Exception as e:
        logger.debug(f"usage record (deepseek) skipped: {e}")

    # Final answer is always in content (reasoning_content holds the CoT, ignored here).
    return (response.choices[0].message.content or "").strip()


# ── Gemini (Vertex AI) fallback ───────────────────────────────────────────────

async def _call_gemini(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model_name: Optional[str] = None,
) -> str:
    try:
        from vertexai.generative_models import (  # type: ignore
            GenerativeModel,
            SafetySetting,
            HarmCategory,
            HarmBlockThreshold,
        )
    except ImportError as e:
        raise RuntimeError(f"Vertex AI SDK unavailable: {e}")

    name = model_name or GEMINI_MODEL

    # Disable safety filters so benign health/finance queries aren't blocked
    safety_settings = [
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold=HarmBlockThreshold.BLOCK_NONE),
    ]

    def _do_call():
        model = GenerativeModel(name)
        resp = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            safety_settings=safety_settings,
        )
        meta = getattr(resp, "usage_metadata", None)
        usage = None
        if meta is not None:
            usage = (
                getattr(meta, "prompt_token_count", 0) or 0,
                getattr(meta, "candidates_token_count", 0) or 0,
                getattr(meta, "cached_content_token_count", 0) or 0,
            )
        return (resp.text or "").strip(), usage

    loop = asyncio.get_event_loop()
    text, usage = await loop.run_in_executor(None, _do_call)

    # Record usage in the async context (ContextVars don't cross into executor threads)
    if usage:
        try:
            from usage import record_usage
            record_usage(name, usage[0], usage[1], usage[2])
        except Exception as e:
            logger.debug(f"usage record (gemini) skipped: {e}")

    return text
