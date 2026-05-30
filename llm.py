"""
llm.py — Unified LLM client for ShaktiAgent.
Primary: DeepSeek (deepseek-chat via OpenAI-compatible API).
Fallback: Google Gemini via Vertex AI.

Set DEEPSEEK_API_KEY to enable DeepSeek. If missing or if DeepSeek
errors, every call falls back to GEMINI_MODEL automatically.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger("shakti.llm")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


async def call_llm(
    prompt: str,
    max_tokens: int = 1200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model_name: Optional[str] = None,  # Gemini model override used only in fallback
) -> str:
    """
    Primary: DeepSeek. Falls back to Gemini if key is absent or call fails.
    """
    if os.getenv("DEEPSEEK_API_KEY", ""):
        try:
            return await _call_deepseek(prompt, max_tokens, temperature, top_p)
        except Exception as e:
            logger.warning(f"DeepSeek call failed ({e}) — falling back to Gemini")

    return await _call_gemini(prompt, max_tokens, temperature, top_p, model_name)


# ── DeepSeek ──────────────────────────────────────────────────────────────────

async def _call_deepseek(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed — run: pip install openai")

    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
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

    def _do_call() -> str:
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
        return (resp.text or "").strip()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_call)
