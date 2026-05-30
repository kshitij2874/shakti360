"""
safety.py — Crisis detection that runs BEFORE the normal chat flow.

A trusted companion must never run a clarifying-question checklist on someone
in danger. detect_crisis() screens every incoming message; when it matches a
crisis signal it returns a warm, immediate response with the right Indian
helplines, bypassing classification and clarification entirely.

Design:
  1. Fast keyword screen (zero added latency for normal messages — most have
     no crisis words at all, so we return None instantly).
  2. DEFINITE patterns trigger immediately, even if the LLM is down.
  3. AMBIGUOUS patterns run a single cheap LLM confirmation to cut false
     positives ("this traffic is killing me").
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger("shakti.safety")


# ── Verified Indian helplines, per crisis category ──────────────────────────
HELPLINES_BY_CATEGORY: dict[str, list[dict[str, str]]] = {
    "self_harm": [
        {"name": "Tele-MANAS (Govt of India)", "number": "14416", "hours": "24/7"},
        {"name": "Vandrevala Foundation", "number": "1860-2662-2345", "hours": "24/7"},
        {"name": "iCall (TISS)", "number": "9152987821", "hours": "Mon–Sat, 8am–10pm"},
        {"name": "AASRA", "number": "9820466726", "hours": "24/7"},
    ],
    "abuse": [
        {"name": "Women's Helpline", "number": "181", "hours": "24/7"},
        {"name": "National Commission for Women", "number": "7827-170-170", "hours": "24/7"},
        {"name": "Police Emergency", "number": "112", "hours": "24/7"},
        {"name": "Domestic Violence (NCW WhatsApp)", "number": "7827-170-170", "hours": "24/7"},
    ],
    "assault": [
        {"name": "Women in Distress", "number": "1091", "hours": "24/7"},
        {"name": "Women's Helpline", "number": "181", "hours": "24/7"},
        {"name": "Police Emergency", "number": "112", "hours": "24/7"},
    ],
    "emergency": [
        {"name": "Ambulance", "number": "108", "hours": "24/7"},
        {"name": "National Emergency", "number": "112", "hours": "24/7"},
    ],
}


# ── DEFINITE patterns — unambiguous, trigger immediately (no LLM) ────────────
_DEFINITE: dict[str, list[str]] = {
    "self_harm": [
        r"\bkill myself\b", r"\bend my life\b", r"\btake my (own )?life\b",
        r"\bwant to die\b", r"\bsuicid(e|al)\b", r"\bself[- ]?harm\b",
        r"\bcut myself\b", r"\bno reason to live\b", r"\bdon'?t want to live\b",
        r"\bharm myself\b", r"\boverdose\b", r"\bend it all\b",
    ],
    "abuse": [
        r"\b(he|husband|partner|father|in[- ]?laws?) (hits|beats|hit|beat|abuses|abused) (me|her)\b",
        r"\bbeing (abused|beaten|hit)\b", r"\bdomestic violence\b",
        r"\bhe threatens (to kill|me)\b", r"\bafraid of my husband\b",
        r"\bbeats me\b", r"\bhits me\b",
    ],
    "assault": [
        r"\b(raped|molest(ed|ing)?|sexual(ly)? assault(ed)?)\b",
        r"\bforced me to\b.*\b(sex|touch)\b",
    ],
    "emergency": [
        r"\bbleeding heavily\b", r"\bcan'?t breathe\b", r"\bchest pain\b.*\b(now|severe)\b",
        r"\bunconscious\b", r"\boverdosed\b",
    ],
}

# ── AMBIGUOUS — run LLM confirmation before triggering ───────────────────────
_AMBIGUOUS = [
    r"\bdepress(ed|ion)\b", r"\bhopeless\b", r"\bcan'?t go on\b",
    r"\bgive up\b", r"\bhurt(ing)? me\b", r"\bscared\b.*\bhome\b",
    r"\bnot safe\b", r"\bhe controls\b", r"\btrapped\b",
]

_DEFINITE_COMPILED = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in _DEFINITE.items()
}
_AMBIGUOUS_COMPILED = [re.compile(p, re.IGNORECASE) for p in _AMBIGUOUS]


def _build_response(category: str) -> dict[str, Any]:
    """Warm, validating message + tappable helplines for a crisis category."""
    helplines = HELPLINES_BY_CATEGORY.get(category, HELPLINES_BY_CATEGORY["emergency"])

    intros = {
        "self_harm": (
            "I'm really glad you told me this. What you're feeling right now is heavy, "
            "and you don't have to carry it alone. Please reach out to someone who can "
            "stay with you through this — these people are trained to listen, any time of day:"
        ),
        "abuse": (
            "Thank you for trusting me with this. No one deserves to be hurt, and what's "
            "happening is not your fault. You have options, and there are people ready to "
            "help you stay safe — confidentially:"
        ),
        "assault": (
            "I'm so sorry this happened to you. It was not your fault. You deserve support "
            "and safety right now. These services are confidential and available any time:"
        ),
        "emergency": (
            "This sounds urgent. Please get medical help right away — call one of these now, "
            "or ask someone near you to call:"
        ),
    }
    closers = {
        "self_harm": "If you're in immediate danger, please call 112. I'm here too — you can keep talking to me.",
        "abuse": "If you're in immediate danger, call 112. When you're ready, I can help you find a nearby safe place or a women's shelter.",
        "assault": "If you're in immediate danger, call 112. I can also help you find the nearest hospital or police women's help desk when you're ready.",
        "emergency": "If it's life-threatening, call 112 immediately.",
    }

    lines = [intros.get(category, intros["emergency"]), ""]
    for h in helplines:
        lines.append(f"📞 {h['name']}: {h['number']}  ({h['hours']})")
    lines.append("")
    lines.append(closers.get(category, closers["emergency"]))

    return {
        "type": "crisis",
        "category": category,
        "is_crisis": True,
        "response": "\n".join(lines),
        "helplines": helplines,
        "pillar": "SAFETY",
        "citations": [],
        "citation_chips": [],
        "next_steps": [],
        "tools_to_call": [],
        "awaiting_approval": False,
        "from_cache": False,
    }


async def _llm_confirm(query: str) -> Optional[str]:
    """Ask the LLM whether an ambiguous message is a genuine crisis. Returns a
    category string or None. Fails safe to None on any error."""
    try:
        from llm import call_llm
        prompt = (
            "A user wrote this message to a women's support companion:\n"
            f"\"{query}\"\n\n"
            "Is this person expressing a genuine crisis right now? Choose ONE:\n"
            "- self_harm  (suicidal thoughts, wanting to harm themselves)\n"
            "- abuse      (domestic violence, being hurt/controlled/threatened by someone)\n"
            "- assault    (sexual assault)\n"
            "- emergency  (urgent medical danger)\n"
            "- none       (not a crisis — venting, figure of speech, or general question)\n\n"
            "Reply with ONLY one word."
        )
        raw = (await call_llm(prompt, max_tokens=5, temperature=0.0)).strip().lower()
        for cat in ("self_harm", "abuse", "assault", "emergency"):
            if cat in raw:
                return cat
        return None
    except Exception as e:
        logger.warning(f"Crisis LLM confirmation failed: {e}")
        return None


async def detect_crisis(query: str) -> Optional[dict[str, Any]]:
    """
    Screen a message for crisis signals. Returns a ready-to-send response dict
    if a crisis is detected, else None (so the normal flow continues).
    """
    if not query or not query.strip():
        return None

    # 1. DEFINITE — immediate, no LLM
    for category, patterns in _DEFINITE_COMPILED.items():
        for pat in patterns:
            if pat.search(query):
                logger.warning(f"CRISIS detected (definite/{category}): '{query[:60]}'")
                return _build_response(category)

    # 2. AMBIGUOUS — only pay for an LLM call when a soft signal is present
    if any(pat.search(query) for pat in _AMBIGUOUS_COMPILED):
        category = await _llm_confirm(query)
        if category:
            logger.warning(f"CRISIS detected (llm/{category}): '{query[:60]}'")
            return _build_response(category)

    return None
