"""
clarifier.py — Generates pillar + age-specific clarifying questions
that Kalpana asks BEFORE producing a final answer.

Each turn of clarifying yields a single question. The orchestrator collects
answers across turns and then synthesises the final response.

filter_already_answered() uses the LLM to drop templated questions whose
answers are already obvious from the user's original query / profile, so
Kalpana doesn't ask 'do you have any conditions?' when the user just said
'I have PCOS and am planning a pregnancy'.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("shakti.clarifier")

# ── Templates keyed by (pillar, age_band) ──
CLARIFYING_QUESTION_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "HEALTH": {
        "11-24": [
            "Do you have any health conditions \u2014 anything you take medication for?",
            "Have you talked to a doctor recently about this?",
            "Is this something new, or has it been bothering you for a while?",
        ],
        "25-40": [
            "Do you have any health conditions or take medication regularly?",
            "Have you seen a doctor for this concern? Any recent reports to share?",
            "Which city are you in? So I can point you to local options.",
        ],
        "41+": [
            "Do you have any ongoing conditions \u2014 diabetes, blood pressure, thyroid?",
            "Are you under any doctor's care currently?",
            "Any recent reports or prescriptions you can share?",
        ],
    },
    "FINANCE": {
        "11-24": [
            "Do you currently earn or save anything \u2014 pocket money, part-time work?",
            "Do you already have a bank account?",
            "Are you thinking about a specific goal \u2014 college, gadgets, savings?",
        ],
        "25-40": [
            "What's your current monthly income range, roughly? (just for context)",
            "Any existing investments \u2014 SIPs, FDs, EPF, anything?",
            "Is there a specific goal \u2014 house, child's education, retirement?",
        ],
        "41+": [
            "Do you have existing investments or pension savings?",
            "Is your goal more about preservation or growth right now?",
            "Any dependents you're planning for \u2014 children, parents?",
        ],
    },
    "CAREER": {
        "11-24": [
            "What stage are you at \u2014 school, college, just graduated?",
            "Any specific field you're drawn to, or still exploring?",
            "Are you preparing for any entrance exam or scholarship?",
        ],
        "25-40": [
            "What's your current work situation \u2014 working, break, looking?",
            "How many years of experience do you have?",
            "Are you looking to switch, upskill, or restart after a break?",
        ],
        "41+": [
            "What's your current role and experience?",
            "Are you thinking of switching, consulting, or starting fresh?",
            "Any specific skills or industries you're drawn to?",
        ],
    },
}


# ── Yes/No detection — gives the UI a hint to show quick-reply pills ──
_YES_NO_HINTS = (
    "do you ", "have you ", "are you ", "is this ", "any ", "do they "
)


def is_yes_no(question: str) -> bool:
    """Best-effort: does this clarifying question accept a yes/no answer?"""
    q = (question or "").lower().strip()
    if not q:
        return False
    # Multi-part or "what/which/how" questions are not yes/no
    if any(w in q for w in ("which ", "what ", "how ", "where ")):
        return False
    return any(q.startswith(h) for h in _YES_NO_HINTS)


def get_clarifying_questions(pillar: str, age_band: str, query: str = "") -> list[str]:
    """Return up to 3 clarifying questions for this pillar + age band."""
    pillar = (pillar or "").upper()
    questions = CLARIFYING_QUESTION_TEMPLATES.get(pillar, {}).get(age_band, [])
    return list(questions[:3])


def build_clarification_context(
    original_query: str,
    qa_pairs: list[dict[str, str]],
) -> str:
    """Format the original query + clarifying Q/As into a context block for the LLM."""
    lines = [f"Original question: {original_query}"]
    if qa_pairs:
        lines.append("\nClarifying details from the user:")
        for qa in qa_pairs:
            q = (qa.get("q") or "").strip()
            a = (qa.get("a") or "").strip()
            if not a or a.lower() in ("skip", "skipped", "-"):
                continue
            lines.append(f"  Q: {q}")
            lines.append(f"  A: {a}")
    return "\n".join(lines)


async def _call_gemini(prompt: str, max_tokens: int = 120, temperature: float = 0.2) -> str:
    """Sync Vertex SDK call wrapped in an executor. Returns text or empty."""
    try:
        import vertexai  # type: ignore
        from vertexai.generative_models import GenerativeModel  # type: ignore
    except Exception:
        return ""

    def _do() -> str:
        try:
            vertexai.init(
                project=os.getenv("PROJECT_ID", "shakti360"),
                location=os.getenv("VERTEX_LOCATION", "us-central1"),
            )
            model = GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
            resp = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
            )
            return (resp.text or "").strip()
        except Exception as e:
            logger.warning(f"Clarifier Gemini call failed: {e}")
            return ""

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do)


async def filter_already_answered(
    questions: list[str],
    original_query: str,
    profile: Optional[dict] = None,
) -> list[str]:
    """Drop templated questions whose answers are already obvious from the
    user's original query or profile.

    Returns a list of questions to actually ask. Always returns at least 1
    question when there's something useful left; returns an empty list when
    the user has essentially given full context already.
    On any failure, returns the full input list unchanged.
    """
    if not questions:
        return []
    query = (original_query or "").strip()
    if not query:
        return list(questions)

    profile = profile or {}
    profile_hint = ""
    if profile:
        bits = []
        if profile.get("preferred_name"):
            bits.append(f"name={profile['preferred_name']}")
        if profile.get("age_band"):
            bits.append(f"age_band={profile['age_band']}")
        if profile.get("subject_role"):
            bits.append(f"life_stage={profile['subject_role']}")
        if profile.get("language"):
            bits.append(f"language={profile['language']}")
        if bits:
            profile_hint = "Known profile: " + ", ".join(bits) + "\n"

    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))
    prompt = (
        "You are helping a women's-life-companion AI decide which clarifying "
        "questions are actually still useful to ask.\n\n"
        f"{profile_hint}"
        f"User's original message: \"{query}\"\n\n"
        "Candidate clarifying questions:\n"
        f"{numbered}\n\n"
        "Task: return a JSON array of the INDICES of questions that are STILL "
        "USEFUL to ask \u2014 i.e. the user has NOT already answered them in "
        "their original message and the profile does not already imply the "
        "answer. Drop questions that would feel redundant or annoying.\n\n"
        "Rules:\n"
        "- Output ONLY a JSON array of integers, e.g. [0, 2]\n"
        "- Empty array [] if every question is already answered\n"
        "- Keep at most 3 indices\n"
        "- Preserve the original order"
    )

    raw = await _call_gemini(prompt, max_tokens=60, temperature=0.0)
    if not raw:
        return list(questions[:3])

    # Strip code fences and find the JSON array
    text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return list(questions[:3])

    try:
        indices = json.loads(text[start:end + 1])
    except Exception:
        return list(questions[:3])

    if not isinstance(indices, list):
        return list(questions[:3])

    out = []
    seen: set[int] = set()
    for idx in indices:
        try:
            i = int(idx)
        except Exception:
            continue
        if 0 <= i < len(questions) and i not in seen:
            seen.add(i)
            out.append(questions[i])
        if len(out) >= 3:
            break

    return out


def opening_kalpana_line(pillar: str, name: str = "") -> str:
    """A short warm opener Kalpana says before asking the first clarifying question."""
    pillar = (pillar or "").upper()
    addr = f" {name}" if name else ""
    if pillar == "HEALTH":
        return (
            f"I'm here for you{addr}. Let's take this one step at a time \u2014 "
            "I'll ask a couple of quick questions so I can guide you better."
        )
    if pillar == "FINANCE":
        return (
            f"Got it{addr}. Before I give you any numbers, let me understand your situation "
            "with a couple of quick questions."
        )
    if pillar == "CAREER":
        return (
            f"Lovely{addr}. Let me ask a couple of short questions so I can tailor "
            "this to where you actually are."
        )
    return (
        f"I'm here for you{addr}. Let me ask a couple of quick questions first."
    )
